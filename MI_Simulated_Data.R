library(rstan)
library(ggplot2)
library(plotly)
library(igraph)
library(bnlearn)
library(BNSL)

# Number of Bernoulli and normal variables
n1 <- 75
n2 <-76


# Number of samples
N <- 500

# Generate synthetic data
set.seed(123)

# For Bernoulli variables
p_true <- runif(n1, 0, 1)
binomial_matrix <- matrix(NA, N, n1)
for (k in 1:n1) {
  binomial_matrix[, k] <- rbinom(N, 1, p_true[k])
}

# For normal variables
mu_true <- rnorm(n2, 0, 1)
sigma_true <- rep(1, n2)
normal_matrix <- matrix(NA, N, n2)
for (l in 1:n2) {
  normal_matrix[, l] <- rnorm(N, mu_true[l], sigma_true[l])
}



# Combine data into a list
data_list <- list(N = N, num_binomial = n1, num_normal = n2, y1 = binomial_matrix, y2 = t(normal_matrix))

# Compile the Stan model
fit <-stan(file="Clayton_Copula.stan", chains = 4, data=data_list)



# Function to calculate free energy
free_energy <- function(log_likelihood) {
  return(-mean(rowSums(log_likelihood)))
}


log_lik_pairwise <- extract(fit, 'log_lik_pairwise')$log_lik_pairwise
log_lik_bernoulli <- extract(fit, 'log_lik_bernoulli')$log_lik_bernoulli
log_lik_normal <- extract(fit, 'log_lik_normal')$log_lik_normal

# Initialize lists to store free energy values
free_energy_bernoulli <- list()
free_energy_normal <- list()
free_energy_pairwise <- list()


# Calculate free energy for each Bernoulli variable
for (i in 1:dim(log_lik_bernoulli)[2]) {
  free_energy_bernoulli[paste0("x", i)] <- free_energy(log_lik_bernoulli[, i, ])
}

# Calculate free energy for each Normal variable
for (i in 1:dim(log_lik_normal)[2]) {
  free_energy_normal[paste0("y", i)] <- free_energy(log_lik_normal[, i, ])
}

# Calculate free energy for each pair
for (i in 1:dim(log_lik_pairwise)[2]) {
  pair_key <- paste0("Pair", i)
  free_energy_pairwise[pair_key] <- free_energy(log_lik_pairwise[, i, ])
}


# Initialize the J matrix for free energy
mi_bernoulli_normal <- matrix(0, nrow = length(free_energy_bernoulli), ncol = length(free_energy_normal))

# Populate the J matrix with free energy
for (i in 1:length(free_energy_bernoulli)) {
  for (j in 1:length(free_energy_normal)) {
    # Extract free energy values for the i-th Bernoulli variable, j-th Normal variable, and (i, j)-th pair
    fe_b <- free_energy_bernoulli[[paste0("x", i)]]
    fe_n <- free_energy_normal[[paste0("y", j)]]
    fe_p <- free_energy_pairwise[[paste0("Pair", (i - 1) * length(free_energy_normal) + j)]]
    
    # Calculate J_ij using the formula
    mi_bernoulli_normal[i, j] <- (1 / N) * (fe_b + fe_n - fe_p)
  }
}

# MI Matrix for Bernoulli


f_1=function(x,m){
  n=length(x); cc=array(0,dim=m)
  S=0
  for(i in 1:n){
    S=S-(1/n)*log((cc[x[i]]+0.5)/(i-1+0.5*m)); cc[x[i]]=cc[x[i]]+1
  }
  return(S) }
f_2=function(x,y,m.1,m.2){
  n=length(x); cc=array(0,dim=c(m.1,m.2))
  S=0
  for(i in 1:n){
    S=S-(1/n)*log((cc[x[i],y[i]]+0.5)/(i-1+0.5*m.1*m.2)); cc[x[i],y[i]]=cc[x[i],y[i]]+1
  }
  return(S)
}
multiplication=
  function(n,prob){
    x=runif(n); y=array(dim=n); m=length(prob)
    for(i in 1:n)for(j in 1:m){if(x[i] < prob[j]){y[i]=j; break}; x[i]=x[i]-prob[j]}
    return(y)
  }
r=100; n=1000
S=T=NULL
for(k in 1:r){
  x=multiplication(n,c(3/4,1/4)); y=multiplication(n,c(1/2,1/2))
  S=c(S, f_1(x,2)+f_1(y,2)-f_2(x,y,2,2))
}
T=NULL
for(i in 1:r){
  x=multiplication(n,c(3/4,1/4)); z=(x+multiplication(n,c(1/10,9/10)))%%2+1
  T=c(T,f_1(x,2)+f_1(z,2)-f_2(x,z,2,2))
}
MI3=function(x,y) (1/n)*(max(f_1(x,2)+f_1(y,2)-f_2(x,y,2,2),0))
n=nrow(binomial_matrix)
p=ncol(binomial_matrix)
x=matrix(nrow=n,ncol=p)
for(i in 1:p)x[,i]=binomial_matrix[[i]]
mi_bernoulli_bernoulli=matrix(nrow=p,ncol=p)
for(i in 1:(p-1))for(j in (i + 1):p) mi_bernoulli_bernoulli[i,j] = MI3(x[,i], x[,j])





# MI matrix for normal 

library(lg) 

l_lik=function(x){
  if(ncol(as.matrix(x))==1) dlg_marginal(x, eval_points=x)$f_est else dlg(lg_main(x),grid=x)$f_est
}

MI2=function(x,y){
  z=cbind(x,y)
  n=nrow(z)
  value=sum(log(l_lik(z)))/n-sum(log(l_lik(x)))/n-sum(log(l_lik(y)))/n
  return(value)
}


mat=as.matrix(normal_matrix)
p=ncol(mat); mi_normal_normal=matrix(nrow=p,ncol=p)
for(i in 1:(p-1))for(j in (i+1):p) mi_normal_normal[i,j]=MI2(mat[,i], mat[,j])

# Final MI for the mixture of Bernoulli and normal mixture

# Get dimensions
n_bernoulli <- dim(mi_bernoulli_bernoulli)[1]
n_normal <- dim(mi_normal_normal)[1]

# Initialize the combined MI matrix
n_total <- n_bernoulli + n_normal
mi_mix <- matrix(0, nrow=n_total, ncol=n_total)

# Fill in the blocks
mi_mix[1:n_bernoulli, 1:n_bernoulli] <- mi_bernoulli_bernoulli
mi_mix[(n_bernoulli + 1):n_total, (n_bernoulli + 1):n_total] <- mi_normal_normal
mi_mix[1:n_bernoulli, (n_bernoulli + 1):n_total] <- mi_bernoulli_normal
mi_mix[(n_bernoulli + 1):n_total, 1:n_bernoulli] <- t(mi_bernoulli_normal) # transpose



# Generate new row and column names
bernoulli_indices <- as.character(1:n_bernoulli)
normal_indices <- as.character((n_bernoulli + 1):(n_bernoulli + n_normal))
combined_indices <- c(bernoulli_indices, normal_indices)

# Assign the new names to the matrix
colnames(mi_mix) <- combined_indices
rownames(mi_mix) <- combined_indices

# Set lower triangular part to NA
mi_mix[lower.tri(mi_combined, diag = FALSE)] <- NA

# Set negative terms to 0
mi_mix[mi_combined <= 0] <- 0

# Print or return the combined MI matrix
print(mi_mix)


# Diagnosis

par(mfrow=c(2,2))
stan_diag(fit)
stan_rhat(fit, bins=100)
stan_ess(fit, bins=100)
