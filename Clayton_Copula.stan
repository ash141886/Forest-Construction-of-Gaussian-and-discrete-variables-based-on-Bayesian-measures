functions {
  real clayton_copula_density(real u, real v, real theta) {
    return pow(pow(u, -theta) + pow(v, -theta) - 1, -1/theta - 1) * pow(u, -theta-1) * pow(v, -theta-1);
  }
}

data {
  int<lower=0> N;
  int<lower=0> num_binomial;
  int<lower=0> num_normal;
  int y1[N, num_binomial];
  real y2[num_normal, N];
}

parameters {
  real<lower=0, upper=1> p[num_binomial];
  real mu[num_normal];
  real<lower=0> sigma[num_normal];
  real<lower=0> theta[num_binomial, num_normal];  // Changed rho to theta with lower bound 0
}

model {
  for (k in 1:num_binomial) {
    p[k] ~ beta(1, 1);
  }
  for (l in 1:num_normal) {
    mu[l] ~ normal(0, 10);
    sigma[l] ~ cauchy(0, 5);
  }
  for (k in 1:num_binomial) {
    for (l in 1:num_normal) {
      theta[k, l] ~ exponential(1);  
    }
  }
  
  for (n in 1:N) {
    for (k in 1:num_binomial) {
      real u = bernoulli_lpmf(y1[n, k] | p[k]);
      target += 1/log(N)*u;  // Removed 1/log(N)
      for (l in 1:num_normal) {
        real v = normal_lpdf(y2[l, n] | mu[l], sigma[l]);
        target += 1/log(N)*v;  // Removed 1/log(N)
        target +=1/log(N)*(log(clayton_copula_density(Phi(inv_logit(u)), Phi(inv_logit(v)), theta[k, l])) + u + v); 
      }
    }
  }
}

generated quantities {
  real log_lik_bernoulli[num_binomial, N];
  real log_lik_normal[num_normal, N];
  real log_lik_pairwise[num_binomial * num_normal, N];

  for (n in 1:N) {
    for (k in 1:num_binomial) {
      real u = bernoulli_lpmf(y1[n, k] | p[k]);
      log_lik_bernoulli[k][n] = u;
    }
    for (l in 1:num_normal) {
      real v = normal_lpdf(y2[l, n] | mu[l], sigma[l]);
      log_lik_normal[l][n] = v;
    }
    for (k in 1:num_binomial) {
      for (l in 1:num_normal) {
        real u = bernoulli_lpmf(y1[n, k] | p[k]);
        real v = normal_lpdf(y2[l, n] | mu[l], sigma[l]);
        log_lik_pairwise[(k - 1) * num_normal + l, n] = log(clayton_copula_density(Phi(inv_logit(u)), Phi(inv_logit(v)), theta[k, l])) + u + v;
      }
    }
  }
}
