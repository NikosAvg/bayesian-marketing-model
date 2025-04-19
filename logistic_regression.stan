data {
    int<lower=0> N;
    int<lower=0> K;
    matrix[N, K] X;
    array[N] int<lower=0, upper=1> y;
}
parameters {
    vector[K] beta;
    real alpha;
}
model {
    beta ~ normal(0,1);
    alpha ~ normal(0,1);
    y ~ bernoulli_logit(alpha + X * beta);
}