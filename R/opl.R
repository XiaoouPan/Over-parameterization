rm(list = ls())

library(glmnet)
Rcpp::sourceCpp('src/opl.cpp')

n = 200
s = 10
p = 2000
sig = 2
beta = c(rep(sig, s), rep(0, p - s))
beta0 = 1

X = matrix(rnorm(n * p), n, p)
err = rnorm(n)
Y = beta0 + X %*% beta + err

fit = linearReg(X, Y, alpha = 0.1)
fit$coeff
