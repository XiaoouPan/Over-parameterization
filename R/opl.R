rm(list = ls())

library(glmnet)
library(MASS)
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

fit = linearReg(X, Y, alpha = 0.01)
beta.hat = fit$coeff[-1]
beta.hat[1:s]
max(abs(beta.hat[-(1:s)]))
norm(beta.hat - beta, "2")

fit.huber = huberReg(X, Y, alpha = 0.01)
beta.hat.huber = fit.huber$coeff[-1]
beta.hat.huber[1:s]
max(abs(beta.hat.huber[-(1:s)]))
norm(beta.hat.huber - beta, "2")


### glmnet
fit.lasso = cv.glmnet(X, Y)
beta.lasso = coef(fit.lasso, s = "lambda.min")[-1]
beta.lasso [1:s]
max(abs(beta.lasso [-(1:s)]))
norm(beta.lasso  - beta, "2")

