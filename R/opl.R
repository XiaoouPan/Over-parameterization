rm(list = ls())

library(glmnet)
library(matrixStats)
library(MASS)
Rcpp::sourceCpp('src/opl.cpp')


linear.opl = function(X, Y, alpha, beta, s, eta = 0.01, tol = 0.0001, iteMax = 5000) {
  n = nrow(X)
  p = ncol(X)
  Z = cbind(1, X)
  g = rep(alpha, p + 1)
  l = rep(alpha, p + 1)
  res = Z %*% (g * l) - Y
  err.all = rep(0, iteMax)
  err.s = rep(0, iteMax)
  err.sc = rep(0, iteMax)
  for (i in 1:iteMax) {
    g = g - eta * l * (t(Z) %*% res) / n
    l = l - eta * g * (t(Z) %*% res) / n
    res = Z %*% (g * l) - Y
    beta.hat = (g * l)[-1]
    err.all[i] = norm(beta.hat - beta, "2")
    err.s[i] = norm(beta.hat[1:s] - beta[1:s], "2")
    err.sc[i] = norm(beta.hat[-(1:s)], "2")
  }
  return (list(err.all = err.all, err.s = err.s, err.sc = err.sc))
}


n = 200
s = 10
p = 2000
sig = 2
beta = c(rep(sig, s), rep(0, p - s))
beta0 = 1

X = matrix(rnorm(n * p), n, p)
err = rnorm(n)
Y = beta0 + X %*% beta + err
alpha = 0.0001

### glmnet
fit.lasso = cv.glmnet(X, Y)
beta.lasso = coef(fit.lasso, s = "lambda.min")[-1]
err.all.lasso = norm(beta.lasso - beta, "2")
err.s.lasso = norm(beta.lasso[1:s] - beta[1:s], "2")
err.sc.lasso = norm(beta.lasso[-(1:s)], "2")

rst = linear.opl(X, Y, alpha, beta, s)
plot(1:5000, rst$err.all, type = "l", xlab = "", ylab = "", lwd = 5)
lines(1:5000, rep(err.all.lasso, 5000), col = "red", lwd = 3)
plot(1:5000, rst$err.s, type = "l", xlab = "", ylab = "", lwd = 5)
lines(1:5000, rep(err.s.lasso, 5000), col = "red", lwd = 3)
plot(1:5000, rst$err.sc, type = "l", xlab = "", ylab = "", lwd = 5)
lines(1:5000, rep(err.sc.lasso, 5000), col = "red", lwd = 3)


fit = linearReg(X, Y, alpha = alpha)
beta.hat = fit$coeff[-1]
beta.hat[1:s]
max(abs(beta.hat[-(1:s)]))
norm(beta.hat - beta, "2")

fit.huber = huberReg(X, Y, alpha = alpha)
beta.hat.huber = fit.huber$coeff[-1]
beta.hat.huber[1:s]
max(abs(beta.hat.huber[-(1:s)]))
norm(beta.hat.huber - beta, "2")



