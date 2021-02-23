# include <RcppArmadillo.h>
# include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
arma::mat standardize(arma::mat X, const arma::rowvec& mx, const arma::vec& sx, const int p) {
  for (int i = 0; i < p; i++) {
    X.col(i) = (X.col(i) - mx(i)) / sx(i);
  }
  return X;
}

// [[Rcpp::export]]
Rcpp::List linearReg(const arma::mat& X, arma::vec Y, const double alpha, double eta = 0.01, const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  double n1Eta = eta / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx = arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec g = alpha * arma::ones(p + 1), l = alpha * arma::ones(p + 1);
  arma::vec res = Z * (g % l) - Y;
  int ite = 1;
  while (arma::norm(res, "inf") > tol && ite <= iteMax) {
    g -= n1Eta * l % (Z.t() * res);
    l -= n1Eta * g % (Z.t() * res);
    res = Z * (g % l) - Y;
    ite++;
  }
  arma::vec beta = g % l;
  beta.rows(1, p) /= sx;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("g") = g, Rcpp::Named("l") = l, Rcpp::Named("ite") = ite, 
                            Rcpp::Named("residual") = res);
}

// [[Rcpp::export]]
int sgn(const double x) {
  return (x > 0) - (x < 0);
}

// [[Rcpp::export]]
double mad(const arma::vec& x) {
  return 1.482602 * arma::median(arma::abs(x - arma::median(x)));
}

// [[Rcpp::export]]
arma::vec gradHuber(const arma::mat& Z, const arma::vec& res, const double tau, const int n) {
  arma::vec der(n);
  for (int i = 0; i < n; i++) {
    double cur = res(i);
    der(i) = std::abs(cur) <= tau ? cur : tau * sgn(cur);
  }
  return Z.t() * der;
}

// [[Rcpp::export]]
Rcpp::List huberReg(const arma::mat& X, arma::vec Y, const double alpha, const double constTau = 1.345, double eta = 0.01, 
                    const double tol = 0.0001, const int iteMax = 5000) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  double n1Eta = eta / n;
  arma::rowvec mx = arma::mean(X, 0);
  arma::vec sx = arma::stddev(X, 0, 0).t();
  arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx, p));
  double my = arma::mean(Y);
  Y -= my;
  arma::vec g = alpha * arma::ones(p + 1), l = alpha * arma::ones(p + 1);
  arma::vec res = Z * (g % l) - Y;
  double tau = constTau * mad(res);
  int ite = 1;
  while (arma::norm(res, "inf") > tol && ite <= iteMax) {
    arma:vec grad = gradHuber(res, tau, n);
    g -= n1Eta * l % grad;
    l -= n1Eta * g % grad;
    res = Z * (g % l) - Y;
    tau = constTau * mad(res);
    ite++;
  }
  arma::vec beta = g % l;
  beta.rows(1, p) /= sx;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("g") = g, Rcpp::Named("l") = l, Rcpp::Named("ite") = ite, 
                            Rcpp::Named("residual") = res);
}


