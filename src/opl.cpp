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
  arma::vec g, l = alpha * arma::ones(p + 1);
  arma::vec res = Z * (g % l) - Y;
  int ite = 1;
  while (arma::norm(res, "inf") > tol && ite <= iteMax) {
    g -= n1Eta * l % (X.t() * res);
    l -= n1Eta * g % (X.t() * res);
    res = Z * (g % l) - Y;
    ite++;
  }
  arma::vec beta = g % l;
  beta.rows(1, p) /= sx;
  beta(0) += my - arma::as_scalar(mx * beta.rows(1, p));
  return Rcpp::List::create(Rcpp::Named("coeff") = beta, Rcpp::Named("g") = g, Rcpp::Named("l") = l, Rcpp::Named("ite") = ite, 
                            Rcpp::Named("residual") = res);
}

