# sklearn.linear_model.SGDRegressor.fit
Version: 1.5.2

## sklearn.linear_model.SGDRegressor.fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None)

Fit linear model with Stochastic Gradient Descent.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training data.

y : ndarray of shape (n_samples,)
    Target values.

coef_init : ndarray of shape (n_features,), default=None
    The initial coefficients to warm-start the optimization.

intercept_init : ndarray of shape (1,), default=None
    The initial intercept to warm-start the optimization.

sample_weight : array-like, shape (n_samples,), default=None
    Weights applied to individual samples (1. for unweighted).

Returns
-------
self : object
    Fitted `SGDRegressor` estimator.
