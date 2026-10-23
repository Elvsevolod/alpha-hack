# sklearn.linear_model.Ridge.fit
Version: 1.5.2

## sklearn.linear_model.Ridge.fit(self, X, y, sample_weight=None)

Fit Ridge regression model.

Parameters
----------
X : {ndarray, sparse matrix} of shape (n_samples, n_features)
    Training data.

y : ndarray of shape (n_samples,) or (n_samples, n_targets)
    Target values.

sample_weight : float or ndarray of shape (n_samples,), default=None
    Individual weights for each sample. If given a float, every sample
    will have the same weight.

Returns
-------
self : object
    Fitted estimator.
