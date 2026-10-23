# sklearn.feature_selection.VarianceThreshold.fit
Version: 1.5.2

## sklearn.feature_selection.VarianceThreshold.fit(self, X, y=None)

Learn empirical variances from X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Data from which to compute variances, where `n_samples` is
    the number of samples and `n_features` is the number of features.

y : any, default=None
    Ignored. This parameter exists only for compatibility with
    sklearn.pipeline.Pipeline.

Returns
-------
self : object
    Returns the instance itself.
