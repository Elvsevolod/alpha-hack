# sklearn.preprocessing.RobustScaler.fit
Version: 1.5.2

## sklearn.preprocessing.RobustScaler.fit(self, X, y=None)

Compute the median and quantiles to be used for scaling.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The data used to compute the median and quantiles
    used for later scaling along the features axis.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
self : object
    Fitted scaler.
