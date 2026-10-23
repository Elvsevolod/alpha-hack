# sklearn.preprocessing.PowerTransformer.fit
Version: 1.5.2

## sklearn.preprocessing.PowerTransformer.fit(self, X, y=None)

Estimate the optimal parameter lambda for each feature.

The optimal lambda parameter for minimizing skewness is estimated on
each feature independently using maximum likelihood.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data used to estimate the optimal transformation parameters.

y : None
    Ignored.

Returns
-------
self : object
    Fitted transformer.
