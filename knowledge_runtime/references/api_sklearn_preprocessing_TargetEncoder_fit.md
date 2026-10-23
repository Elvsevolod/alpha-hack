# sklearn.preprocessing.TargetEncoder.fit
Version: 1.5.2

## sklearn.preprocessing.TargetEncoder.fit(self, X, y)

Fit the :class:`TargetEncoder` to X and y.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data to determine the categories of each feature.

y : array-like of shape (n_samples,)
    The target data used to encode the categories.

Returns
-------
self : object
    Fitted encoder.
