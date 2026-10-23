# sklearn.impute.MissingIndicator.fit_transform
Version: 1.5.2

## sklearn.impute.MissingIndicator.fit_transform(self, X, y=None)

Generate missing values indicator for `X`.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input data to complete.

y : Ignored
    Not used, present for API consistency by convention.

Returns
-------
Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)         or (n_samples, n_features_with_missing)
    The missing indicator for input data. The data type of `Xt`
    will be boolean.
