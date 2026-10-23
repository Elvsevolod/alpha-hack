# sklearn.impute.MissingIndicator.transform
Version: 1.5.2

## sklearn.impute.MissingIndicator.transform(self, X)

Generate missing values indicator for `X`.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input data to complete.

Returns
-------
Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)         or (n_samples, n_features_with_missing)
    The missing indicator for input data. The data type of `Xt`
    will be boolean.
