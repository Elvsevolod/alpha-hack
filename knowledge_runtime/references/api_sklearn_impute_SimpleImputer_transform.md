# sklearn.impute.SimpleImputer.transform
Version: 1.5.2

## sklearn.impute.SimpleImputer.transform(self, X)

Impute all missing values in `X`.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    The input data to complete.

Returns
-------
X_imputed : {ndarray, sparse matrix} of shape                 (n_samples, n_features_out)
    `X` with imputed values.
