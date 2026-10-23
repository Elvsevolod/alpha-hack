# sklearn.impute.KNNImputer.transform
Version: 1.5.2

## sklearn.impute.KNNImputer.transform(self, X)

Impute all missing values in X.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The input data to complete.

Returns
-------
X : array-like of shape (n_samples, n_output_features)
    The imputed dataset. `n_output_features` is the number of features
    that is not always missing during `fit`.
