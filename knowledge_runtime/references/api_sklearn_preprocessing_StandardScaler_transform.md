# sklearn.preprocessing.StandardScaler.transform
Version: 1.5.2

## sklearn.preprocessing.StandardScaler.transform(self, X, copy=None)

Perform standardization by centering and scaling.

Parameters
----------
X : {array-like, sparse matrix of shape (n_samples, n_features)
    The data used to scale along the features axis.
copy : bool, default=None
    Copy the input X or not.

Returns
-------
X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
    Transformed array.
