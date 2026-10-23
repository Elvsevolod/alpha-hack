# sklearn.preprocessing.SplineTransformer.transform
Version: 1.5.2

## sklearn.preprocessing.SplineTransformer.transform(self, X)

Transform each feature data to B-splines.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data to transform.

Returns
-------
XBS : {ndarray, sparse matrix} of shape (n_samples, n_features * n_splines)
    The matrix of features, where n_splines is the number of bases
    elements of the B-splines, n_knots + degree - 1.
