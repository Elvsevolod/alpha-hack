# sklearn.preprocessing.OneHotEncoder.transform
Version: 1.5.2

## sklearn.preprocessing.OneHotEncoder.transform(self, X)

Transform X using one-hot encoding.

If `sparse_output=True` (default), it returns an instance of
:class:`scipy.sparse._csr.csr_matrix` (CSR format).

If there are infrequent categories for a feature, set by specifying
`max_categories` or `min_frequency`, the infrequent categories are
grouped into a single category.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data to encode.

Returns
-------
X_out : {ndarray, sparse matrix} of shape                 (n_samples, n_encoded_features)
    Transformed input. If `sparse_output=True`, a sparse matrix will be
    returned.
