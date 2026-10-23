# sklearn.compose.ColumnTransformer.transform
Version: 1.5.2

## sklearn.compose.ColumnTransformer.transform(self, X, **params)

Transform X separately by each transformer, concatenate results.

Parameters
----------
X : {array-like, dataframe} of shape (n_samples, n_features)
    The data to be transformed by subset.

**params : dict, default=None
    Parameters to be passed to the underlying transformers' ``transform``
    method.

    You can only pass this if metadata routing is enabled, which you
    can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

    .. versionadded:: 1.4

Returns
-------
X_t : {array-like, sparse matrix} of                 shape (n_samples, sum_n_components)
    Horizontally stacked results of transformers. sum_n_components is the
    sum of n_components (output dimension) over transformers. If
    any result is a sparse matrix, everything will be converted to
    sparse matrices.
