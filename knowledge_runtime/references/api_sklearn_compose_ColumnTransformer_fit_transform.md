# sklearn.compose.ColumnTransformer.fit_transform
Version: 1.5.2

## sklearn.compose.ColumnTransformer.fit_transform(self, X, y=None, **params)

Fit all transformers, transform the data and concatenate results.

Parameters
----------
X : {array-like, dataframe} of shape (n_samples, n_features)
    Input data, of which specified subsets are used to fit the
    transformers.

y : array-like of shape (n_samples,), default=None
    Targets for supervised learning.

**params : dict, default=None
    Parameters to be passed to the underlying transformers' ``fit`` and
    ``transform`` methods.

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
