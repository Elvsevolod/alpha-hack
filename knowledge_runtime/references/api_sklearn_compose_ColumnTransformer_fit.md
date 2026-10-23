# sklearn.compose.ColumnTransformer.fit
Version: 1.5.2

## sklearn.compose.ColumnTransformer.fit(self, X, y=None, **params)

Fit all transformers using X.

Parameters
----------
X : {array-like, dataframe} of shape (n_samples, n_features)
    Input data, of which specified subsets are used to fit the
    transformers.

y : array-like of shape (n_samples,...), default=None
    Targets for supervised learning.

**params : dict, default=None
    Parameters to be passed to the underlying transformers' ``fit`` and
    ``transform`` methods.

    You can only pass this if metadata routing is enabled, which you
    can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

    .. versionadded:: 1.4

Returns
-------
self : ColumnTransformer
    This estimator.
