# sklearn.impute.IterativeImputer.fit_transform
Version: 1.5.2

## sklearn.impute.IterativeImputer.fit_transform(self, X, y=None, **params)

Fit the imputer on `X` and return the transformed `X`.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Input data, where `n_samples` is the number of samples and
    `n_features` is the number of features.

y : Ignored
    Not used, present for API consistency by convention.

**params : dict
    Parameters routed to the `fit` method of the sub-estimator via the
    metadata routing API.

    .. versionadded:: 1.5
      Only available if
      `sklearn.set_config(enable_metadata_routing=True)` is set. See
      :ref:`Metadata Routing User Guide <metadata_routing>` for more
      details.

Returns
-------
Xt : array-like, shape (n_samples, n_features)
    The imputed input data.
