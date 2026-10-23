# sklearn.impute.IterativeImputer.fit
Version: 1.5.2

## sklearn.impute.IterativeImputer.fit(self, X, y=None, **fit_params)

Fit the imputer on `X` and return self.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Input data, where `n_samples` is the number of samples and
    `n_features` is the number of features.

y : Ignored
    Not used, present for API consistency by convention.

**fit_params : dict
    Parameters routed to the `fit` method of the sub-estimator via the
    metadata routing API.

    .. versionadded:: 1.5
      Only available if
      `sklearn.set_config(enable_metadata_routing=True)` is set. See
      :ref:`Metadata Routing User Guide <metadata_routing>` for more
      details.

Returns
-------
self : object
    Fitted estimator.
