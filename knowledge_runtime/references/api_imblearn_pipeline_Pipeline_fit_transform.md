# imblearn.pipeline.Pipeline.fit_transform
Version: 0.12.4

## imblearn.pipeline.Pipeline.fit_transform(self, X, y=None, **params)

Fit the model and transform with the final estimator.

Fits all the transformers/samplers one after the other and
transform/sample the data, then uses fit_transform on
transformed data with the final estimator.

Parameters
----------
X : iterable
    Training data. Must fulfill input requirements of first step of the
    pipeline.

y : iterable, default=None
    Training targets. Must fulfill label requirements for all steps of
    the pipeline.

**params : dict of str -> object
    - If `enable_metadata_routing=False` (default):

        Parameters passed to the ``fit`` method of each step, where
        each parameter name is prefixed such that parameter ``p`` for step
        ``s`` has key ``s__p``.

    - If `enable_metadata_routing=True`:

        Parameters requested and accepted by steps. Each step must have
        requested certain metadata for these parameters to be forwarded to
        them.

    .. versionchanged:: 1.4
        Parameters are now passed to the ``transform`` method of the
        intermediate steps as well, if requested, and if
        `enable_metadata_routing=True`.

    See :ref:`Metadata Routing User Guide <metadata_routing>` for more
    details.

Returns
-------
Xt : array-like of shape (n_samples, n_transformed_features)
    Transformed samples.
