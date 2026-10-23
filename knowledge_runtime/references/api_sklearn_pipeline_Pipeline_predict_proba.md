# sklearn.pipeline.Pipeline.predict_proba
Version: 1.5.2

## sklearn.pipeline.Pipeline.predict_proba(self, X, **params)

Transform the data, and apply `predict_proba` with the final estimator.

Call `transform` of each transformer in the pipeline. The transformed
data are finally passed to the final estimator that calls
`predict_proba` method. Only valid if the final estimator implements
`predict_proba`.

Parameters
----------
X : iterable
    Data to predict on. Must fulfill input requirements of first step
    of the pipeline.

**params : dict of str -> object
    - If `enable_metadata_routing=False` (default):

        Parameters to the `predict_proba` called at the end of all
        transformations in the pipeline.

    - If `enable_metadata_routing=True`:

        Parameters requested and accepted by steps. Each step must have
        requested certain metadata for these parameters to be forwarded to
        them.

    .. versionadded:: 0.20

    .. versionchanged:: 1.4
        Parameters are now passed to the ``transform`` method of the
        intermediate steps as well, if requested, and if
        `enable_metadata_routing=True`.

    See :ref:`Metadata Routing User Guide <metadata_routing>` for more
    details.

Returns
-------
y_proba : ndarray of shape (n_samples, n_classes)
    Result of calling `predict_proba` on the final estimator.
