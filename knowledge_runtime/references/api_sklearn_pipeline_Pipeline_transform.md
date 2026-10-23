# sklearn.pipeline.Pipeline.transform
Version: 1.5.2

## sklearn.pipeline.Pipeline.transform(self, X, **params)

Transform the data, and apply `transform` with the final estimator.

Call `transform` of each transformer in the pipeline. The transformed
data are finally passed to the final estimator that calls
`transform` method. Only valid if the final estimator
implements `transform`.

This also works where final estimator is `None` in which case all prior
transformations are applied.

Parameters
----------
X : iterable
    Data to transform. Must fulfill input requirements of first step
    of the pipeline.

**params : dict of str -> object
    Parameters requested and accepted by steps. Each step must have
    requested certain metadata for these parameters to be forwarded to
    them.

    .. versionadded:: 1.4
        Only available if `enable_metadata_routing=True`. See
        :ref:`Metadata Routing User Guide <metadata_routing>` for more
        details.

Returns
-------
Xt : ndarray of shape (n_samples, n_transformed_features)
    Transformed data.
