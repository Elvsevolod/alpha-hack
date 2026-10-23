# imblearn.pipeline.Pipeline.fit
Version: 0.12.4

## imblearn.pipeline.Pipeline.fit(self, X, y=None, **params)

Fit the model.

Fit all the transforms/samplers one after the other and
transform/sample the data, then fit the transformed/sampled
data using the final estimator.

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
        `enable_metadata_routing=True` is set via
        :func:`~sklearn.set_config`.

    See :ref:`Metadata Routing User Guide <metadata_routing>` for more
    details.

Returns
-------
self : Pipeline
    This estimator.
