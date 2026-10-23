# sklearn.linear_model.ElasticNet.fit
Version: 1.5.2

## sklearn.linear_model.ElasticNet.fit(self, X, y, sample_weight=None, check_input=True)

Fit model with coordinate descent.

Parameters
----------
X : {ndarray, sparse matrix, sparse array} of (n_samples, n_features)
    Data.

    Note that large sparse matrices and arrays requiring `int64`
    indices are not accepted.

y : ndarray of shape (n_samples,) or (n_samples, n_targets)
    Target. Will be cast to X's dtype if necessary.

sample_weight : float or array-like of shape (n_samples,), default=None
    Sample weights. Internally, the `sample_weight` vector will be
    rescaled to sum to `n_samples`.

    .. versionadded:: 0.23

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
self : object
    Fitted estimator.

Notes
-----
Coordinate descent is an algorithm that considers each column of
data at a time hence it will automatically convert the X input
as a Fortran-contiguous numpy array if necessary.

To avoid memory re-allocation it is advised to allocate the
initial data in memory directly using that format.
