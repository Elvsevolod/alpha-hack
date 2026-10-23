# sklearn.ensemble.ExtraTreesRegressor.predict
Version: 1.5.2

## sklearn.ensemble.ExtraTreesRegressor.predict(self, X)

Predict regression target for X.

The predicted regression target of an input sample is computed as the
mean predicted regression targets of the trees in the forest.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, its dtype will be converted to
    ``dtype=np.float32``. If a sparse matrix is provided, it will be
    converted into a sparse ``csr_matrix``.

Returns
-------
y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
    The predicted values.
