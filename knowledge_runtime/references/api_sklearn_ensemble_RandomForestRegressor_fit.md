# sklearn.ensemble.RandomForestRegressor.fit
Version: 1.5.2

## sklearn.ensemble.RandomForestRegressor.fit(self, X, y, sample_weight=None)

Build a forest of trees from the training set (X, y).

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The training input samples. Internally, its dtype will be converted
    to ``dtype=np.float32``. If a sparse matrix is provided, it will be
    converted into a sparse ``csc_matrix``.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The target values (class labels in classification, real numbers in
    regression).

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node. In the case of
    classification, splits are also ignored if they would result in any
    single class carrying a negative weight in either child node.

Returns
-------
self : object
    Fitted estimator.
