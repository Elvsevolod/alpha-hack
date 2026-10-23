# sklearn.preprocessing.StandardScaler.fit
Version: 1.5.2

## sklearn.preprocessing.StandardScaler.fit(self, X, y=None, sample_weight=None)

Compute the mean and std to be used for later scaling.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The data used to compute the mean and standard deviation
    used for later scaling along the features axis.

y : None
    Ignored.

sample_weight : array-like of shape (n_samples,), default=None
    Individual weights for each sample.

    .. versionadded:: 0.24
       parameter *sample_weight* support to StandardScaler.

Returns
-------
self : object
    Fitted scaler.
