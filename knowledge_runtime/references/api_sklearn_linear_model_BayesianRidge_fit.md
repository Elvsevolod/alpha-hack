# sklearn.linear_model.BayesianRidge.fit
Version: 1.5.2

## sklearn.linear_model.BayesianRidge.fit(self, X, y, sample_weight=None)

Fit the model.

Parameters
----------
X : ndarray of shape (n_samples, n_features)
    Training data.
y : ndarray of shape (n_samples,)
    Target values. Will be cast to X's dtype if necessary.

sample_weight : ndarray of shape (n_samples,), default=None
    Individual weights for each sample.

    .. versionadded:: 0.20
       parameter *sample_weight* support to BayesianRidge.

Returns
-------
self : object
    Returns the instance itself.
