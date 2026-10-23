# sklearn.compose.TransformedTargetRegressor.fit
Version: 1.5.2

## sklearn.compose.TransformedTargetRegressor.fit(self, X, y, **fit_params)

Fit the model according to the given training data.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vector, where `n_samples` is the number of samples and
    `n_features` is the number of features.

y : array-like of shape (n_samples,)
    Target values.

**fit_params : dict
    Parameters passed to the `fit` method of the underlying
    regressor.

Returns
-------
self : object
    Fitted estimator.
