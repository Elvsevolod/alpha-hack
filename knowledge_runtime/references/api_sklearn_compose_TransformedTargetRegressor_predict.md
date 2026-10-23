# sklearn.compose.TransformedTargetRegressor.predict
Version: 1.5.2

## sklearn.compose.TransformedTargetRegressor.predict(self, X, **predict_params)

Predict using the base regressor, applying inverse.

The regressor is used to predict and the `inverse_func` or
`inverse_transform` is applied before returning the prediction.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Samples.

**predict_params : dict of str -> object
    Parameters passed to the `predict` method of the underlying
    regressor.

Returns
-------
y_hat : ndarray of shape (n_samples,)
    Predicted values.
