# sklearn.linear_model.BayesianRidge.predict
Version: 1.5.2

## sklearn.linear_model.BayesianRidge.predict(self, X, return_std=False)

Predict using the linear model.

In addition to the mean of the predictive distribution, also its
standard deviation can be returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Samples.

return_std : bool, default=False
    Whether to return the standard deviation of posterior prediction.

Returns
-------
y_mean : array-like of shape (n_samples,)
    Mean of predictive distribution of query points.

y_std : array-like of shape (n_samples,)
    Standard deviation of predictive distribution of query points.
