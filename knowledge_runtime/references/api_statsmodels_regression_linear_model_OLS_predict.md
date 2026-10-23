# statsmodels.regression.linear_model.OLS.predict
Version: 0.14.0

## statsmodels.regression.linear_model.OLS.predict(self, params, exog=None)

Return linear predicted values from a design matrix.

Parameters
----------
params : array_like
    Parameters of a linear model.
exog : array_like, optional
    Design / exogenous data. Model exog is used if None.

Returns
-------
array_like
    An array of fitted values.

Notes
-----
If the model has not yet been fit, params is not optional.
