# statsmodels.regression.linear_model.OLS.fit
Version: 0.14.0

## statsmodels.regression.linear_model.OLS.fit(self, method: "Literal['pinv', 'qr']" = 'pinv', cov_type: "Literal['nonrobust', 'fixed scale', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC', 'hac-panel', 'hac-groupsum', 'cluster']" = 'nonrobust', cov_kwds=None, use_t: 'bool | None' = None, **kwargs)

Full fit of the model.

The results include an estimate of covariance matrix, (whitened)
residuals and an estimate of scale.

Parameters
----------
method : str, optional
    Can be "pinv", "qr".  "pinv" uses the Moore-Penrose pseudoinverse
    to solve the least squares problem. "qr" uses the QR
    factorization.
cov_type : str, optional
    See `regression.linear_model.RegressionResults` for a description
    of the available covariance estimators.
cov_kwds : list or None, optional
    See `linear_model.RegressionResults.get_robustcov_results` for a
    description required keywords for alternative covariance
    estimators.
use_t : bool, optional
    Flag indicating to use the Student's t distribution when computing
    p-values.  Default behavior depends on cov_type. See
    `linear_model.RegressionResults.get_robustcov_results` for
    implementation details.
**kwargs
    Additional keyword arguments that contain information used when
    constructing a model using the formula interface.

Returns
-------
RegressionResults
    The model estimation results.

See Also
--------
RegressionResults
    The results container.
RegressionResults.get_robustcov_results
    A method to change the covariance estimator used when fitting the
    model.

Notes
-----
The fit method uses the pseudoinverse of the design/exogenous variables
to solve the least squares minimization.
