# sklearn.calibration.CalibratedClassifierCV.fit
Version: 1.5.2

## sklearn.calibration.CalibratedClassifierCV.fit(self, X, y, sample_weight=None, **fit_params)

Fit the calibrated model.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data.

y : array-like of shape (n_samples,)
    Target values.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted.

**fit_params : dict
    Parameters to pass to the `fit` method of the underlying
    classifier.

Returns
-------
self : object
    Returns an instance of self.
