# sklearn.calibration.CalibratedClassifierCV.predict
Version: 1.5.2

## sklearn.calibration.CalibratedClassifierCV.predict(self, X)

Predict the target of new samples.

The predicted class is the class that has the highest probability,
and can thus be different from the prediction of the uncalibrated classifier.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The samples, as accepted by `estimator.predict`.

Returns
-------
C : ndarray of shape (n_samples,)
    The predicted class.
