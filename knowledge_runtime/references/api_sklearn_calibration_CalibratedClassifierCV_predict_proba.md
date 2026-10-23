# sklearn.calibration.CalibratedClassifierCV.predict_proba
Version: 1.5.2

## sklearn.calibration.CalibratedClassifierCV.predict_proba(self, X)

Calibrated probabilities of classification.

This function returns calibrated probabilities of classification
according to each class on an array of test vectors X.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The samples, as accepted by `estimator.predict_proba`.

Returns
-------
C : ndarray of shape (n_samples, n_classes)
    The predicted probas.
