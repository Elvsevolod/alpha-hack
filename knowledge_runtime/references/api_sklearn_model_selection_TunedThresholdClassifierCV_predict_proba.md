# sklearn.model_selection.TunedThresholdClassifierCV.predict_proba
Version: 1.5.2

## sklearn.model_selection.TunedThresholdClassifierCV.predict_proba(self, X)

Predict class probabilities for `X` using the fitted estimator.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where `n_samples` is the number of samples and
    `n_features` is the number of features.

Returns
-------
probabilities : ndarray of shape (n_samples, n_classes)
    The class probabilities of the input samples.
