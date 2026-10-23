# sklearn.linear_model.SGDClassifier.predict_proba
Version: 1.5.2

## sklearn.linear_model.SGDClassifier.predict_proba(self, X)

Probability estimates.

This method is only available for log loss and modified Huber loss.

Multiclass probability estimates are derived from binary (one-vs.-rest)
estimates by simple normalization, as recommended by Zadrozny and
Elkan.

Binary probability estimates for loss="modified_huber" are given by
(clip(decision_function(X), -1, 1) + 1) / 2. For other loss functions
it is necessary to perform proper probability calibration by wrapping
the classifier with
:class:`~sklearn.calibration.CalibratedClassifierCV` instead.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Input data for prediction.

Returns
-------
ndarray of shape (n_samples, n_classes)
    Returns the probability of the sample for each class in the model,
    where classes are ordered as they are in `self.classes_`.

References
----------
Zadrozny and Elkan, "Transforming classifier scores into multiclass
probability estimates", SIGKDD'02,
https://dl.acm.org/doi/pdf/10.1145/775047.775151

The justification for the formula in the loss="modified_huber"
case is in the appendix B in:
http://jmlr.csail.mit.edu/papers/volume2/zhang02c/zhang02c.pdf
