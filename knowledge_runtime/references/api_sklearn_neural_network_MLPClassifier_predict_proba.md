# sklearn.neural_network.MLPClassifier.predict_proba
Version: 1.5.2

## sklearn.neural_network.MLPClassifier.predict_proba(self, X)

Probability estimates.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input data.

Returns
-------
y_prob : ndarray of shape (n_samples, n_classes)
    The predicted probability of the sample for each class in the
    model, where classes are ordered as they are in `self.classes_`.
