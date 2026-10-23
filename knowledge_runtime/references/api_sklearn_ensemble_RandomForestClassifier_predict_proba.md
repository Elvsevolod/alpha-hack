# sklearn.ensemble.RandomForestClassifier.predict_proba
Version: 1.5.2

## sklearn.ensemble.RandomForestClassifier.predict_proba(self, X)

Predict class probabilities for X.

The predicted class probabilities of an input sample are computed as
the mean predicted class probabilities of the trees in the forest.
The class probability of a single tree is the fraction of samples of
the same class in a leaf.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, its dtype will be converted to
    ``dtype=np.float32``. If a sparse matrix is provided, it will be
    converted into a sparse ``csr_matrix``.

Returns
-------
p : ndarray of shape (n_samples, n_classes), or a list of such arrays
    The class probabilities of the input samples. The order of the
    classes corresponds to that in the attribute :term:`classes_`.
