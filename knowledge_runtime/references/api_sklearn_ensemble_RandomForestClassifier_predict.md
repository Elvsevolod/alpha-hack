# sklearn.ensemble.RandomForestClassifier.predict
Version: 1.5.2

## sklearn.ensemble.RandomForestClassifier.predict(self, X)

Predict class for X.

The predicted class of an input sample is a vote by the trees in
the forest, weighted by their probability estimates. That is,
the predicted class is the one with highest mean probability
estimate across the trees.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, its dtype will be converted to
    ``dtype=np.float32``. If a sparse matrix is provided, it will be
    converted into a sparse ``csr_matrix``.

Returns
-------
y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
    The predicted classes.
