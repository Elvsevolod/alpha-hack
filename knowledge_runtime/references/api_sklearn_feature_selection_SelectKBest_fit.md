# sklearn.feature_selection.SelectKBest.fit
Version: 1.5.2

## sklearn.feature_selection.SelectKBest.fit(self, X, y=None)

Run score function on (X, y) and get the appropriate features.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The training input samples.

y : array-like of shape (n_samples,) or None
    The target values (class labels in classification, real numbers in
    regression). If the selector is unsupervised then `y` can be set to `None`.

Returns
-------
self : object
    Returns the instance itself.
