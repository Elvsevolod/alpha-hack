# sklearn.preprocessing.PowerTransformer.fit_transform
Version: 1.5.2

## sklearn.preprocessing.PowerTransformer.fit_transform(self, X, y=None)

Fit `PowerTransformer` to `X`, then transform `X`.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data used to estimate the optimal transformation parameters
    and to be transformed using a power transformation.

y : Ignored
    Not used, present for API consistency by convention.

Returns
-------
X_new : ndarray of shape (n_samples, n_features)
    Transformed data.
