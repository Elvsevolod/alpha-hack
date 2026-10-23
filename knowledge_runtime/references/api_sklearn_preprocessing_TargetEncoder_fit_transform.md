# sklearn.preprocessing.TargetEncoder.fit_transform
Version: 1.5.2

## sklearn.preprocessing.TargetEncoder.fit_transform(self, X, y)

Fit :class:`TargetEncoder` and transform X with the target encoding.

.. note::
    `fit(X, y).transform(X)` does not equal `fit_transform(X, y)` because a
    :term:`cross fitting` scheme is used in `fit_transform` for encoding.
    See the :ref:`User Guide <target_encoder>`. for details.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data to determine the categories of each feature.

y : array-like of shape (n_samples,)
    The target data used to encode the categories.

Returns
-------
X_trans : ndarray of shape (n_samples, n_features) or                     (n_samples, (n_features * n_classes))
    Transformed input.
