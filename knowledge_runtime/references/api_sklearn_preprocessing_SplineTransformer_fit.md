# sklearn.preprocessing.SplineTransformer.fit
Version: 1.5.2

## sklearn.preprocessing.SplineTransformer.fit(self, X, y=None, sample_weight=None)

Compute knot positions of splines.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data.

y : None
    Ignored.

sample_weight : array-like of shape (n_samples,), default = None
    Individual weights for each sample. Used to calculate quantiles if
    `knots="quantile"`. For `knots="uniform"`, zero weighted
    observations are ignored for finding the min and max of `X`.

Returns
-------
self : object
    Fitted transformer.
