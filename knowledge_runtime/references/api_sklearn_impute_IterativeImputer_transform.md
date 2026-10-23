# sklearn.impute.IterativeImputer.transform
Version: 1.5.2

## sklearn.impute.IterativeImputer.transform(self, X)

Impute all missing values in `X`.

Note that this is stochastic, and that if `random_state` is not fixed,
repeated calls, or permuted input, results will differ.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The input data to complete.

Returns
-------
Xt : array-like, shape (n_samples, n_features)
     The imputed input data.
