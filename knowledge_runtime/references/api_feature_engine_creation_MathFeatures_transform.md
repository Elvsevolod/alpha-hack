# feature_engine.creation.MathFeatures.transform
Version: 1.8.3

## feature_engine.creation.MathFeatures.transform(self, X: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame

Create and add new variables.

Parameters
----------
X: pandas dataframe of shape = [n_samples, n_features]
    The data to transform.

Returns
-------
X_new: Pandas dataframe, shape = [n_samples, n_features + n_operations]
    The input dataframe plus the new variables.
