# feature_engine.transformation.LogCpTransformer.transform
Version: 1.8.3

## feature_engine.transformation.LogCpTransformer.transform(self, X: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame

Transform the variables with the logarithm of x plus a constant C.

Parameters
----------
X: Pandas DataFrame of shape = [n_samples, n_features]
    The data to be transformed.

Returns
-------
X_new: pandas dataframe
    The dataframe with the transformed variables.
