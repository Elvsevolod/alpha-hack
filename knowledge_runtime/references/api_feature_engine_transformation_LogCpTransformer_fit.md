# feature_engine.transformation.LogCpTransformer.fit
Version: 1.8.3

## feature_engine.transformation.LogCpTransformer.fit(self, X: pandas.core.frame.DataFrame, y: Optional[pandas.core.series.Series] = None)

Learn the constant C to add to the variable before the logarithm transformation
if C="auto".

Select the numerical variables or check that the variables entered by the user
are numerical. Then check that the selected variables are positive after
addition of C.

Parameters
----------
X: Pandas DataFrame of shape = [n_samples, n_features].
    The training input samples. Can be the entire dataframe, not just the
    variables to transform.

y: pandas Series, default=None
    It is not needed in this transformer. You can pass y or None.
