# feature_engine.datetime.DatetimeFeatures.fit
Version: 1.8.3

## feature_engine.datetime.DatetimeFeatures.fit(self, X: pandas.core.frame.DataFrame, y: Optional[pandas.core.series.Series] = None)

This transformer does not learn any parameter.

Finds datetime variables or checks that the variables selected by the user
can be converted to datetime.

Parameters
----------
X: pandas dataframe of shape = [n_samples, n_features]
    The training input samples. Can be the entire dataframe, not just the
    variables to transform.

y: pandas Series, default=None
    It is not needed in this transformer. You can pass y or None.
