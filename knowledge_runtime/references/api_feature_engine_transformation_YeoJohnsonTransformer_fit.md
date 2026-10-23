# feature_engine.transformation.YeoJohnsonTransformer.fit
Version: 1.8.3

## feature_engine.transformation.YeoJohnsonTransformer.fit(self, X: pandas.core.frame.DataFrame, y: Optional[pandas.core.series.Series] = None)

Learn the optimal lambda for the Yeo-Johnson transformation.

Parameters
----------
X: pandas dataframe of shape = [n_samples, n_features]
    The training input samples. Can be the entire dataframe, not just the
    variables to transform.

y: pandas Series, default=None
    It is not needed in this transformer. You can pass y or None.
