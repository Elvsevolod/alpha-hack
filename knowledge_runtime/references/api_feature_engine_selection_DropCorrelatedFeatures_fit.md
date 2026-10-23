# feature_engine.selection.DropCorrelatedFeatures.fit
Version: 1.8.3

## feature_engine.selection.DropCorrelatedFeatures.fit(self, X: pandas.core.frame.DataFrame, y: pandas.core.series.Series = None)

Find the correlated features.

Parameters
----------
X : pandas dataframe of shape = [n_samples, n_features]
    The training dataset.

y : pandas series. Default = None
    y is not needed in this transformer. You can pass y or None.
