# feature_engine.timeseries.forecasting.LagFeatures.transform
Version: 1.8.3

## feature_engine.timeseries.forecasting.LagFeatures.transform(self, X: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame

Adds lag features.

Parameters
----------
X: pandas dataframe of shape = [n_samples, n_features]
    The data to transform.

Returns
-------
X_new: Pandas dataframe, shape = [n_samples, n_features + lag_features]
    The dataframe with the original plus the new variables.
