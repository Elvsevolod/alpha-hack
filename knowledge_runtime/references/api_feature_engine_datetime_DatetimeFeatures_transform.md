# feature_engine.datetime.DatetimeFeatures.transform
Version: 1.8.3

## feature_engine.datetime.DatetimeFeatures.transform(self, X: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame

Extract the date and time features and add them to the dataframe.

Parameters
----------
X: pandas dataframe of shape = [n_samples, n_features]
    The data to transform.

Returns
-------
X_new: Pandas dataframe, shape = [n_samples, n_features x n_df_features]
    The dataframe with the original variables plus the new variables.
