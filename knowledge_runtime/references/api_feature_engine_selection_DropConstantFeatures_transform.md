# feature_engine.selection.DropConstantFeatures.transform
Version: 1.8.3

## feature_engine.selection.DropConstantFeatures.transform(self, X: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame

Return dataframe with selected features.

Parameters
----------
X: pandas dataframe of shape = [n_samples, n_features].
    The input dataframe.

Returns
-------
X_new: pandas dataframe of shape = [n_samples, n_selected_features]
    Pandas dataframe with the selected features.
