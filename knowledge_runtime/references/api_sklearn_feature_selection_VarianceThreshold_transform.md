# sklearn.feature_selection.VarianceThreshold.transform
Version: 1.5.2

## sklearn.feature_selection.VarianceThreshold.transform(self, X)

Reduce X to the selected features.

Parameters
----------
X : array of shape [n_samples, n_features]
    The input samples.

Returns
-------
X_r : array of shape [n_samples, n_selected_features]
    The input samples with only the selected features.
