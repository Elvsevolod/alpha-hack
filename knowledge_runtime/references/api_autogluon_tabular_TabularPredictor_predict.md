# autogluon.tabular.TabularPredictor.predict
Version: 1.3.1

## autogluon.tabular.TabularPredictor.predict(self, data: 'pd.DataFrame | str', model: 'str | None' = None, as_pandas: 'bool' = True, transform_features: 'bool' = True, *, decision_threshold: 'float | None' = None) -> 'pd.Series | np.ndarray'

Use trained models to produce predictions of `label` column values for new data.

Parameters
----------
data : :class:`pd.DataFrame` or str
    The data to make predictions for. Should contain same column names as training data and follow same format
    (may contain extra columns that won't be used by Predictor, including the label-column itself).
    If str is passed, `data` will be loaded using the str value as the file path.
model : str (optional)
    The name of the model to get predictions from. Defaults to None, which uses the highest scoring model on the validation set.
    Valid models are listed in this `predictor` by calling `predictor.model_names()`
as_pandas : bool, default = True
    Whether to return the output as a :class:`pd.Series` (True) or :class:`np.ndarray` (False).
transform_features : bool, default = True
    If True, preprocesses data before predicting with models.
    If False, skips global feature preprocessing.
        This is useful to save on inference time if you have already called `data = predictor.transform_features(data)`.
decision_threshold : float, default = None
    The decision threshold used to convert prediction probabilities to predictions.
    Only relevant for binary classification, otherwise ignored.
    If None, defaults to `predictor.decision_threshold`.
    Valid values are in the range [0.0, 1.0]
    You can obtain an optimized `decision_threshold` by first calling `predictor.calibrate_decision_threshold()`.
    Useful to set for metrics such as `balanced_accuracy` and `f1` as `0.5` is often not an optimal threshold.
    Predictions are calculated via the following logic on the positive class: `1 if pred > decision_threshold else 0`

Returns
-------
Array of predictions, one corresponding to each row in given dataset. Either :class:`np.ndarray` or :class:`pd.Series` depending on `as_pandas` argument.
