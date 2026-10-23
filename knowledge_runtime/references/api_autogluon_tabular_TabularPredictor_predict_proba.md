# autogluon.tabular.TabularPredictor.predict_proba
Version: 1.3.1

## autogluon.tabular.TabularPredictor.predict_proba(self, data: 'pd.DataFrame | str', model: 'str | None' = None, as_pandas: 'bool' = True, as_multiclass: 'bool' = True, transform_features: 'bool' = True) -> 'pd.DataFrame | pd.Series | np.ndarray'

Use trained models to produce predicted class probabilities rather than class-labels (if task is classification).
If `predictor.problem_type` is regression or quantile, this will raise an AssertionError.

Parameters
----------
data : :class:`pd.DataFrame` or str
    The data to make predictions for. Should contain same column names as training dataset and follow same format
    (may contain extra columns that won't be used by Predictor, including the label-column itself).
    If str is passed, `data` will be loaded using the str value as the file path.
model : str (optional)
    The name of the model to get prediction probabilities from. Defaults to None, which uses the highest scoring model on the validation set.
    Valid models are listed in this `predictor` by calling `predictor.model_names()`.
as_pandas : bool, default = True
    Whether to return the output as a pandas object (True) or numpy array (False).
    Pandas object is a DataFrame if this is a multiclass problem or `as_multiclass=True`, otherwise it is a Series.
    If the output is a DataFrame, the column order will be equivalent to `predictor.class_labels`.
as_multiclass : bool, default = True
    Whether to return binary classification probabilities as if they were for multiclass classification.
        Output will contain two columns, and if `as_pandas=True`, the column names will correspond to the binary class labels.
        The columns will be the same order as `predictor.class_labels`.
    If False, output will contain only 1 column for the positive class (get positive_class name via `predictor.positive_class`).
    Only impacts output for binary classification problems.
transform_features : bool, default = True
    If True, preprocesses data before predicting with models.
    If False, skips global feature preprocessing.
        This is useful to save on inference time if you have already called `data = predictor.transform_features(data)`.

Returns
-------
Array of predicted class-probabilities, corresponding to each row in the given data.
May be a :class:`np.ndarray` or :class:`pd.DataFrame` / :class:`pd.Series` depending on `as_pandas` and `as_multiclass` arguments and the type of prediction problem.
For binary classification problems, the output contains for each datapoint the predicted probabilities of the negative and positive classes, unless you specify `as_multiclass=False`.
