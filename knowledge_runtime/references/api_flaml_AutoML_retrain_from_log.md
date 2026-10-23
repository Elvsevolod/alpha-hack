# flaml.AutoML.retrain_from_log
Version: 2.3.6

## flaml.AutoML.retrain_from_log(self, log_file_name, X_train=None, y_train=None, dataframe=None, label=None, time_budget=inf, task: 'str | Task | None' = None, eval_method=None, split_ratio=None, n_splits=None, split_type=None, groups=None, n_jobs=-1, train_best=True, train_full=False, record_id=-1, auto_augment=None, custom_hp=None, skip_transform=None, preserve_checkpoint=True, fit_kwargs_by_estimator=None, **fit_kwargs)

Retrain from log file.

This function is intended to retrain the logged configurations.
NOTE: In some rare case, the last config is early stopped to meet time_budget and it's the best config.
But the logged config's ITER_HP (e.g., n_estimators) is not reduced.

Args:
    log_file_name: A string of the log file name.
    X_train: A numpy array or dataframe of training data in shape n*m.
        For time series forecast tasks, the first column of X_train must be the timestamp column (datetime type). Other columns in the dataframe are assumed to be exogenous variables (categorical or numeric).
    y_train: A numpy array or series of labels in shape n*1.
    dataframe: A dataframe of training data including label column.
        For time series forecast tasks, dataframe must be specified and should
        have at least two columns: timestamp and label, where the first
        column is the timestamp column (datetime type). Other columns
        in the dataframe are assumed to be exogenous variables
        (categorical or numeric).
    label: A str of the label column name, e.g., 'label';
        Note: If X_train and y_train are provided,
        dataframe and label are ignored;
        If not, dataframe and label must be provided.
    time_budget: A float number of the time budget in seconds.
    task: A string of the task type, e.g.,
        'classification', 'regression', 'ts_forecast', 'rank',
        'seq-classification', 'seq-regression', 'summarization',
        or an instance of Task class.
    eval_method: A string of resampling strategy, one of
        ['auto', 'cv', 'holdout'].
    split_ratio: A float of the validation data percentage for holdout.
    n_splits: An integer of the number of folds for cross-validation.
    split_type: str or splitter object, default="auto" | the data split type.
        * A valid splitter object is an instance of a derived class of scikit-learn
        [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold)
        and have ``split`` and ``get_n_splits`` methods with the same signatures.
        Set eval_method to "cv" to use the splitter object.
        * Valid str options depend on different tasks.
        For classification tasks, valid choices are
            ["auto", 'stratified', 'uniform', 'time', 'group']. "auto" -> stratified.
        For regression tasks, valid choices are ["auto", 'uniform', 'time', 'group'].
            "auto" -> uniform.
        For time series forecast tasks, must be "auto" or 'time'.
        For ranking task, must be "auto" or 'group'.
    groups: None or array-like | Group labels (with matching length to
        y_train) or groups counts (with sum equal to length of y_train)
        for training data.
    n_jobs: An integer of the number of threads for training | default=-1.
        Use all available resources when n_jobs == -1.
    train_best: A boolean of whether to train the best config in the
        time budget; if false, train the last config in the budget.
    train_full: A boolean of whether to train on the full data. If true,
        eval_method and sample_size in the log file will be ignored.
    record_id: the ID of the training log record from which the model will
        be retrained. By default `record_id = -1` which means this will be
        ignored. `record_id = 0` corresponds to the first trial, and
        when `record_id >= 0`, `time_budget` will be ignored.
    auto_augment: boolean, default=True | Whether to automatically
        augment rare classes.
    custom_hp: dict, default=None | The custom search space specified by user
        Each key is the estimator name, each value is a dict of the custom search space for that estimator. Notice the
        domain of the custom search space can either be a value or a sample.Domain object.

```python
custom_hp = {
    "transformer_ms": {
        "model_path": {
            "domain": "albert-base-v2",
        },
        "learning_rate": {
            "domain": tune.choice([1e-4, 1e-5]),
        }
    }
}
```
    fit_kwargs_by_estimator: dict, default=None | The user specified keywords arguments, grouped by estimator name.
        e.g.,

```python
fit_kwargs_by_estimator = {
    "transformer": {
        "output_dir": "test/data/output/",
        "fp16": False,
    }
}
```

    **fit_kwargs: Other key word arguments to pass to fit() function of
        the searched learners, such as sample_weight. Below are a few examples of
        estimator-specific parameters:
            period: int | forecast horizon for all time series forecast tasks.
            gpu_per_trial: float, default = 0 | A float of the number of gpus per trial,
                only used by TransformersEstimator, XGBoostSklearnEstimator, and
                TemporalFusionTransformerEstimator.
            group_ids: list of strings of column names identifying a time series, only
                used by TemporalFusionTransformerEstimator, required for
                'ts_forecast_panel' task. `group_ids` is a parameter for TimeSeriesDataSet object
                from PyTorchForecasting.
                For other parameters to describe your dataset, refer to
                [TimeSeriesDataSet PyTorchForecasting](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html).
                To specify your variables, use `static_categoricals`, `static_reals`,
                `time_varying_known_categoricals`, `time_varying_known_reals`,
                `time_varying_unknown_categoricals`, `time_varying_unknown_reals`,
                `variable_groups`. To provide more information on your data, use
                `max_encoder_length`, `min_encoder_length`, `lags`.
            log_dir: str, default = "lightning_logs" | Folder into which to log results
                for tensorboard, only used by TemporalFusionTransformerEstimator.
            max_epochs: int, default = 20 | Maximum number of epochs to run training,
                only used by TemporalFusionTransformerEstimator.
            batch_size: int, default = 64 | Batch size for training model, only
                used by TemporalFusionTransformerEstimator and TCNEstimator.
