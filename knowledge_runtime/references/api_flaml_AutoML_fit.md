# flaml.AutoML.fit
Version: 2.3.6

## flaml.AutoML.fit(self, X_train=None, y_train=None, dataframe=None, label=None, metric=None, task: 'str | Task | None' = None, n_jobs=None, log_file_name=None, estimator_list=None, time_budget=None, max_iter=None, sample=None, ensemble=None, eval_method=None, log_type=None, model_history=None, split_ratio=None, n_splits=None, log_training_metric=None, mem_thres=None, pred_time_limit=None, train_time_limit=None, X_val=None, y_val=None, sample_weight_val=None, groups_val=None, groups=None, verbose=None, retrain_full=None, split_type=None, learner_selector=None, hpo_method=None, starting_points=None, seed=None, n_concurrent_trials=None, keep_search_state=None, preserve_checkpoint=True, early_stop=None, force_cancel=None, append_log=None, auto_augment=None, min_sample_size=None, use_ray=None, use_spark=None, free_mem_ratio=0, metric_constraints=None, custom_hp=None, time_col=None, cv_score_agg_func=None, skip_transform=None, mlflow_logging=None, fit_kwargs_by_estimator=None, mlflow_exp_name=None, **fit_kwargs)

Find a model for a given task.

Args:
    X_train: A numpy array or a pandas dataframe of training data in
        shape (n, m). For time series forecsat tasks, the first column of X_train
        must be the timestamp column (datetime type). Other columns in
        the dataframe are assumed to be exogenous variables (categorical or numeric).
        When using ray, X_train can be a ray.ObjectRef.
    y_train: A numpy array or a pandas series of labels in shape (n, ).
    dataframe: A dataframe of training data including label column.
        For time series forecast tasks, dataframe must be specified and must have
        at least two columns, timestamp and label, where the first
        column is the timestamp column (datetime type). Other columns in
        the dataframe are assumed to be exogenous variables (categorical or numeric).
        When using ray, dataframe can be a ray.ObjectRef.
    label: A str of the label column name for, e.g., 'label';
        Note: If X_train and y_train are provided,
        dataframe and label are ignored;
        If not, dataframe and label must be provided.
    metric: A string of the metric name or a function,
        e.g., 'accuracy', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_weighted',
        'roc_auc_ovo_weighted', 'roc_auc_ovr_weighted', 'f1', 'micro_f1', 'macro_f1',
        'log_loss', 'mae', 'mse', 'r2', 'mape'. Default is 'auto'.
        If passing a customized metric function, the function needs to
        have the following input arguments:

```python
def custom_metric(
    X_test, y_test, estimator, labels,
    X_train, y_train, weight_test=None, weight_train=None,
    config=None, groups_test=None, groups_train=None,
):
    return metric_to_minimize, metrics_to_log
```
        which returns a float number as the minimization objective,
        and a dictionary as the metrics to log. E.g.,

```python
def custom_metric(
    X_val, y_val, estimator, labels,
    X_train, y_train, weight_val=None, weight_train=None,
    *args,
):
    from sklearn.metrics import log_loss
    import time

    start = time.time()
    y_pred = estimator.predict_proba(X_val)
    pred_time = (time.time() - start) / len(X_val)
    val_loss = log_loss(y_val, y_pred, labels=labels, sample_weight=weight_val)
    y_pred = estimator.predict_proba(X_train)
    train_loss = log_loss(y_train, y_pred, labels=labels, sample_weight=weight_train)
    alpha = 0.5
    return val_loss * (1 + alpha) - alpha * train_loss, {
        "val_loss": val_loss,
        "train_loss": train_loss,
        "pred_time": pred_time,
    }
```
    task: A string of the task type, e.g.,
        'classification', 'regression', 'ts_forecast_regression',
        'ts_forecast_classification', 'rank', 'seq-classification',
        'seq-regression', 'summarization', or an instance of Task class
    n_jobs: An integer of the number of threads for training | default=-1.
        Use all available resources when n_jobs == -1.
    log_file_name: A string of the log file name | default="". To disable logging,
        set it to be an empty string "".
    estimator_list: A list of strings for estimator names, or 'auto'.
        e.g., ```['lgbm', 'xgboost', 'xgb_limitdepth', 'catboost', 'rf', 'extra_tree']```.
    time_budget: A float number of the time budget in seconds.
        Use -1 if no time limit.
    max_iter: An integer of the maximal number of iterations.
        NOTE: when both time_budget and max_iter are unspecified,
        only one model will be trained per estimator.
    sample: A boolean of whether to sample the training data during
        search.
    ensemble: boolean or dict | default=False. Whether to perform
        ensemble after search. Can be a dict with keys 'passthrough'
        and 'final_estimator' to specify the passthrough and
        final_estimator in the stacker. The dict can also contain
        'n_jobs' as the key to specify the number of jobs for the stacker.
    eval_method: A string of resampling strategy, one of
        ['auto', 'cv', 'holdout'].
    split_ratio: A float of the valiation data percentage for holdout.
    n_splits: An integer of the number of folds for cross - validation.
    log_type: A string of the log type, one of
        ['better', 'all'].
        'better' only logs configs with better loss than previos iters
        'all' logs all the tried configs.
    model_history: A boolean of whether to keep the trained best
        model per estimator. Make sure memory is large enough if setting to True.
        Default value is False. If False, best_model_for_estimator would return a
        untrained model for non-best learner.
    log_training_metric: A boolean of whether to log the training
        metric for each model.
    mem_thres: A float of the memory size constraint in bytes.
    pred_time_limit: A float of the prediction latency constraint in seconds.
        It refers to the average prediction time per row in validation data.
    train_time_limit: None or a float of the training time constraint in seconds for each trial.
        Only valid for sequential search.
    X_val: None or a numpy array or a pandas dataframe of validation data.
    y_val: None or a numpy array or a pandas series of validation labels.
    sample_weight_val: None or a numpy array of the sample weight of
        validation data of the same shape as y_val.
    groups_val: None or array-like | group labels (with matching length
        to y_val) or group counts (with sum equal to length of y_val)
        for validation data. Need to be consistent with groups.
    groups: None or array-like | Group labels (with matching length to
        y_train) or groups counts (with sum equal to length of y_train)
        for training data.
    verbose: int, default=3 | Controls the verbosity, higher means more
        messages.
        verbose=0: logger level = CRITICAL
        verbose=1: logger level = ERROR
        verbose=2: logger level = WARNING
        verbose=3: logger level = INFO
        verbose=4: logger level = DEBUG
        verbose>5: logger level = NOTSET
    retrain_full: bool or str, default=True | whether to retrain the
        selected model on the full training data when using holdout.
        True - retrain only after search finishes; False - no retraining;
        'budget' - do best effort to retrain without violating the time
        budget.
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
    hpo_method: str, default="auto" | The hyperparameter
        optimization method. By default, CFO is used for sequential
        search and BlendSearch is used for parallel search.
        No need to set when using flaml's default search space or using
        a simple customized search space. When set to 'bs', BlendSearch
        is used. BlendSearch can be tried when the search space is
        complex, for example, containing multiple disjoint, discontinuous
        subspaces. When set to 'random', random search is used.
    starting_points: A dictionary or a str to specify the starting hyperparameter
        config for the estimators | default="data".
        If str:
            - if "data", use data-dependent defaults;
            - if "data:path" use data-dependent defaults which are stored at path;
            - if "static", use data-independent defaults.
        If dict, keys are the name of the estimators, and values are the starting
        hyperparameter configurations for the corresponding estimators.
        The value can be a single hyperparameter configuration dict or a list
        of hyperparameter configuration dicts.
        In the following code example, we get starting_points from the
        `automl` object and use them in the `new_automl` object.
        e.g.,

```python
from flaml import AutoML
automl = AutoML()
X_train, y_train = load_iris(return_X_y=True)
automl.fit(X_train, y_train)
starting_points = automl.best_config_per_estimator

new_automl = AutoML()
new_automl.fit(X_train, y_train, starting_points=starting_points)
```

    seed: int or None, default=None | The random seed for hpo.
    n_concurrent_trials: [In preview] int, default=1 | The number of
        concurrent trials. When n_concurrent_trials > 1, flaml performes
        [parallel tuning](/docs/Use-Cases/Task-Oriented-AutoML#parallel-tuning)
        and installation of ray or spark is required: `pip install flaml[ray]`
        or `pip install flaml[spark]`. Please check
        [here](https://spark.apache.org/docs/latest/api/python/getting_started/install.html)
        for more details about installing Spark.
    keep_search_state: boolean, default=False | Whether to keep data needed
        for model search after fit(). By default the state is deleted for
        space saving.
    preserve_checkpoint: boolean, default=True | Whether to preserve the saved checkpoint
        on disk when deleting automl. By default the checkpoint is preserved.
    early_stop: boolean, default=False | Whether to stop early if the
        search is considered to converge.
    force_cancel: boolean, default=False | Whether to forcely cancel the PySpark job if overtime.
    mlflow_exp_name: str, default=None | The name of the mlflow experiment. This should be specified if
        enable mlflow autologging on Spark. Otherwise it will log all the results into the experiment of the
        same name as the basename of main entry file.
    append_log: boolean, default=False | Whetehr to directly append the log
        records to the input log file if it exists.
    auto_augment: boolean, default=True | Whether to automatically
        augment rare classes.
    min_sample_size: int, default=MIN_SAMPLE_TRAIN | the minimal sample
        size when sample=True.
    use_ray: boolean or dict.
        If boolean: default=False | Whether to use ray to run the training
        in separate processes. This can be used to prevent OOM for large
        datasets, but will incur more overhead in time.
        If dict: the dict contains the keywords arguments to be passed to
        [ray.tune.run](https://docs.ray.io/en/latest/tune/api_docs/execution.html).
    use_spark: boolean, default=False | Whether to use spark to run the training
        in parallel spark jobs. This can be used to accelerate training on large models
        and large datasets, but will incur more overhead in time and thus slow down
        training in some cases.
    free_mem_ratio: float between 0 and 1, default=0. The free memory ratio to keep during training.
    metric_constraints: list, default=[] | The list of metric constraints.
        Each element in this list is a 3-tuple, which shall be expressed
        in the following format: the first element of the 3-tuple is the name of the
        metric, the second element is the inequality sign chosen from ">=" and "<=",
        and the third element is the constraint value. E.g., `('precision', '>=', 0.9)`.
        Note that all the metric names in metric_constraints need to be reported via
        the metrics_to_log dictionary returned by a customized metric function.
        The customized metric function shall be provided via the `metric` key word argument
        of the fit() function or the automl constructor.
        Find examples in this [test](https://github.com/microsoft/FLAML/tree/main/test/automl/test_constraints.py).
        If `pred_time_limit` is provided as one of keyword arguments to fit() function or
        the automl constructor, flaml will automatically (and under the hood)
        add it as an additional element in the metric_constraints. Essentially 'pred_time_limit'
        specifies a constraint about the prediction latency constraint in seconds.
    custom_hp: dict, default=None | The custom search space specified by user
        Each key is the estimator name, each value is a dict of the custom search space for that estimator. Notice the
        domain of the custom search space can either be a value of a sample.Domain object.



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
    time_col: for a time series task, name of the column containing the timestamps. If not
        provided, defaults to the first column of X_train/X_val

    cv_score_agg_func: customized cross-validation scores aggregate function. Default to average metrics across folds. If specificed, this function needs to
        have the following input arguments:

        * val_loss_folds: list of floats, the loss scores of each fold;
        * log_metrics_folds: list of dicts/floats, the metrics of each fold to log.

        This function should return the final aggregate result of all folds. A float number of the minimization objective, and a dictionary as the metrics to log or None.
            E.g.,

```python
def cv_score_agg_func(val_loss_folds, log_metrics_folds):
    metric_to_minimize = sum(val_loss_folds)/len(val_loss_folds)
    metrics_to_log = None
    for single_fold in log_metrics_folds:
        if metrics_to_log is None:
            metrics_to_log = single_fold
        elif isinstance(metrics_to_log, dict):
            metrics_to_log = {k: metrics_to_log[k] + v for k, v in single_fold.items()}
        else:
            metrics_to_log += single_fold
    if metrics_to_log:
        n = len(val_loss_folds)
        metrics_to_log = (
            {k: v / n for k, v in metrics_to_log.items()}
            if isinstance(metrics_to_log, dict)
            else metrics_to_log / n
        )
    return metric_to_minimize, metrics_to_log
```

    skip_transform: boolean, default=False | Whether to pre-process data prior to modeling.
    mlflow_logging: boolean, default=None | Whether to log the training results to mlflow.
        Default value is None, which means the logging decision is made based on
        AutoML.__init__'s mlflow_logging argument. Not valid if mlflow is not installed.
    fit_kwargs_by_estimator: dict, default=None | The user specified keywords arguments, grouped by estimator name.
        For TransformersEstimator, available fit_kwargs can be found from
        [TrainingArgumentsForAuto](nlp/huggingface/training_args).
        e.g.,

```python
fit_kwargs_by_estimator = {
    "transformer": {
        "output_dir": "test/data/output/",
        "fp16": False,
    },
    "tft": {
        "max_encoder_length": 1,
        "min_encoder_length": 1,
        "static_categoricals": [],
        "static_reals": [],
        "time_varying_known_categoricals": [],
        "time_varying_known_reals": [],
        "time_varying_unknown_categoricals": [],
        "time_varying_unknown_reals": [],
        "variable_groups": {},
        "lags": {},
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
