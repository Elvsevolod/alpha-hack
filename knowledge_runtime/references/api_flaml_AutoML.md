# flaml.AutoML

Version: 2.3.6

flaml.AutoML(**settings)

The AutoML class.
Example:

```python
automl = AutoML()
automl_settings = {
    "time_budget": 60,
    "metric": 'accuracy',
    "task": 'classification',
    "log_file_name": 'mylog.log',
}
automl.fit(X_train = X_train, y_train = y_train, **automl_settings)
```

Constructor parameters:
Constructor.

Many settings in fit() can be passed to the constructor too.
If an argument in fit() is provided, it will override the setting passed to the constructor.
If an argument in fit() is not provided but provided in the constructor, the value passed to the constructor will be used.

Args:
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
        'classification', 'regression', 'ts_forecast', 'rank',
        'seq-classification', 'seq-regression', 'summarization',
        or an instance of the Task class.
    n_jobs: An integer of the number of threads for training | default=-1.
        Use all available resources when n_jobs == -1.
    log_file_name: A string of the log file name | default="". To disable logging,
        set it to be an empty string "".
    estimator_list: A list of strings for estimator names, or 'auto'.
        e.g., ```['lgbm', 'xgboost', 'xgb_limitdepth', 'catboost', 'rf', 'extra_tree']```.
    time_budget: A float number of the time budget in seconds.
        Use -1 if no time limit.
    max_iter: An integer of the maximal number of iterations.
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
    model_history: A boolean of whether to keep the best
        model per estimator. Make sure memory is large enough if setting to True. Default False.
    log_training_metric: A boolean of whether to log the training
        metric for each model.
    mem_thres: A float of the memory size constraint in bytes.
    pred_time_limit: A float of the prediction latency constraint in seconds.
        It refers to the average prediction time per row in validation data.
    train_time_limit: None or a float of the training time constraint in seconds for each trial.
        Only valid for sequential search.
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
        config for the estimators | default="static".
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
    force_cancel: boolean, default=False | Whether to forcely cancel Spark jobs if the
        search time exceeded the time budget.
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
        training in some cases. GPU training is not supported yet when use_spark is True.
        For Spark clusters, by default, we will launch one trial per executor. However,
        sometimes we want to launch more trials than the number of executors (e.g., local mode).
        In this case, we can set the environment variable `FLAML_MAX_CONCURRENT` to override
        the detected `num_executors`. The final number of concurrent trials will be the minimum
        of `n_concurrent_trials` and `num_executors`.
    free_mem_ratio: float between 0 and 1, default=0. The free memory ratio to keep during training.
    metric_constraints: list, default=[] | The list of metric constraints.
        Each element in this list is a 3-tuple, which shall be expressed
        in the following format: the first element of the 3-tuple is the name of the
        metric, the second element is the inequality sign chosen from ">=" and "<=",
        and the third element is the constraint value. E.g., `('val_loss', '<=', 0.1)`.
        Note that all the metric names in metric_constraints need to be reported via
        the metrics_to_log dictionary returned by a customized metric function.
        The customized metric function shall be provided via the `metric` key word
        argument of the fit() function or the automl constructor.
        Find an example in the 4th constraint type in this [doc](/docs/Use-Cases/Task-Oriented-AutoML#constraint).
        If `pred_time_limit` is provided as one of keyword arguments to fit() function or
        the automl constructor, flaml will automatically (and under the hood)
        add it as an additional element in the metric_constraints. Essentially 'pred_time_limit'
        specifies a constraint about the prediction latency constraint in seconds.
    custom_hp: dict, default=None | The custom search space specified by user.
        It is a nested dict with keys being the estimator names, and values being dicts
        per estimator search space. In the per estimator search space dict,
        the keys are the hyperparameter names, and values are dicts of info ("domain",
        "init_value", and "low_cost_init_value") about the search space associated with
        the hyperparameter (i.e., per hyperparameter search space dict). When custom_hp
        is provided, the built-in search space which is also a nested dict of per estimator
        search space dict, will be updated with custom_hp. Note that during this nested dict update,
        the per hyperparameter search space dicts will be replaced (instead of updated) by the ones
        provided in custom_hp. Note that the value for "domain" can either be a constant
        or a sample.Domain object.
        e.g.,

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
    skip_transform: boolean, default=False | Whether to pre-process data prior to modeling.
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
    mlflow_logging: boolean, default=True | Whether to log the training results to mlflow. Not valid if mlflow is not installed.

