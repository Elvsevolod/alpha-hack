# autogluon.tabular.TabularPredictor

Version: 1.3.1

autogluon.tabular.TabularPredictor(label: 'str', problem_type: 'str' = None, eval_metric: 'str | Scorer' = None, path: 'str' = None, verbosity: 'int' = 2, log_to_file: 'bool' = False, log_file_path: 'str' = 'auto', sample_weight: 'str' = None, weight_evaluation: 'bool' = False, groups: 'str' = None, positive_class: 'int | str | None' = None, **kwargs)

AutoGluon TabularPredictor predicts values in a column of a tabular dataset (classification or regression).

Parameters
----------
label : str
    Name of the column that contains the target variable to predict.
problem_type : str, default = None
    Type of prediction problem, i.e. is this a binary/multiclass classification or regression problem (options: 'binary', 'multiclass', 'regression', 'quantile').
    If `problem_type = None`, the prediction problem type is inferred based on the label-values in provided dataset.
eval_metric : str or Scorer, default = None
    Metric by which predictions will be ultimately evaluated on test data.
    AutoGluon tunes factors such as hyperparameters, early-stopping, ensemble-weights, etc. in order to improve this metric on validation data.

    If `eval_metric = None`, it is automatically chosen based on `problem_type`.
    Defaults to 'accuracy' for binary and multiclass classification, 'root_mean_squared_error' for regression, and 'pinball_loss' for quantile.

    Otherwise, options for classification:
        ['accuracy', 'balanced_accuracy', 'log_loss', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
        'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_macro', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_macro', 'roc_auc_ovr_micro',
        'roc_auc_ovr_weighted', 'average_precision', 'precision', 'precision_macro', 'precision_micro', 'precision_weighted',
        'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'mcc', 'pac_score']
    Options for regression:
        ['root_mean_squared_error', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error', 'mean_absolute_percentage_error', 'r2', 'symmetric_mean_absolute_percentage_error']
    For more information on these options, see `sklearn.metrics`: https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    For metric source code, see `autogluon.core.metrics`.

    You can also pass your own evaluation function here as long as it follows formatting of the functions defined in folder `autogluon.core.metrics`.
    For detailed instructions on creating and using a custom metric, refer to https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-metric.html
path : Union[str, pathlib.Path], default = None
    Path to directory where models and intermediate outputs should be saved.
    If unspecified, a time-stamped folder called "AutogluonModels/ag-[TIMESTAMP]" will be created in the working directory to store all models.
    Note: To call `fit()` twice and save all results of each fit, you must specify different `path` locations or don't specify `path` at all.
    Otherwise files from first `fit()` will be overwritten by second `fit()`.
verbosity : int, default = 2
    Verbosity levels range from 0 to 4 and control how much information is printed.
    Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
    If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
    where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels).
    Verbosity levels:
        0: Only log exceptions
        1: Only log warnings + exceptions
        2: Standard logging
        3: Verbose logging (ex: log validation score every 50 iterations)
        4: Maximally verbose logging (ex: log validation score every iteration)
log_to_file: bool, default = False
    Whether to save the logs into a file for later reference
log_file_path: str, default = "auto"
    File path to save the logs.
    If auto, logs will be saved under `predictor_path/logs/predictor_log.txt`.
    Will be ignored if `log_to_file` is set to False
sample_weight : str, default = None
    If specified, this column-name indicates which column of the data should be treated as sample weights. This column will NOT be considered as a predictive feature.
    Sample weights should be non-negative (and cannot be nan), with larger values indicating which rows are more important than others.
    If you want your usage of sample weights to match results obtained outside of this Predictor, then ensure sample weights for your training (or tuning) data sum to the number of rows in the training (or tuning) data.
    You may also specify two special strings: 'auto_weight' (automatically choose a weighting strategy based on the data) or 'balance_weight' (equally weight classes in classification, no effect in regression). If specifying your own sample_weight column, make sure its name does not match these special strings.
weight_evaluation : bool, default = False
    Only considered when `sample_weight` column is not None. Determines whether sample weights should be taken into account when computing evaluation metrics on validation/test data.
    If True, then weighted metrics will be reported based on the sample weights provided in the specified `sample_weight` (in which case `sample_weight` column must also be present in test data).
    In this case, the 'best' model used by default for prediction will also be decided based on a weighted version of evaluation metric.
    Note: we do not recommend specifying `weight_evaluation` when `sample_weight` is 'auto_weight' or 'balance_weight', instead specify appropriate `eval_metric`.
groups : str, default = None
    [Experimental] If specified, AutoGluon will use the column named the value of groups in `train_data` during `.fit` as the data splitting indices for the purposes of bagging.
    This column will not be used as a feature during model training.
    This parameter is ignored if bagging is not enabled. To instead specify a custom validation set with bagging disabled, specify `tuning_data` in `.fit`.
    The data will be split via `sklearn.model_selection.LeaveOneGroupOut`.
    Use this option to control the exact split indices AutoGluon uses.
    It is not recommended to use this option unless it is required for very specific situations.
    Bugs may arise from edge cases if the provided groups are not valid to properly train models, such as if not all classes are present during training in multiclass classification. It is up to the user to sanitize their groups.

    As an example, if you want your data folds to preserve adjacent rows in the table without shuffling, then for 3 fold bagging with 6 rows of data, the groups column values should be [0, 0, 1, 1, 2, 2].
positive_class : str or int, default = None
    Used to determine the positive class in binary classification.
    This is used for certain metrics such as 'f1' which produce different scores depending on which class is considered the positive class.
    If not set, will be inferred as the second element of the existing unique classes after sorting them.
        If classes are [0, 1], then 1 will be selected as the positive class.
        If classes are ['def', 'abc'], then 'def' will be selected as the positive class.
        If classes are [True, False], then True will be selected as the positive class.
**kwargs :
    learner_type : AbstractLearner, default = DefaultLearner
        A class which inherits from `AbstractLearner`. This dictates the inner logic of predictor.
        If you don't know what this is, keep it as the default.
    learner_kwargs : dict, default = None
        Kwargs to send to the learner. Options include:

        ignored_columns : list, default = None
            Banned subset of column names that predictor may not use as predictive features (e.g. unique identifier to a row or user-ID).
            These columns are ignored during `fit()`.
        label_count_threshold : int, default = 10
            For multi-class classification problems, this is the minimum number of times a label must appear in dataset in order to be considered an output class.
            AutoGluon will ignore any classes whose labels do not appear at least this many times in the dataset (i.e. will never predict them).
        cache_data : bool, default = True
            When enabled, the training and validation data are saved to disk for future reuse.
            Enables advanced functionality in predictor such as `fit_extra()` and feature importance calculation on the original data.
        trainer_type : AbstractTabularTrainer, default = AutoTrainer
            A class inheriting from `AbstractTabularTrainer` that controls training/ensembling of many models.
            If you don't know what this is, keep it as the default.

