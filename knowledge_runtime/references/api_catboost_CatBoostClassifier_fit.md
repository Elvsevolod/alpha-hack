# catboost.CatBoostClassifier.fit
Version: 1.2.8

## catboost.CatBoostClassifier.fit(self, X, y=None, cat_features=None, text_features=None, embedding_features=None, graph=None, sample_weight=None, baseline=None, use_best_model=None, eval_set=None, verbose=None, logging_level=None, plot=False, plot_file=None, column_description=None, verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None, log_cout=None, log_cerr=None)

Fit the CatBoostClassifier model.

Parameters
----------
X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
    If not catboost.Pool, 2 dimensional Feature matrix or string - file with dataset.

y : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
    Labels of the training data.
    If not None, can be a single- or two- dimensional array with either:
      - numerical values - for binary classification problems
      - class labels (boolean, integer or string)
    Use only if X is not catboost.Pool and does not point to a file.

cat_features : list or numpy.ndarray, optional (default=None)
    If not None, giving the list of Categ columns indices.
    Use only if X is not catboost.Pool.

text_features : list or numpy.ndarray, optional (default=None)
    If not None, giving the list of Text columns indices.
    Use only if X is not catboost.Pool.

embedding_features : list or numpy.ndarray, optional (default=None)
    If not None, giving the list of Embedding columns indices.
    Use only if X is not catboost.Pool.

graph : list or numpy.ndarray or pandas.DataFrame
    The graph edges list description.
    If list or numpy.ndarrays or pandas.DataFrame, giving 2 dimensional.

sample_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
    Instance weights, 1 dimensional array like.

baseline : list or numpy.ndarray, optional (default=None)
    If not None, giving 2 dimensional array like data.
    Use only if X is not catboost.Pool.

use_best_model : bool, optional (default=None)
    Flag to use best model

eval_set : catboost.Pool or list of catboost.Pool or tuple (X, y) or list [(X, y)], optional (default=None)
    Validation dataset or datasets for metrics calculation and possibly early stopping.

metric_period : int
    Frequency of evaluating metrics.

verbose : bool or int
    If verbose is bool, then if set to True, logging_level is set to Verbose,
    if set to False, logging_level is set to Silent.
    If verbose is int, it determines the frequency of writing metrics to output and
    logging_level is set to Verbose.

silent : bool
    If silent is True, logging_level is set to Silent.
    If silent is False, logging_level is set to Verbose.

logging_level : string, optional (default=None)
    Possible values:
        - 'Silent'
        - 'Verbose'
        - 'Info'
        - 'Debug'

plot : bool, optional (default=False)
    If True, draw train and eval error in Jupyter notebook

plot_file : file-like or str, optional (default=None)
    If not None, save train and eval error graphs to file

verbose_eval : bool or int
    Synonym for verbose. Only one of these parameters should be set.

early_stopping_rounds : int
    Activates Iter overfitting detector with od_wait set to early_stopping_rounds.

save_snapshot : bool, [default=None]
    Enable progress snapshotting for restoring progress after crashes or interruptions

snapshot_file : string or pathlib.Path, [default=None]
    Learn progress snapshot file path, if None will use default filename

snapshot_interval: int, [default=600]
    Interval between saving snapshots (seconds)

init_model : CatBoost class or string or pathlib.Path, [default=None]
    Continue training starting from the existing model.
    If this parameter is a string or pathlib.Path, load initial model from the path specified by this string.

callbacks : list, optional (default=None)
    List of callback objects that are applied at end of each iteration.

log_cout: output stream or callback for logging (default=None)
    If None is specified, sys.stdout is used

log_cerr: error stream or callback for logging (default=None)
    If None is specified, sys.stderr is used

Returns
-------
model : CatBoost
