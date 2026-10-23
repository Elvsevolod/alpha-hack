# lightgbm.LGBMRegressor.fit
Version: 4.6.0

## lightgbm.LGBMRegressor.fit(self, X: Union[lightgbm.compat.dt_DataTable, List[Union[List[float], List[int]]], numpy.ndarray, pandas.core.frame.DataFrame, scipy.sparse._matrix.spmatrix], y: Union[List[float], List[int], numpy.ndarray, pandas.core.series.Series, pandas.core.frame.DataFrame, lightgbm.compat.pa_Array, lightgbm.compat.pa_ChunkedArray], sample_weight: Union[List[float], List[int], numpy.ndarray, pandas.core.series.Series, lightgbm.compat.pa_Array, lightgbm.compat.pa_ChunkedArray, NoneType] = None, init_score: Union[List[float], List[List[float]], numpy.ndarray, pandas.core.series.Series, pandas.core.frame.DataFrame, lightgbm.compat.pa_Table, lightgbm.compat.pa_Array, lightgbm.compat.pa_ChunkedArray, NoneType] = None, eval_set: Optional[List[Tuple[Union[lightgbm.compat.dt_DataTable, List[Union[List[float], List[int]]], numpy.ndarray, pandas.core.frame.DataFrame, scipy.sparse._matrix.spmatrix], Union[List[float], List[int], numpy.ndarray, pandas.core.series.Series, pandas.core.frame.DataFrame, lightgbm.compat.pa_Array, lightgbm.compat.pa_ChunkedArray]]]] = None, eval_names: Optional[List[str]] = None, eval_sample_weight: Optional[List[Union[List[float], List[int], numpy.ndarray, pandas.core.series.Series, lightgbm.compat.pa_Array, lightgbm.compat.pa_ChunkedArray]]] = None, eval_init_score: Optional[List[Union[List[float], List[List[float]], numpy.ndarray, pandas.core.series.Series, pandas.core.frame.DataFrame, lightgbm.compat.pa_Table, lightgbm.compat.pa_Array, lightgbm.compat.pa_ChunkedArray]]] = None, eval_metric: Union[str, Callable[[Optional[numpy.ndarray], numpy.ndarray], Tuple[str, float, bool]], Callable[[Optional[numpy.ndarray], numpy.ndarray], List[Tuple[str, float, bool]]], Callable[[Optional[numpy.ndarray], numpy.ndarray, Optional[numpy.ndarray]], Tuple[str, float, bool]], Callable[[Optional[numpy.ndarray], numpy.ndarray, Optional[numpy.ndarray]], List[Tuple[str, float, bool]]], Callable[[Optional[numpy.ndarray], numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]], Tuple[str, float, bool]], Callable[[Optional[numpy.ndarray], numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]], List[Tuple[str, float, bool]]], List[Union[str, Callable[[Optional[numpy.ndarray], numpy.ndarray], Tuple[str, float, bool]], Callable[[Optional[numpy.ndarray], numpy.ndarray], List[Tuple[str, float, bool]]], Callable[[Optional[numpy.ndarray], numpy.ndarray, Optional[numpy.ndarray]], Tuple[str, float, bool]], Callable[[Optional[numpy.ndarray], numpy.ndarray, Optional[numpy.ndarray]], List[Tuple[str, float, bool]]], Callable[[Optional[numpy.ndarray], numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]], Tuple[str, float, bool]], Callable[[Optional[numpy.ndarray], numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]], List[Tuple[str, float, bool]]]]], NoneType] = None, feature_name: Union[List[str], ForwardRef("Literal['auto']")] = 'auto', categorical_feature: Union[List[str], List[int], ForwardRef("Literal['auto']")] = 'auto', callbacks: Optional[List[Callable]] = None, init_model: Union[str, pathlib.Path, lightgbm.basic.Booster, lightgbm.sklearn.LGBMModel, NoneType] = None) -> 'LGBMRegressor'

Build a gradient boosting model from the training set (X, y).

Parameters
----------
X : numpy array, pandas DataFrame, H2O DataTable's Frame (deprecated), scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]
    Input feature matrix.
y : numpy array, pandas DataFrame, pandas Series, list of int or float of shape = [n_samples]
    The target values (class labels in classification, real numbers in regression).
sample_weight : numpy array, pandas Series, list of int or float of shape = [n_samples] or None, optional (default=None)
    Weights of training data. Weights should be non-negative.
init_score : numpy array, pandas DataFrame, pandas Series, list of int or float of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task) or shape = [n_samples, n_classes] (for multi-class task) or None, optional (default=None)
    Init score of training data.
eval_set : list or None, optional (default=None)
    A list of (X, y) tuple pairs to use as validation sets.
eval_names : list of str, or None, optional (default=None)
    Names of eval_set.
eval_sample_weight : list of array (same types as ``sample_weight`` supports), or None, optional (default=None)
    Weights of eval data. Weights should be non-negative.
eval_init_score : list of array (same types as ``init_score`` supports), or None, optional (default=None)
    Init score of eval data.
eval_metric : str, callable, list or None, optional (default=None)
    If str, it should be a built-in evaluation metric to use.
    If callable, it should be a custom evaluation metric, see note below for more details.
    If list, it can be a list of built-in metrics, a list of custom evaluation metrics, or a mix of both.
    In either case, the ``metric`` from the model parameters will be evaluated and used as well.
    Default: 'l2' for LGBMRegressor, 'logloss' for LGBMClassifier, 'ndcg' for LGBMRanker.
feature_name : list of str, or 'auto', optional (default='auto')
    Feature names.
    If 'auto' and data is pandas DataFrame, data columns names are used.
categorical_feature : list of str or int, or 'auto', optional (default='auto')
    Categorical features.
    If list of int, interpreted as indices.
    If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
    If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
    All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
    Large values could be memory consuming. Consider using consecutive integers starting from zero.
    All negative values in categorical features will be treated as missing values.
    The output cannot be monotonically constrained with respect to a categorical feature.
    Floating point numbers in categorical features will be rounded towards 0.
callbacks : list of callable, or None, optional (default=None)
    List of callback functions that are applied at each iteration.
    See Callbacks in Python API for more information.
init_model : str, pathlib.Path, Booster, LGBMModel or None, optional (default=None)
    Filename of LightGBM model, Booster instance or LGBMModel instance used for continue training.

Returns
-------
self : LGBMRegressor
    Returns self.



Note
----
Custom eval function expects a callable with following signatures:
``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or
``func(y_true, y_pred, weight, group)``
and returns (eval_name, eval_result, is_higher_better) or
list of (eval_name, eval_result, is_higher_better):

    y_true : numpy 1-D array of shape = [n_samples]
        The target values.
    y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
        The predicted values.
        In case of custom ``objective``, predicted values are returned before any transformation,
        e.g. they are raw margin instead of probability of positive class for binary task in this case.
    weight : numpy 1-D array of shape = [n_samples]
        The weight of samples. Weights should be non-negative.
    group : numpy 1-D array
        Group/query data.
        Only used in the learning-to-rank task.
        sum(group) = n_samples.
        For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
        where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
    eval_name : str
        The name of evaluation function (without whitespace).
    eval_result : float
        The eval result.
    is_higher_better : bool
        Is eval result higher better, e.g. AUC is ``is_higher_better``.
