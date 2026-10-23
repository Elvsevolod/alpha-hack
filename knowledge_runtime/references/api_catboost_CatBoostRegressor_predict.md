# catboost.CatBoostRegressor.predict
Version: 1.2.8

## catboost.CatBoostRegressor.predict(self, data, prediction_type=None, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type='CPU')

Predict with data.

Parameters
----------
data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
        or catboost.FeaturesData
    Data to apply model on.
    If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
    as a list of features for a single object.

prediction_type : string, optional (default='RawFormulaVal')
    Can be:
    - 'RawFormulaVal' : return raw formula value.
    - 'Exponent' : return Exponent of raw formula value.

ntree_start: int, optional (default=0)
    Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

ntree_end: int, optional (default=0)
    Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
    If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

thread_count : int (default=-1)
    The number of threads to use when applying the model.
    Allows you to optimize the speed of execution. This parameter doesn't affect results.
    If -1, then the number of threads is set to the number of CPU cores.

verbose : bool
    If True, writes the evaluation metric measured set to stderr.

task_type : string, [default=None]
    The evaluator type.
    Possible values:
        - 'CPU'
        - 'GPU' (models with only numerical features are supported for now)

Returns
-------
prediction :
    If data is for a single object, the return value is single float formula return value
    otherwise one-dimensional numpy.ndarray of formula return values for each object.
