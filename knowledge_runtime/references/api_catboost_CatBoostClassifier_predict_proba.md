# catboost.CatBoostClassifier.predict_proba
Version: 1.2.8

## catboost.CatBoostClassifier.predict_proba(self, X, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type='CPU')

Predict class probability with X.

Parameters
----------
X : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
        or catboost.FeaturesData
    Data to apply model on.
    If X is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
    as a list of features for a single object.

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
    If X is for a single object
        return one-dimensional numpy.ndarray with probability for every class.
    otherwise
        return two-dimensional numpy.ndarray with shape (number_of_objects x number_of_classes)
        with probability for every class for each object.
