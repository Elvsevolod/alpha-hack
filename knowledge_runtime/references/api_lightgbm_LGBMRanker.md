# lightgbm.LGBMRanker

Version: 4.6.0

lightgbm.LGBMRanker(*, boosting_type: str = 'gbdt', num_leaves: int = 31, max_depth: int = -1, learning_rate: float = 0.1, n_estimators: int = 100, subsample_for_bin: int = 200000, objective: Union[str, Callable[[Optional[numpy.ndarray], numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]], Callable[[Optional[numpy.ndarray], numpy.ndarray, Optional[numpy.ndarray]], Tuple[numpy.ndarray, numpy.ndarray]], Callable[[Optional[numpy.ndarray], numpy.ndarray, Optional[numpy.ndarray], Optional[numpy.ndarray]], Tuple[numpy.ndarray, numpy.ndarray]], NoneType] = None, class_weight: Union[Dict, str, NoneType] = None, min_split_gain: float = 0.0, min_child_weight: float = 0.001, min_child_samples: int = 20, subsample: float = 1.0, subsample_freq: int = 0, colsample_bytree: float = 1.0, reg_alpha: float = 0.0, reg_lambda: float = 0.0, random_state: Union[int, numpy.random.mtrand.RandomState, numpy.random._generator.Generator, NoneType] = None, n_jobs: Optional[int] = None, importance_type: str = 'split', **kwargs: Any) -> None

LightGBM ranker.

.. warning::

    scikit-learn doesn't support ranking applications yet,
    therefore this class is not really compatible with the sklearn ecosystem.
    Please use this class mainly for training and applying ranking models in common sklearnish way.

Constructor parameters:
Construct a gradient boosting model.

Parameters
----------
boosting_type : str, optional (default='gbdt')
    'gbdt', traditional Gradient Boosting Decision Tree.
    'dart', Dropouts meet Multiple Additive Regression Trees.
    'rf', Random Forest.
num_leaves : int, optional (default=31)
    Maximum tree leaves for base learners.
max_depth : int, optional (default=-1)
    Maximum tree depth for base learners, <=0 means no limit.
    If setting this to a positive value, consider also changing ``num_leaves`` to ``<= 2^max_depth``.
learning_rate : float, optional (default=0.1)
    Boosting learning rate.
    You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
    in training using ``reset_parameter`` callback.
    Note, that this will ignore the ``learning_rate`` argument in training.
n_estimators : int, optional (default=100)
    Number of boosted trees to fit.
subsample_for_bin : int, optional (default=200000)
    Number of samples for constructing bins.
objective : str, callable or None, optional (default=None)
    Specify the learning task and the corresponding learning objective or
    a custom objective function to be used (see note below).
    Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
class_weight : dict, 'balanced' or None, optional (default=None)
    Weights associated with classes in the form ``{class_label: weight}``.
    Use this parameter only for multi-class classification task;
    for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
    Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
    You may want to consider performing probability calibration
    (https://scikit-learn.org/stable/modules/calibration.html) of your model.
    The 'balanced' mode uses the values of y to automatically adjust weights
    inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
    If None, all classes are supposed to have weight one.
    Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
    if ``sample_weight`` is specified.
min_split_gain : float, optional (default=0.)
    Minimum loss reduction required to make a further partition on a leaf node of the tree.
min_child_weight : float, optional (default=1e-3)
    Minimum sum of instance weight (Hessian) needed in a child (leaf).
min_child_samples : int, optional (default=20)
    Minimum number of data needed in a child (leaf).
subsample : float, optional (default=1.)
    Subsample ratio of the training instance.
subsample_freq : int, optional (default=0)
    Frequency of subsample, <=0 means no enable.
colsample_bytree : float, optional (default=1.)
    Subsample ratio of columns when constructing each tree.
reg_alpha : float, optional (default=0.)
    L1 regularization term on weights.
reg_lambda : float, optional (default=0.)
    L2 regularization term on weights.
random_state : int, RandomState object or None, optional (default=None)
    Random number seed.
    If int, this number is used to seed the C++ code.
    If RandomState or Generator object (numpy), a random integer is picked based on its state to seed the C++ code.
    If None, default seeds in C++ code are used.
n_jobs : int or None, optional (default=None)
    Number of parallel threads to use for training (can be changed at prediction time by
    passing it as an extra keyword argument).

    For better performance, it is recommended to set this to the number of physical cores
    in the CPU.

    Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
    scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
    threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
    to using the number of physical cores in the system (its correct detection requires
    either the ``joblib`` or the ``psutil`` util libraries to be installed).

    .. versionchanged:: 4.0.0

importance_type : str, optional (default='split')
    The type of feature importance to be filled into ``feature_importances_``.
    If 'split', result contains numbers of times the feature is used in a model.
    If 'gain', result contains total gains of splits which use the feature.
**kwargs
    Other parameters for the model.
    Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

    .. warning::

        \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

Note
----
A custom objective function can be provided for the ``objective`` parameter.
In this case, it should have the signature
``objective(y_true, y_pred) -> grad, hess``,
``objective(y_true, y_pred, weight) -> grad, hess``
or ``objective(y_true, y_pred, weight, group) -> grad, hess``:

    y_true : numpy 1-D array of shape = [n_samples]
        The target values.
    y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
        The predicted values.
        Predicted values are returned before any transformation,
        e.g. they are raw margin instead of probability of positive class for binary task.
    weight : numpy 1-D array of shape = [n_samples]
        The weight of samples. Weights should be non-negative.
    group : numpy 1-D array
        Group/query data.
        Only used in the learning-to-rank task.
        sum(group) = n_samples.
        For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
        where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
    grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
        The value of the first order derivative (gradient) of the loss
        with respect to the elements of y_pred for each sample point.
    hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
        The value of the second order derivative (Hessian) of the loss
        with respect to the elements of y_pred for each sample point.

For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
and grad and hess should be returned in the same format.

