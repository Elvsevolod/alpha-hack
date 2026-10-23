# xgboost.XGBRegressor

Version: 2.1.4

xgboost.XGBRegressor(*, objective: Union[str, xgboost.sklearn._SklObjWProto, Callable[[Any, Any], Tuple[numpy.ndarray, numpy.ndarray]], NoneType] = 'reg:squarederror', **kwargs: Any) -> None

Implementation of the scikit-learn API for XGBoost regression.
See :doc:`/python/sklearn_estimator` for more information.

Parameters
----------

    n_estimators : typing.Optional[int]
        Number of gradient boosted trees.  Equivalent to number of boosting
        rounds.

    max_depth :  typing.Optional[int]

        Maximum tree depth for base learners.

    max_leaves : typing.Optional[int]

        Maximum number of leaves; 0 indicates no limit.

    max_bin : typing.Optional[int]

        If using histogram-based algorithm, maximum number of bins per feature

    grow_policy : typing.Optional[str]

        Tree growing policy.

        - depthwise: Favors splitting at nodes closest to the node,
        - lossguide: Favors splitting at nodes with highest loss change.

    learning_rate : typing.Optional[float]

        Boosting learning rate (xgb's "eta")

    verbosity : typing.Optional[int]

        The degree of verbosity. Valid values are 0 (silent) - 3 (debug).

    objective : typing.Union[str, xgboost.sklearn._SklObjWProto, typing.Callable[[typing.Any, typing.Any], typing.Tuple[numpy.ndarray, numpy.ndarray]], NoneType]

        Specify the learning task and the corresponding learning objective or a custom
        objective function to be used.

        For custom objective, see :doc:`/tutorials/custom_metric_obj` and
        :ref:`custom-obj-metric` for more information, along with the end note for
        function signatures.

    booster: typing.Optional[str]

        Specify which booster to use: ``gbtree``, ``gblinear`` or ``dart``.

    tree_method : typing.Optional[str]

        Specify which tree method to use.  Default to auto.  If this parameter is set to
        default, XGBoost will choose the most conservative option available.  It's
        recommended to study this option from the parameters document :doc:`tree method
        </treemethod>`

    n_jobs : typing.Optional[int]

        Number of parallel threads used to run xgboost.  When used with other
        Scikit-Learn algorithms like grid search, you may choose which algorithm to
        parallelize and balance the threads.  Creating thread contention will
        significantly slow down both algorithms.

    gamma : typing.Optional[float]

        (min_split_loss) Minimum loss reduction required to make a further partition on
        a leaf node of the tree.

    min_child_weight : typing.Optional[float]

        Minimum sum of instance weight(hessian) needed in a child.

    max_delta_step : typing.Optional[float]

        Maximum delta step we allow each tree's weight estimation to be.

    subsample : typing.Optional[float]

        Subsample ratio of the training instance.

    sampling_method : typing.Optional[str]

        Sampling method. Used only by the GPU version of ``hist`` tree method.

        - ``uniform``: Select random training instances uniformly.
        - ``gradient_based``: Select random training instances with higher probability
            when the gradient and hessian are larger. (cf. CatBoost)

    colsample_bytree : typing.Optional[float]

        Subsample ratio of columns when constructing each tree.

    colsample_bylevel : typing.Optional[float]

        Subsample ratio of columns for each level.

    colsample_bynode : typing.Optional[float]

        Subsample ratio of columns for each split.

    reg_alpha : typing.Optional[float]

        L1 regularization term on weights (xgb's alpha).

    reg_lambda : typing.Optional[float]

        L2 regularization term on weights (xgb's lambda).

    scale_pos_weight : typing.Optional[float]
        Balancing of positive and negative weights.

    base_score : typing.Optional[float]

        The initial prediction score of all instances, global bias.

    random_state : typing.Union[numpy.random.mtrand.RandomState, numpy.random._generator.Generator, int, NoneType]

        Random number seed.

        .. note::

           Using gblinear booster with shotgun updater is nondeterministic as
           it uses Hogwild algorithm.

    missing : float

        Value in the data which needs to be present as a missing value. Default to
        :py:data:`numpy.nan`.

    num_parallel_tree: typing.Optional[int]

        Used for boosting random forest.

    monotone_constraints : typing.Union[typing.Dict[str, int], str, NoneType]

        Constraint of variable monotonicity.  See :doc:`tutorial </tutorials/monotonic>`
        for more information.

    interaction_constraints : typing.Union[str, typing.List[typing.Tuple[str]], NoneType]

        Constraints for interaction representing permitted interactions.  The
        constraints must be specified in the form of a nested list, e.g. ``[[0, 1], [2,
        3, 4]]``, where each inner list is a group of indices of features that are
        allowed to interact with each other.  See :doc:`tutorial
        </tutorials/feature_interaction_constraint>` for more information

    importance_type: typing.Optional[str]

        The feature importance type for the feature_importances\_ property:

        * For tree model, it's either "gain", "weight", "cover", "total_gain" or
          "total_cover".
        * For linear model, only "weight" is defined and it's the normalized
          coefficients without bias.

    device : typing.Optional[str]

        .. versionadded:: 2.0.0

        Device ordinal, available options are `cpu`, `cuda`, and `gpu`.

    validate_parameters : typing.Optional[bool]

        Give warnings for unknown parameter.

    enable_categorical : bool

        See the same parameter of :py:class:`DMatrix` for details.

    feature_types : typing.Optional[typing.Sequence[str]]

        .. versionadded:: 1.7.0

        Used for specifying feature types without constructing a dataframe. See
        :py:class:`DMatrix` for details.

    max_cat_to_onehot : typing.Optional[int]

        .. versionadded:: 1.6.0

        .. note:: This parameter is experimental

        A threshold for deciding whether XGBoost should use one-hot encoding based split
        for categorical data.  When number of categories is lesser than the threshold
        then one-hot encoding is chosen, otherwise the categories will be partitioned
        into children nodes. Also, `enable_categorical` needs to be set to have
        categorical feature support. See :doc:`Categorical Data
        </tutorials/categorical>` and :ref:`cat-param` for details.

    max_cat_threshold : typing.Optional[int]

        .. versionadded:: 1.7.0

        .. note:: This parameter is experimental

        Maximum number of categories considered for each split. Used only by
        partition-based splits for preventing over-fitting. Also, `enable_categorical`
        needs to be set to have categorical feature support. See :doc:`Categorical Data
        </tutorials/categorical>` and :ref:`cat-param` for details.

    multi_strategy : typing.Optional[str]

        .. versionadded:: 2.0.0

        .. note:: This parameter is working-in-progress.

        The strategy used for training multi-target models, including multi-target
        regression and multi-class classification. See :doc:`/tutorials/multioutput` for
        more information.

        - ``one_output_per_tree``: One model for each target.
        - ``multi_output_tree``:  Use multi-target trees.

    eval_metric : typing.Union[str, typing.List[str], typing.Callable, NoneType]

        .. versionadded:: 1.6.0

        Metric used for monitoring the training result and early stopping.  It can be a
        string or list of strings as names of predefined metric in XGBoost (See
        doc/parameter.rst), one of the metrics in :py:mod:`sklearn.metrics`, or any
        other user defined metric that looks like `sklearn.metrics`.

        If custom objective is also provided, then custom metric should implement the
        corresponding reverse link function.

        Unlike the `scoring` parameter commonly used in scikit-learn, when a callable
        object is provided, it's assumed to be a cost function and by default XGBoost
        will minimize the result during early stopping.

        For advanced usage on Early stopping like directly choosing to maximize instead
        of minimize, see :py:obj:`xgboost.callback.EarlyStopping`.

        See :doc:`/tutorials/custom_metric_obj` and :ref:`custom-obj-metric` for more
        information.

        .. code-block:: python

            from sklearn.datasets import load_diabetes
            from sklearn.metrics import mean_absolute_error
            X, y = load_diabetes(return_X_y=True)
            reg = xgb.XGBRegressor(
                tree_method="hist",
                eval_metric=mean_absolute_error,
            )
            reg.fit(X, y, eval_set=[(X, y)])

    early_stopping_rounds : typing.Optional[int]

        .. versionadded:: 1.6.0

        - Activates early stopping. Validation metric needs to improve at least once in
          every **early_stopping_rounds** round(s) to continue training.  Requires at
          least one item in **eval_set** in :py:meth:`fit`.

        - If early stopping occurs, the model will have two additional attributes:
          :py:attr:`best_score` and :py:attr:`best_iteration`. These are used by the
          :py:meth:`predict` and :py:meth:`apply` methods to determine the optimal
          number of trees during inference. If users want to access the full model
          (including trees built after early stopping), they can specify the
          `iteration_range` in these inference methods. In addition, other utilities
          like model plotting can also use the entire model.

        - If you prefer to discard the trees after `best_iteration`, consider using the
          callback function :py:class:`xgboost.callback.EarlyStopping`.

        - If there's more than one item in **eval_set**, the last entry will be used for
          early stopping.  If there's more than one metric in **eval_metric**, the last
          metric will be used for early stopping.

    callbacks : typing.Optional[typing.List[xgboost.callback.TrainingCallback]]

        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using
        :ref:`Callback API <callback_api>`.

        .. note::

           States in callback are not preserved during training, which means callback
           objects can not be reused for multiple training sessions without
           reinitialization or deepcopy.

        .. code-block:: python

            for params in parameters_grid:
                # be sure to (re)initialize the callbacks before each run
                callbacks = [xgb.callback.LearningRateScheduler(custom_rates)]
                reg = xgboost.XGBRegressor(**params, callbacks=callbacks)
                reg.fit(X, y)

    kwargs : typing.Optional[typing.Any]

        Keyword arguments for XGBoost Booster object.  Full documentation of parameters
        can be found :doc:`here </parameter>`.
        Attempting to set a parameter via the constructor args and \*\*kwargs
        dict simultaneously will result in a TypeError.

        .. note:: \*\*kwargs unsupported by scikit-learn

            \*\*kwargs is unsupported by scikit-learn.  We do not guarantee
            that parameters passed via this argument will interact properly
            with scikit-learn.

        .. note::  Custom objective function

            A custom objective function can be provided for the ``objective``
            parameter. In this case, it should have the signature ``objective(y_true,
            y_pred) -> [grad, hess]`` or ``objective(y_true, y_pred, *, sample_weight)
            -> [grad, hess]``:

            y_true: array_like of shape [n_samples]
                The target values
            y_pred: array_like of shape [n_samples]
                The predicted values
            sample_weight :
                Optional sample weights.

            grad: array_like of shape [n_samples]
                The value of the gradient for each sample point.
            hess: array_like of shape [n_samples]
                The value of the second derivative for each sample point

