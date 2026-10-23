# xgboost.XGBRegressor.fit
Version: 2.1.4

## xgboost.XGBRegressor.fit(self, X: Any, y: Any, *, sample_weight: Optional[Any] = None, base_margin: Optional[Any] = None, eval_set: Optional[Sequence[Tuple[Any, Any]]] = None, verbose: Union[bool, int, NoneType] = True, xgb_model: Union[xgboost.core.Booster, ForwardRef('XGBModel'), str, NoneType] = None, sample_weight_eval_set: Optional[Sequence[Any]] = None, base_margin_eval_set: Optional[Sequence[Any]] = None, feature_weights: Optional[Any] = None) -> 'XGBModel'

Fit gradient boosting model.

Note that calling ``fit()`` multiple times will cause the model object to be
re-fit from scratch. To resume training from a previous checkpoint, explicitly
pass ``xgb_model`` argument.

Parameters
----------
X :
    Feature matrix. See :ref:`py-data` for a list of supported types.

    When the ``tree_method`` is set to ``hist``, internally, the
    :py:class:`QuantileDMatrix` will be used instead of the :py:class:`DMatrix`
    for conserving memory. However, this has performance implications when the
    device of input data is not matched with algorithm. For instance, if the
    input is a numpy array on CPU but ``cuda`` is used for training, then the
    data is first processed on CPU then transferred to GPU.
y :
    Labels
sample_weight :
    instance weights
base_margin :
    Global bias for each instance. See :doc:`/tutorials/intercept` for details.
eval_set :
    A list of (X, y) tuple pairs to use as validation sets, for which
    metrics will be computed.
    Validation metrics will help us track the performance of the model.

verbose :
    If `verbose` is True and an evaluation set is used, the evaluation metric
    measured on the validation set is printed to stdout at each boosting stage.
    If `verbose` is an integer, the evaluation metric is printed at each
    `verbose` boosting stage. The last boosting stage / the boosting stage found
    by using `early_stopping_rounds` is also printed.
xgb_model :
    file name of stored XGBoost model or 'Booster' instance XGBoost model to be
    loaded before training (allows training continuation).
sample_weight_eval_set :
    A list of the form [L_1, L_2, ..., L_n], where each L_i is an array like
    object storing instance weights for the i-th validation set.
base_margin_eval_set :
    A list of the form [M_1, M_2, ..., M_n], where each M_i is an array like
    object storing base margin for the i-th validation set.
feature_weights :
    Weight for each feature, defines the probability of each feature being
    selected when colsample is being used.  All values must be greater than 0,
    otherwise a `ValueError` is thrown.
