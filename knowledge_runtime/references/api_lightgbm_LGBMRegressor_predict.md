# lightgbm.LGBMRegressor.predict
Version: 4.6.0

## lightgbm.LGBMRegressor.predict(self, X: Union[lightgbm.compat.dt_DataTable, List[Union[List[float], List[int]]], numpy.ndarray, pandas.core.frame.DataFrame, scipy.sparse._matrix.spmatrix], raw_score: bool = False, start_iteration: int = 0, num_iteration: Optional[int] = None, pred_leaf: bool = False, pred_contrib: bool = False, validate_features: bool = False, **kwargs: Any)

Return the predicted value for each sample.

Parameters
----------
X : numpy array, pandas DataFrame, H2O DataTable's Frame (deprecated), scipy.sparse, list of lists of int or float of shape = [n_samples, n_features]
    Input features matrix.
raw_score : bool, optional (default=False)
    Whether to predict raw scores.
start_iteration : int, optional (default=0)
    Start index of the iteration to predict.
    If <= 0, starts from the first iteration.
num_iteration : int or None, optional (default=None)
    Total number of iterations used in the prediction.
    If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
    otherwise, all iterations from ``start_iteration`` are used (no limits).
    If <= 0, all iterations from ``start_iteration`` are used (no limits).
pred_leaf : bool, optional (default=False)
    Whether to predict leaf index.
pred_contrib : bool, optional (default=False)
    Whether to predict feature contributions.

    .. note::

        If you want to get more explanations for your model's predictions using SHAP values,
        like SHAP interaction values,
        you can install the shap package (https://github.com/slundberg/shap).
        Note that unlike the shap package, with ``pred_contrib`` we return a matrix with an extra
        column, where the last column is the expected value.

validate_features : bool, optional (default=False)
    If True, ensure that the features used to predict match the ones used to train.
    Used only if data is pandas DataFrame.
**kwargs
    Other parameters for the prediction.

Returns
-------
predicted_result : array-like of shape = [n_samples] or shape = [n_samples, n_classes]
    The predicted values.
X_leaves : array-like of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]
    If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
X_SHAP_values : array-like of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects
    If ``pred_contrib=True``, the feature contributions for each sample.
