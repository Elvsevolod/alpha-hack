# xgboost.XGBClassifier.predict_proba
Version: 2.1.4

## xgboost.XGBClassifier.predict_proba(self, X: Any, validate_features: bool = True, base_margin: Optional[Any] = None, iteration_range: Optional[Tuple[Union[int, numpy.integer], Union[int, numpy.integer]]] = None) -> numpy.ndarray

Predict the probability of each `X` example being of a given class. If the
model is trained with early stopping, then :py:attr:`best_iteration` is used
automatically. The estimator uses `inplace_predict` by default and falls back to
using :py:class:`DMatrix` if devices between the data and the estimator don't
match.

.. note:: This function is only thread safe for `gbtree` and `dart`.

Parameters
----------
X :
    Feature matrix. See :ref:`py-data` for a list of supported types.
validate_features :
    When this is True, validate that the Booster's and data's feature_names are
    identical.  Otherwise, it is assumed that the feature_names are the same.
base_margin :
    Global bias for each instance. See :doc:`/tutorials/intercept` for details.
iteration_range :
    Specifies which layer of trees are used in prediction.  For example, if a
    random forest is trained with 100 rounds.  Specifying `iteration_range=(10,
    20)`, then only the forests built during [10, 20) (half open set) rounds are
    used in this prediction.

Returns
-------
prediction :
    a numpy array of shape array-like of shape (n_samples, n_classes) with the
    probability of each data example being of a given class.
