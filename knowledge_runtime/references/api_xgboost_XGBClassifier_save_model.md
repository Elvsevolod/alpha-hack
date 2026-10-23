# xgboost.XGBClassifier.save_model
Version: 2.1.4

## xgboost.XGBClassifier.save_model(self, fname: Union[str, os.PathLike]) -> None

Save the model to a file.

The model is saved in an XGBoost internal format which is universal among the
various XGBoost interfaces. Auxiliary attributes of the Python Booster object
(such as feature_names) are only saved when using JSON or UBJSON (default)
format. See :doc:`Model IO </tutorials/saving_model>` for more info.

.. code-block:: python

  model.save_model("model.json")
  # or
  model.save_model("model.ubj")

Parameters
----------
fname :
    Output file name
