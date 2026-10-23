# xgboost.XGBRegressor.load_model
Version: 2.1.4

## xgboost.XGBRegressor.load_model(self, fname: Union[str, bytearray, os.PathLike]) -> None

Load the model from a file or a bytearray.

The model is saved in an XGBoost internal format which is universal among the
various XGBoost interfaces. Auxiliary attributes of the Python Booster object
(such as feature_names) are only saved when using JSON or UBJSON (default)
format. See :doc:`Model IO </tutorials/saving_model>` for more info.

.. code-block:: python

  model.load_model("model.json")
  # or
  model.load_model("model.ubj")

Parameters
----------
fname :
    Input file name or memory buffer(see also save_raw)
