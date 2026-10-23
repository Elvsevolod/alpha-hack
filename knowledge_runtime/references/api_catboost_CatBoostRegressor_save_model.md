# catboost.CatBoostRegressor.save_model
Version: 1.2.8

## catboost.CatBoostRegressor.save_model(self, fname, format='cbm', export_parameters=None, pool=None)

Save the model to a file.

Parameters
----------
fname : string
    Output file name.
format : string
    Possible values:
        * 'cbm' for catboost binary format,
        * 'coreml' to export into Apple CoreML format
        * 'onnx' to export into ONNX-ML format
        * 'pmml' to export into PMML format
        * 'cpp' to export as C++ code
        * 'python' to export as Python code.
export_parameters : dict
    Parameters for CoreML export:
        * prediction_type : string - either 'probability' or 'raw'
        * coreml_description : string
        * coreml_model_version : string
        * coreml_model_author : string
        * coreml_model_license: string
    Parameters for PMML export:
        * pmml_copyright : string
        * pmml_description : string
        * pmml_model_version : string
pool : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series or catboost.FeaturesData
    Training pool.
