# fit
Source: https://auto.gluon.ai/1.3.1/api/autogluon.tabular.TabularPredictor.fit.html



    

    

    

        TabularPredictor.fit - AutoGluon 1.3.1 documentation

    

    

    

  

    

  
    Contents

    

    Menu

      

  
  
    Expand

    

    Light mode

      

      

      

  
  
    Dark mode

      

  
    Auto light/dark, in light mode

      

      

      

    

    Auto light/dark, in dark mode

      

      

      

    

    

      

  

      

      

  

  
Hide navigation sidebar

  
Hide table of contents sidebar

Skip to content

  

      

  

    

  

  

  
Get Started

Install

Tabular Quick Start

Multimodal Quick Start

Time Series Quick Start

Tutorials

Tabular
Toggle navigation of Tabular

Essentials

In Depth

How It Works

Feature Engineering

Tabular + Text + Images

Advanced
Toggle navigation of Advanced

Multilabel

Kaggle

GPU

Custom Metrics

Custom Models

Custom Models Advanced

Deployment

Multimodal
Toggle navigation of Multimodal

Multimodal Prediction
Toggle navigation of Multimodal Prediction

AutoMM for Image + Text + Tabular - Quick Start

AutoMM for Entity Extraction with Text and Image - Quick Start

AutoMM for Text + Tabular - Quick Start

Object Detection
Toggle navigation of Object Detection

Object Detection Quick Start
Toggle navigation of Object Detection Quick Start

AutoMM Detection - Quick Start on a Tiny COCO Format Dataset

Object Detection Advanced
Toggle navigation of Object Detection Advanced

AutoMM Detection - Finetune on COCO Format Dataset with Customized Settings

Object Detection Data Preparation
Toggle navigation of Object Detection Data Preparation

Convert Data to COCO Format

AutoMM Detection - Prepare Pothole Dataset

AutoMM Detection - Prepare Watercolor Dataset

AutoMM Detection - Prepare COCO2017 Dataset

AutoMM Detection - Prepare Pascal VOC Dataset

AutoMM Detection - Convert VOC Format Dataset to COCO Format

Image Prediction
Toggle navigation of Image Prediction

AutoMM for Image Classification - Quick Start

Zero-Shot Image Classification with CLIP

Image Segmentation
Toggle navigation of Image Segmentation

AutoMM for Semantic Segmentation - Quick Start

Text Prediction
Toggle navigation of Text Prediction

AutoMM for Text - Quick Start

AutoMM for Named Entity Recognition - Quick Start

AutoMM for Named Entity Recognition in Chinese - Quick Start

AutoMM for Text - Multilingual Problems

Document Prediction
Toggle navigation of Document Prediction

AutoMM for Scanned Document Classification

Classifying PDF Documents with AutoMM

Semantic Matching
Toggle navigation of Semantic Matching

Image-to-Image Semantic Matching with AutoMM

Image-Text Semantic Matching with AutoMM

Text-to-Text Semantic Matching with AutoMM

Text Semantic Search with AutoMM

Image-Text Semantic Matching with AutoMM - Zero-Shot

Advanced Topics
Toggle navigation of Advanced Topics

AutoMM Problem Types And Metrics

Hyperparameter Optimization in AutoMM

Continuous Training with AutoMM

Customize AutoMM

Knowledge Distillation in AutoMM

Single GPU Billion-scale Model Training via Parameter-Efficient Finetuning

Few Shot Learning with AutoMM

Handling Class Imbalance with AutoMM - Focal Loss

AutoMM Presets

Faster Prediction with TensorRT

Time Series
Toggle navigation of Time Series

Quick Start

In Depth

Forecasting with Chronos

Model Zoo

Metrics

Advanced
Toggle navigation of Advanced

Custom Models

Cloud Training and Deployment
Toggle navigation of Cloud Training and Deployment

AutoGluon Cloud

AutoGluon Tabular on SageMaker AutoPilot

Deploy AutoGluon Models on Serverless Templates

Cloud Training and Deployment with Amazon SageMaker

Resources

Cheat Sheets

Versions

What's New
Toggle navigation of What's New

Version 1.3.1

Version 1.3.0

Version 1.2.0

Version 1.1.1

Version 1.1.0

Version 1.0.0

Version 0.8.3

Version 0.8.2

Version 0.8.1

Version 0.8.0

Version 0.7.0

Version 0.6.2

Version 0.6.1

Version 0.6.0

Version 0.5.2

Version 0.5.1

Version 0.4.3

Version 0.4.2

Version 0.4.1

Version 0.4.0

GitHub

Tabular FAQ

Multimodal FAQ

Time Series FAQ

API

TabularPredictor
Toggle navigation of TabularPredictor

TabularPredictor.calibrate_decision_threshold

TabularPredictor.clone

TabularPredictor.clone_for_deployment

TabularPredictor.compile

TabularPredictor.delete_models

TabularPredictor.disk_usage

TabularPredictor.disk_usage_per_file

TabularPredictor.distill

TabularPredictor.evaluate

TabularPredictor.evaluate_predictions

TabularPredictor.feature_importance

TabularPredictor.features

TabularPredictor.fit

TabularPredictor.fit_extra

TabularPredictor.fit_pseudolabel

TabularPredictor.fit_summary

TabularPredictor.fit_weighted_ensemble

TabularPredictor.info

TabularPredictor.leaderboard

TabularPredictor.learning_curves

TabularPredictor.load

TabularPredictor.load_data_internal

TabularPredictor.load_log

TabularPredictor.model_failures

TabularPredictor.model_hyperparameters

TabularPredictor.model_info

TabularPredictor.model_names

TabularPredictor.model_refit_map

TabularPredictor.persist

TabularPredictor.plot_ensemble_model

TabularPredictor.predict

TabularPredictor.predict_from_proba

TabularPredictor.predict_multi

TabularPredictor.predict_oof

TabularPredictor.predict_proba

TabularPredictor.predict_proba_multi

TabularPredictor.predict_proba_oof

TabularPredictor.refit_full

TabularPredictor.save

TabularPredictor.save_space

TabularPredictor.set_decision_threshold

TabularPredictor.set_model_best

TabularPredictor.simulation_artifact

TabularPredictor.transform_features

TabularPredictor.transform_labels

TabularPredictor.unpersist

TabularDataset

Tabular Models

MultiModalPredictor
Toggle navigation of MultiModalPredictor

MultiModalPredictor.dump_model

MultiModalPredictor.evaluate

MultiModalPredictor.export_onnx

MultiModalPredictor.extract_embedding

MultiModalPredictor.fit

MultiModalPredictor.fit_summary

MultiModalPredictor.get_num_gpus

MultiModalPredictor.list_supported_models

MultiModalPredictor.load

MultiModalPredictor.optimize_for_inference

MultiModalPredictor.predict

MultiModalPredictor.predict_proba

MultiModalPredictor.save

MultiModalPredictor.set_num_gpus

MultiModalPredictor.set_verbosity

TimeSeriesPredictor
Toggle navigation of TimeSeriesPredictor

TimeSeriesPredictor.evaluate

TimeSeriesPredictor.feature_importance

TimeSeriesPredictor.fit

TimeSeriesPredictor.fit_summary

TimeSeriesPredictor.info

TimeSeriesPredictor.leaderboard

TimeSeriesPredictor.load

TimeSeriesPredictor.make_future_data_frame

TimeSeriesPredictor.model_names

TimeSeriesPredictor.persist

TimeSeriesPredictor.plot

TimeSeriesPredictor.predict

TimeSeriesPredictor.refit_full

TimeSeriesPredictor.save

TimeSeriesPredictor.unpersist

TimeSeriesDataFrame
Toggle navigation of TimeSeriesDataFrame

TimeSeriesDataFrame.assign

TimeSeriesDataFrame.convert_frequency

TimeSeriesDataFrame.copy

TimeSeriesDataFrame.dropna

TimeSeriesDataFrame.fill_missing_values

TimeSeriesDataFrame.from_data_frame

TimeSeriesDataFrame.from_iterable_dataset

TimeSeriesDataFrame.from_path

TimeSeriesDataFrame.from_pickle

TimeSeriesDataFrame.get_model_inputs_for_scoring

TimeSeriesDataFrame.infer_frequency

TimeSeriesDataFrame.num_timesteps_per_item

TimeSeriesDataFrame.slice_by_time

TimeSeriesDataFrame.slice_by_timestep

TimeSeriesDataFrame.sort_index

TimeSeriesDataFrame.split_by_time

TimeSeriesDataFrame.to_data_frame

TimeSeriesDataFrame.train_test_split

Feature Generators

FeatureMetadata
Toggle navigation of FeatureMetadata

FeatureMetadata.add_special_types

FeatureMetadata.from_df

FeatureMetadata.get_feature_type_raw

FeatureMetadata.get_feature_types_special

FeatureMetadata.get_features

FeatureMetadata.get_type_group_map_raw

FeatureMetadata.get_type_group_map_special_from_type_map_special

FeatureMetadata.get_type_map_special

FeatureMetadata.join_metadata

FeatureMetadata.join_metadatas

FeatureMetadata.keep_features

FeatureMetadata.print_feature_metadata_full

FeatureMetadata.remove_features

FeatureMetadata.rename_features

FeatureMetadata.to_dict

FeatureMetadata.verify_data

FeatureMetadata.verify_data_subset

FeatureMetadata.verify_data_superset

Search Spaces

      

    

  

      

          

          Back to top

          

    
    View this page

            
              
Toggle Light / Dark / Auto color theme

              

            

          
            
Toggle table of contents sidebar

          

        

TabularPredictor.fit¶

TabularPredictor.fit(train_data: DataFrame | str, tuning_data: DataFrame | str = None, time_limit: float = None, presets: list[str] | str = None, hyperparameters: dict | str = None, feature_metadata='infer', infer_limit: float = None, infer_limit_batch_size: int = None, fit_weighted_ensemble: bool = True, fit_full_last_level_weighted_ensemble: bool = True, full_weighted_ensemble_additionally: bool = False, dynamic_stacking: bool | str = False, calibrate_decision_threshold: bool | str = 'auto', num_cpus: int | str = 'auto', num_gpus: int | str = 'auto', fit_strategy: Literal['sequential', 'parallel'] = 'sequential', memory_limit: float | str = 'auto', callbacks: list[AbstractCallback] = None, **kwargs) → TabularPredictor[source]¶

Fit models to predict a column of a data table (label) based on the other columns (features).

Parameters:

train_data (pd.DataFrame or str) – Table of the training data as a pandas DataFrame.
If str is passed, train_data will be loaded using the str value as the file path.

tuning_data (pd.DataFrame or str, optional) – Another dataset containing validation data reserved for tuning processes such as early stopping and hyperparameter tuning.
This dataset should be in the same format as train_data.
If str is passed, tuning_data will be loaded using the str value as the file path.
Note: final model returned may be fit on tuning_data as well as train_data. Do not provide your evaluation test data here!
In particular, when num_bag_folds > 0 or num_stack_levels > 0, models will be trained on both tuning_data and train_data.
If tuning_data = None, fit() will automatically hold out some random validation examples from train_data.

time_limit (int, default = None) – Approximately how long fit() should run for (wallclock time in seconds).
If not specified, fit() will run until all models have completed training, but will not repeatedly bag models unless num_bag_sets is specified.

presets (list or str or dict, default = ['medium_quality']) – 
List of preset configurations for various arguments in fit(). Can significantly impact predictive accuracy, memory-footprint, and inference latency of trained models, and various other properties of the returned predictor.
It is recommended to specify presets and avoid specifying most other fit() arguments or model hyperparameters prior to becoming familiar with AutoGluon.
As an example, to get the most accurate overall predictor (regardless of its efficiency), set presets=’best_quality’.
To get good quality with minimal disk usage, set presets=[‘good_quality’, ‘optimize_for_deployment’]
Any user-specified arguments in fit() will override the values used by presets.
If specifying a list of presets, later presets will override earlier presets if they alter the same argument.
For precise definitions of the provided presets, see file: autogluon/tabular/configs/presets_configs.py.
Users can specify custom presets by passing in a dictionary of argument values as an element to the list.

Available Presets: [‘best_quality’, ‘high_quality’, ‘good_quality’, ‘medium_quality’, ‘experimental_quality’, ‘optimize_for_deployment’, ‘interpretable’, ‘ignore_text’]

It is recommended to only use one quality based preset in a given call to fit() as they alter many of the same arguments and are not compatible with each-other.

In-depth Preset Info:
best_quality={‘auto_stack’: True, ‘dynamic_stacking’: ‘auto’, ‘hyperparameters’: ‘zeroshot’}
Best predictive accuracy with little consideration to inference time or disk usage. Achieve even better results by specifying a large time_limit value.
Recommended for applications that benefit from the best possible model accuracy.

high_quality={‘auto_stack’: True, ‘dynamic_stacking’: ‘auto’, ‘hyperparameters’: ‘zeroshot’, ‘refit_full’: True, ‘set_best_to_refit_full’: True, ‘save_bag_folds’: False}
High predictive accuracy with fast inference. ~8x faster inference and ~8x lower disk usage than best_quality.
Recommended for applications that require reasonable inference speed and/or model size.

good_quality={‘auto_stack’: True, ‘dynamic_stacking’: ‘auto’, ‘hyperparameters’: ‘light’, ‘refit_full’: True, ‘set_best_to_refit_full’: True, ‘save_bag_folds’: False}
Good predictive accuracy with very fast inference. ~4x faster inference and ~4x lower disk usage than high_quality.
Recommended for applications that require fast inference speed.

medium_quality={‘auto_stack’: False}
Medium predictive accuracy with very fast inference and very fast training time. ~20x faster training than good_quality.
This is the default preset in AutoGluon, but should generally only be used for quick prototyping, as good_quality results in significantly better predictive accuracy and faster inference time.

experimental_quality={‘auto_stack’: True, ‘dynamic_stacking’: ‘auto’, ‘hyperparameters’: ‘experimental’, ‘fit_strategy’: ‘parallel’, ‘num_gpus’: 0}
This preset acts as a testing ground for cutting edge features and models which could later be added to the best_quality preset in future releases.
Recommended when best_quality was already being used and the user wants to push performance even further.

optimize_for_deployment={‘keep_only_best’: True, ‘save_space’: True}
Optimizes result immediately for deployment by deleting unused models and removing training artifacts.
Often can reduce disk usage by ~2-4x with no negatives to model accuracy or inference speed.
This will disable numerous advanced functionality, but has no impact on inference.
This will make certain functionality less informative, such as predictor.leaderboard() and predictor.fit_summary().

Because unused models will be deleted under this preset, methods like predictor.leaderboard() and predictor.fit_summary() will no longer show the full set of models that were trained during fit().

Recommended for applications where the inner details of AutoGluon’s training is not important and there is no intention of manually choosing between the final models.
This preset pairs well with the other presets such as good_quality to make a very compact final model.
Identical to calling predictor.delete_models(models_to_keep=’best’, dry_run=False) and predictor.save_space() directly after fit().

interpretable={‘auto_stack’: False, ‘hyperparameters’: ‘interpretable’}
Fits only interpretable rule-based models from the imodels package.
Trades off predictive accuracy for conciseness.

ignore_text={‘_feature_generator_kwargs’: {‘enable_text_ngram_features’: False, ‘enable_text_special_features’: False, ‘enable_raw_text_features’: False}}
Disables automated feature generation when text features are detected.
This is useful to determine how beneficial text features are to the end result, as well as to ensure features are not mistaken for text when they are not.
Ignored if feature_generator was also specified.

hyperparameters (str or dict, default = 'default') – 
Determines the hyperparameters used by the models.
If str is passed, will use a preset hyperparameter configuration.

Valid str options: [‘default’, ‘zeroshot’, ‘light’, ‘very_light’, ‘toy’, ‘multimodal’]
’default’: Default AutoGluon hyperparameters intended to get strong accuracy with reasonable disk usage and inference time. Used in the ‘medium_quality’ preset.
‘zeroshot’: A powerful model portfolio learned from TabRepo’s ensemble simulation on 200 datasets. Contains ~100 models and is used in ‘best_quality’ and ‘high_quality’ presets.
‘light’: Results in smaller models. Generally will make inference speed much faster and disk usage much lower, but with worse accuracy. Used in the ‘good_quality’ preset.
‘very_light’: Results in much smaller models. Behaves similarly to ‘light’, but in many cases with over 10x less disk usage and a further reduction in accuracy.
‘toy’: Results in extremely small models. Only use this when prototyping, as the model quality will be severely reduced.
‘multimodal’: [EXPERIMENTAL] Trains a multimodal transformer model alongside tabular models. Requires that some text columns appear in the data and GPU.

When combined with ‘best_quality’ presets option, this can achieve extremely strong results in multimodal data tables that contain columns with text in addition to numeric/categorical columns.

Reference autogluon/tabular/configs/hyperparameter_configs.py for information on the hyperparameters associated with each preset.

Keys are strings that indicate which model types to train.
Stable model options include:
’GBM’ (LightGBM)
‘CAT’ (CatBoost)
‘XGB’ (XGBoost)
‘RF’ (random forest)
‘XT’ (extremely randomized trees)
‘KNN’ (k-nearest neighbors)
‘LR’ (linear regression)
‘NN_TORCH’ (neural network implemented in Pytorch)
‘FASTAI’ (neural network with FastAI backend)
‘AG_AUTOMM’ (MultimodalPredictor from autogluon.multimodal. Supports Tabular, Text, and Image modalities. GPU is required.)

Experimental model options include:
’FT_TRANSFORMER’ (Tabular Transformer, GPU is recommended. Does not scale well to >100 features.)
‘FASTTEXT’ (FastText. Note: Has not been tested for a long time.)
‘TABPFN’ (TabPFN. Does not scale well to >100 features or >1000 rows, and does not support regression. Extremely slow inference speed.)
‘AG_TEXT_NN’ (Multimodal Text+Tabular model, GPU is required. Recommended to instead use its successor, ‘AG_AUTOMM’.)
‘AG_IMAGE_NN’ (Image model, GPU is required. Recommended to instead use its successor, ‘AG_AUTOMM’.)

If a certain key is missing from hyperparameters, then fit() will not train any models of that type. Omitting a model key from hyperparameters is equivalent to including this model key in excluded_model_types.
For example, set hyperparameters = { ‘NN_TORCH’:{…} } if say you only want to train (PyTorch) neural networks and no other types of models.

Values = dict of hyperparameter settings for each model type, or list of dicts.
Each hyperparameter can either be a single fixed value or a search space containing many possible values.
Unspecified hyperparameters will be set to default values (or default search spaces if hyperparameter_tune_kwargs=’auto’).
Caution: Any provided search spaces will error if hyperparameter_tune_kwargs=None (Default).
To train multiple models of a given type, set the value to a list of hyperparameter dictionaries.

For example, hyperparameters = {‘RF’: [{‘criterion’: ‘gini’}, {‘criterion’: ‘entropy’}]} will result in 2 random forest models being trained with separate hyperparameters.

Advanced functionality: Bring your own model / Custom model support
AutoGluon fully supports custom models. For a detailed tutorial on creating and using custom models with AutoGluon, refer to https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-model.html

Advanced functionality: Custom stack levels
By default, AutoGluon reuses the same models and model hyperparameters at each level during stack ensembling.
To customize this behaviour, create a hyperparameters dictionary separately for each stack level, and then add them as values to a new dictionary, with keys equal to the stack level.

Example: hyperparameters = {1: {‘RF’: rf_params1}, 2: {‘CAT’: [cat_params1, cat_params2], ‘NN_TORCH’: {}}}
This will result in a stack ensemble that has one custom random forest in level 1 followed by two CatBoost models with custom hyperparameters and a default neural network in level 2, for a total of 4 models.

If a level is not specified in hyperparameters, it will default to using the highest specified level to train models. This can also be explicitly controlled by adding a ‘default’ key.

Default:
hyperparameters = {
‘NN_TORCH’: {},
‘GBM’: [

{‘extra_trees’: True, ‘ag_args’: {‘name_suffix’: ‘XT’}},
{},
{

”learning_rate”: 0.03,
“num_leaves”: 128,
“feature_fraction”: 0.9,
“min_data_in_leaf”: 3,
“ag_args”: {“name_suffix”: “Large”, “priority”: 0, “hyperparameter_tune_kwargs”: None},

},

],
‘CAT’: {},
‘XGB’: {},
‘FASTAI’: {},
‘RF’: [

{‘criterion’: ‘gini’, ‘ag_args’: {‘name_suffix’: ‘Gini’, ‘problem_types’: [‘binary’, ‘multiclass’]}},
{‘criterion’: ‘entropy’, ‘ag_args’: {‘name_suffix’: ‘Entr’, ‘problem_types’: [‘binary’, ‘multiclass’]}},
{‘criterion’: ‘squared_error’, ‘ag_args’: {‘name_suffix’: ‘MSE’, ‘problem_types’: [‘regression’]}},

],
‘XT’: [

{‘criterion’: ‘gini’, ‘ag_args’: {‘name_suffix’: ‘Gini’, ‘problem_types’: [‘binary’, ‘multiclass’]}},
{‘criterion’: ‘entropy’, ‘ag_args’: {‘name_suffix’: ‘Entr’, ‘problem_types’: [‘binary’, ‘multiclass’]}},
{‘criterion’: ‘squared_error’, ‘ag_args’: {‘name_suffix’: ‘MSE’, ‘problem_types’: [‘regression’]}},

],
‘KNN’: [

{‘weights’: ‘uniform’, ‘ag_args’: {‘name_suffix’: ‘Unif’}},
{‘weights’: ‘distance’, ‘ag_args’: {‘name_suffix’: ‘Dist’}},

],

}

Details regarding the hyperparameters you can specify for each model are provided in the following files:
NN: autogluon.tabular.models.tabular_nn.hyperparameters.parameters
Note: certain hyperparameter settings may cause these neural networks to train much slower.

GBM: autogluon.tabular.models.lgb.hyperparameters.parameters
See also the lightGBM docs: https://lightgbm.readthedocs.io/en/latest/Parameters.html

CAT: autogluon.tabular.models.catboost.hyperparameters.parameters
See also the CatBoost docs: https://catboost.ai/docs/concepts/parameter-tuning.html

XGB: autogluon.tabular.models.xgboost.hyperparameters.parameters
See also the XGBoost docs: https://xgboost.readthedocs.io/en/latest/parameter.html

FASTAI: autogluon.tabular.models.fastainn.hyperparameters.parameters
See also the FastAI docs: https://docs.fast.ai/tabular.learner.html

RF: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
Note: Hyperparameter tuning is disabled for this model.

XT: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
Note: Hyperparameter tuning is disabled for this model.

KNN: See sklearn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
Note: Hyperparameter tuning is disabled for this model.

LR: autogluon.tabular.models.lr.hyperparameters.parameters
Note: Hyperparameter tuning is disabled for this model.
Note: ‘penalty’ parameter can be used for regression to specify regularization method: ‘L1’ and ‘L2’ values are supported.

Advanced functionality: Custom AutoGluon model arguments
These arguments are optional and can be specified in any model’s hyperparameters.
Example: hyperparameters = {‘RF’: {…, ‘ag_args’: {‘name_suffix’: ‘CustomModelSuffix’, ‘disable_in_hpo’: True}}

ag_args: Dictionary of customization options related to meta properties of the model such as its name, the order it is trained, the problem types it is valid for, and the type of HPO it utilizes.
Valid keys:
name: (str) The name of the model. This overrides AutoGluon’s naming logic and all other name arguments if present.
name_main: (str) The main name of the model. Example: ‘RandomForest’.
name_prefix: (str) Add a custom prefix to the model name. Unused by default.
name_suffix: (str) Add a custom suffix to the model name. Unused by default.
priority: (int) Determines the order in which the model is trained. Larger values result in the model being trained earlier. Default values range from 100 (KNN) to 0 (custom), dictated by model type. If you want this model to be trained first, set priority = 999.
problem_types: (list) List of valid problem types for the model. problem_types=[‘binary’] will result in the model only being trained if problem_type is ‘binary’.
disable_in_hpo: (bool) If True, the model will only be trained if hyperparameter_tune_kwargs=None.
valid_stacker: (bool) If False, the model will not be trained as a level 2 or higher stacker model.
valid_base: (bool) If False, the model will not be trained as a level 1 (base) model.
hyperparameter_tune_kwargs: (dict) Refer to TabularPredictor.fit() hyperparameter_tune_kwargs argument. If specified here, will override global HPO settings for this model.

Reference the default hyperparameters for example usage of these options.

ag_args_fit: Dictionary of model fit customization options related to how and with what constraints the model is trained. These parameters affect stacker fold models, but not stacker models themselves.
Clarification: time_limit is the internal time in seconds given to a particular model to train, which is dictated in part by the time_limit argument given during predictor.fit() but is not the same.
Valid keys:

stopping_metric: (str or autogluon.core.metrics.Scorer, default=None) The metric to use for early stopping of the model. If None, model will decide.
max_memory_usage_ratio: (float, default=1.0) The ratio of memory usage relative to the default to allow before early stopping or killing the model. Values greater than 1.0 will be increasingly prone to out-of-memory errors.
max_time_limit_ratio: (float, default=1.0) The ratio of the provided time_limit to use during model fit(). If time_limit=10 and max_time_limit_ratio=0.3, time_limit would be changed to 3. Does not alter max_time_limit or min_time_limit values.
max_time_limit: (float, default=None) Maximum amount of time to allow this model to train for (in sec). If the provided time_limit is greater than this value, it will be replaced by max_time_limit.
min_time_limit: (float, default=0) Allow this model to train for at least this long (in sec), regardless of the time limit it would otherwise be granted.

If min_time_limit >= max_time_limit, time_limit will be set to min_time_limit.
If min_time_limit=None, time_limit will be set to None and the model will have no training time restriction.

num_cpus(int or str, default=’auto’)
How many CPUs to use during model fit.
If ‘auto’, model will decide.

num_gpus(int or str, default=’auto’)
How many GPUs to use during model fit.
If ‘auto’, model will decide. Some models can use GPUs but don’t by default due to differences in model quality.
Set to 0 to disable usage of GPUs.

ag_args_ensemble: Dictionary of hyperparameters shared by all models that control how they are ensembled, if bag mode is enabled.
Valid keys:
use_orig_features: [True, False, “never”], default True
Whether a stack model will use the original features along with the stack features to train (akin to skip-connections).
If True, will use the original data features.
If False, will discard the original data features and only use stack features, except when no stack features exist (such as in layer 1).
If “never”, will always discard the original data features. Will be skipped in layer 1.

valid_stackerbool, default True
If True, will be marked as valid to include as a stacker model.
If False, will only be fit as a base model (layer 1) and will not be fit in stack layers (layer 2+).

max_base_modelsint, default 0
Maximum number of base models whose predictions form the features input to this stacker model.
If more than max_base_models base models are available, only the top max_base_models models with highest validation score are used.
If 0, the logic is skipped.

max_base_models_per_typeint | str, default “auto”
Similar to max_base_models. If more than max_base_models_per_type of any particular model type are available,
only the top max_base_models_per_type of that type are used. This occurs before the max_base_models filter.
If “auto”, the value will be adaptively set based on the number of training samples.

More samples will lead to larger values, starting at 1 with <1000 samples, increasing up to 12 at >=50000 samples.

If 0, the logic is skipped.

num_folds: (int, default=None) If specified, the number of folds to fit in the bagged model.
If specified, overrides any other value used to determine the number of folds such as predictor.fit num_bag_folds argument.

max_sets: (int, default=None) If specified, the maximum sets to fit in the bagged model.
The lesser of max_sets and the predictor.fit num_bag_sets argument will be used for the given model.
Useful if a particular model is expensive relative to others and you want to avoid repeated bagging of the expensive model while still repeated bagging the cheaper models.

save_bag_folds: (bool, default=True)
If True, bagged models will save their fold models (the models from each individual fold of bagging). This is required to use bagged models for prediction.
If False, bagged models will not save their fold models. This means that bagged models will not be valid models during inference.

This should only be set to False when planning to call predictor.refit_full() or when refit_full is set and set_best_to_refit_full=True.
Particularly useful if disk usage is a concern. By not saving the fold models, bagged models will use only very small amounts of disk space during training.
In many training runs, this will reduce peak disk usage by >10x.

fold_fitting_strategy: (AbstractFoldFittingStrategy default=auto) Whether to fit folds in parallel or in sequential order.
If parallel_local, folds will be trained in parallel with evenly distributed computing resources. This could bring 2-4x speedup compared to SequentialLocalFoldFittingStrategy, but could consume much more memory.
If sequential_local, folds will be trained in sequential.
If auto, strategy will be determined by OS and whether ray is installed or not. MacOS support for parallel_local is unstable, and may crash if enabled.

num_folds_parallel: (int or str, default=’auto’) Number of folds to be trained in parallel if using ParallelLocalFoldFittingStrategy. Consider lowering this value if you encounter either out of memory issue or CUDA out of memory issue(when trained on gpu).
if ‘auto’, will try to train all folds in parallel.

feature_metadata (autogluon.tabular.FeatureMetadata or str, default = ‘infer’) – The feature metadata used in various inner logic in feature preprocessing.
If ‘infer’, will automatically construct a FeatureMetadata object based on the properties of train_data.
In this case, train_data is input into autogluon.tabular.FeatureMetadata.from_df() to infer feature_metadata.
If ‘infer’ incorrectly assumes the dtypes of features, consider explicitly specifying feature_metadata.

infer_limit (float, default = None) – The inference time limit in seconds per row to adhere to during fit.
If infer_limit=0.05 and infer_limit_batch_size=1000, AutoGluon will avoid training models that take longer than 50 ms/row to predict when given a batch of 1000 rows to predict (must predict 1000 rows in no more than 50 seconds).
If bagging is enabled, the inference time limit will be respected based on estimated inference speed of _FULL models after refit_full is called, NOT on the inference speed of the bagged ensembles.
The inference times calculated for models are assuming predictor.persist(‘all’) is called after fit.
If None, no limit is enforced.
If it is impossible to satisfy the constraint, an exception will be raised.

infer_limit_batch_size (int, default = None) – The batch size to use when predicting in bulk to estimate per-row inference time.
Must be an integer greater than 0.
If None and infer_limit is specified, will default to 10000.
It is recommended to set to 10000 unless you must satisfy an online-inference scenario.
Small values, especially infer_limit_batch_size=1, will result in much larger per-row inference times and should be avoided if possible.
Refer to infer_limit for more details on how this is used.
If specified when infer_limit=None, the inference time will be logged during training but will not be limited.

fit_weighted_ensemble (bool, default = True) – If True, a WeightedEnsembleModel will be fit in each stack layer.
A weighted ensemble will often be stronger than an individual model while being very fast to train.
It is recommended to keep this value set to True to maximize predictive quality.

fit_full_last_level_weighted_ensemble (bool, default = True) – If True, the WeightedEnsembleModel of the last stacking level will be fit with all (successful) models from all previous layers as base models.
If stacking is disabled, settings this to True or False makes no difference because the WeightedEnsembleModel L2 always uses all models from L1.
It is recommended to keep this value set to True to maximize predictive quality.

full_weighted_ensemble_additionally (bool, default = False) – If True, AutoGluon will fit two WeightedEnsembleModels after training all stacking levels. Setting this to True, simulates calling
fit_weighted_ensemble() after calling fit(). Has no affect if fit_full_last_level_weighted_ensemble is False and does not fit an additional
WeightedEnsembleModel if stacking is disabled.

dynamic_stacking (bool | str, default = False) – If True and num_stack_levels > 0, AutoGluon will dynamically determine whether to use stacking or not by first validating AutoGluon’s stacking
behavior. This is done to avoid so-called stacked overfitting that can make traditional multi-layer stacking, as used in AutoGluon, fail drastically
and produce unreliable validation scores.
It is recommended to keep this value set to True or “auto” when using stacking,
as long as it is unknown whether the data is affected by stacked overfitting.
If it is known that the data is unaffected by stacked overfitting, then setting this value to False is expected to maximize predictive quality.
If enabled, by default, AutoGluon performs dynamic stacking by spending 25% of the provided time limit for detection and all remaining
time for fitting AutoGluon. This can be adjusted by specifying ds_args with different parameters to fit().
If “auto”, will be set to not use_bag_holdout.
See the documentation of ds_args for more information.

calibrate_decision_threshold (bool | str, default = "auto") – If True, will automatically calibrate the decision threshold at the end of fit for calls to .predict based on the evaluation metric.
If “auto”, will be set to True if eval_metric.needs_class=True and problem_type=”binary”.
By default, the decision threshold is 0.5, however for some metrics such as f1 and balanced_accuracy,
scores can be significantly improved by choosing a threshold other than 0.5.
Only valid for problem_type=’binary’. Ignored for all other problem types.

num_cpus (int | str, default = "auto") – The total amount of cpus you want AutoGluon predictor to use.
Auto means AutoGluon will make the decision based on the total number of cpus available and the model requirement for best performance.
Users generally don’t need to set this value

num_gpus (int | str, default = "auto") – The total amount of gpus you want AutoGluon predictor to use.
Auto means AutoGluon will make the decision based on the total number of gpus available and the model requirement for best performance.
Users generally don’t need to set this value

fit_strategy (Literal["sequential", "parallel"], default = "sequential") – 
The strategy used to fit models.
If “sequential”, models will be fit sequentially. This is the most stable option with the most readable logging.
If “parallel”, models will be fit in parallel with ray, splitting available compute between them.

Note: “parallel” is experimental and may run into issues. It was first added in version 1.2.0.

For machines with 16 or more CPU cores, it is likely that “parallel” will be faster than “sequential”.

Added in version 1.2.0.

memory_limit (float | str, default = "auto") – The total amount of memory in GB you want AutoGluon predictor to use. “auto” means AutoGluon will use all available memory on the system
(that is detectable by psutil).
Note that this is only a soft limit! AutoGluon uses this limit to skip training models that are expected to require too much memory or stop
training a model that would exceed the memory limit. AutoGluon does not guarantee the enforcement of this limit (yet). Nevertheless, we expect
AutoGluon to abide by the limit in most cases or, at most, go over the limit by a small margin.
For most virtualized systems (e.g., in the cloud) and local usage on a server or laptop, “auto” is ideal for this parameter. We recommend manually
setting the memory limit (and any other resources) on systems with shared resources that are controlled by the operating system (e.g., SLURM and
cgroups). Otherwise, AutoGluon might wrongly assume more resources are available for fitting a model than the operating system allows,
which can result in model training failing or being very inefficient.

callbacks (list[AbstractCallback], default = None) – 
:::{warning}
Callbacks are an experimental feature and may change in future releases without warning.
Callback support is preliminary and targeted towards developers.
:::
A list of callback objects inheriting from autogluon.core.callbacks.AbstractCallback.
These objects will be called before and after each model fit within trainer.
They have the ability to skip models or early stop the training process.
They can also theoretically change the entire logical flow of the trainer code by interacting with the passed trainer object.
For more details, refer to AbstractCallback source code.
If None, no callback objects will be used.

[Note] Callback objects can be mutated in-place by the fit call if they are stateful.
Ensure that you avoid reusing a mutated callback object between multiple fit calls.

[Note] Callback objects are deleted from trainer at the end of the fit call. They will not impact operations such as refit_full or fit_extra.

**kwargs – 
auto_stackbool, default = False
Whether AutoGluon should automatically utilize bagging and multi-layer stack ensembling to boost predictive accuracy.
Set this = True if you are willing to tolerate longer training times in order to maximize predictive accuracy!
Automatically sets num_bag_folds and num_stack_levels arguments based on dataset properties.
Note: Setting num_bag_folds and num_stack_levels arguments will override auto_stack.
Note: This can increase training time (and inference time) by up to 20x, but can greatly improve predictive performance.

num_bag_foldsint, default = None
Number of folds used for bagging of models. When num_bag_folds = k, training time is roughly increased by a factor of k (set = 0 to disable bagging).
Disabled by default (0), but we recommend values between 5-10 to maximize predictive performance.
Increasing num_bag_folds will result in models with lower bias but that are more prone to overfitting.
num_bag_folds = 1 is an invalid value, and will raise a ValueError.
Values > 10 may produce diminishing returns, and can even harm overall results due to overfitting.
To further improve predictions, avoid increasing num_bag_folds much beyond 10 and instead increase num_bag_sets.

num_bag_setsint, default = None
Number of repeats of kfold bagging to perform (values must be >= 1). Total number of models trained during bagging = num_bag_folds * num_bag_sets.
Defaults to 1 when unspecified. Value is ignored if num_bag_folds<=2.
Values greater than 1 will result in superior predictive performance, especially on smaller problems and with stacking enabled (reduces overall variance).
Be warned: This will drastically increase overall runtime, and if using a time limit, can very commonly lead to worse performance.
It is recommended to increase this value only as a last resort, as it is the least computationally efficient method to improve performance.

num_stack_levelsint, default = None
Number of stacking levels to use in stack ensemble. Roughly increases model training time by factor of num_stack_levels+1 (set = 0 to disable stack ensembling).
Disabled by default (0), but we recommend num_stack_levels=1 to maximize predictive performance.
To prevent overfitting, num_bag_folds >= 2 must also be set or else a ValueError will be raised.

delay_bag_setsbool, default = False
Controls when repeats of kfold bagging are executed in AutoGluon when under a time limit.
We suggest sticking to False to avoid overfitting.

If True, AutoGluon delays repeating kfold bagging until after evaluating all models
from hyperparameters, if there is enough time. This allows AutoGluon to explore
more hyperparameters to obtain a better final performance but it may lead to
more overfitting.

If False, AutoGluon repeats kfold bagging immediately after evaluating each model.
Thus, AutoGluon might evaluate fewer models with less overfitting.

holdout_fracfloat, default = None
Fraction of train_data to holdout as tuning data for optimizing hyperparameters (ignored unless tuning_data = None, ignored if num_bag_folds != 0 unless use_bag_holdout == True).
Default value (if None) is selected based on the number of rows in the training data. Default values range from 0.2 at 2,500 rows to 0.01 at 250,000 rows.
Default value is doubled if hyperparameter_tune_kwargs is set, up to a maximum of 0.2.
Disabled if num_bag_folds >= 2 unless use_bag_holdout == True.

use_bag_holdoutbool | str, default = False
If True, a holdout_frac portion of the data is held-out from model bagging.
This held-out data is only used to score models and determine weighted ensemble weights.
Enable this if there is a large gap between score_val and score_test in stack models.
Note: If tuning_data was specified, tuning_data is used as the holdout data.
Disabled if not bagging.
If “auto”, will be set to True if the training data has >= 1000000 rows, else it will be set to False.

hyperparameter_tune_kwargsstr or dict, default = None
Hyperparameter tuning strategy and kwargs (for example, how many HPO trials to run).
If None, then hyperparameter tuning will not be performed.
You can either choose to provide a preset

Valid preset values:
’auto’: Performs HPO via bayesian optimization search on NN_TORCH and FASTAI models, and random search on other models using local scheduler.
‘random’: Performs HPO via random search using local scheduler.

Or provide a dict to specify searchers and schedulers
Valid keys:
’num_trials’: How many HPO trials to run
‘scheduler’: Which scheduler to use

Valid values:
’local’: Local scheduler that schedule trials FIFO

’searcher’: Which searching algorithm to use
’local_random’: Uses the ‘random’ searcher
‘random’: Perform random search
‘auto’: Perform bayesian optimization search on NN_TORCH and FASTAI models. Perform random search on other models.

The ‘scheduler’ and ‘searcher’ key are required when providing a dict.
An example of a valid dict:

hyperparameter_tune_kwargs = {
‘num_trials’: 5,
‘scheduler’ : ‘local’,
‘searcher’: ‘auto’,

}

feature_prune_kwargs: dict, default = None
Performs layer-wise feature pruning via recursive feature elimination with permutation feature importance.
This fits all models in a stack layer once, discovers a pruned set of features, fits all models in the stack layer
again with the pruned set of features, and updates input feature lists for models whose validation score improved.
If None, do not perform feature pruning. If empty dictionary, perform feature pruning with default configurations.
For valid dictionary keys, refer to autogluon.core.utils.feature_selection.FeatureSelector and
autogluon.core.trainer.abstract_trainer.AbstractTabularTrainer._proxy_model_feature_prune documentation.
To force all models to work with the pruned set of features, set force_prune=True in the dictionary.

ag_argsdict, default = None
Keyword arguments to pass to all models (i.e. common hyperparameters shared by all AutoGluon models).
See the ag_args argument from “Advanced functionality: Custom AutoGluon model arguments” in the hyperparameters argument documentation for valid values.
Identical to specifying ag_args parameter for all models in hyperparameters.
If a key in ag_args is already specified for a model in hyperparameters, it will not be altered through this argument.

ag_args_fitdict, default = None
Keyword arguments to pass to all models.
See the ag_args_fit argument from “Advanced functionality: Custom AutoGluon model arguments” in the hyperparameters argument documentation for valid values.
Identical to specifying ag_args_fit parameter for all models in hyperparameters.
If a key in ag_args_fit is already specified for a model in hyperparameters, it will not be altered through this argument.

ag_args_ensembledict, default = None
Keyword arguments to pass to all models.
See the ag_args_ensemble argument from “Advanced functionality: Custom AutoGluon model arguments” in the hyperparameters argument documentation for valid values.
Identical to specifying ag_args_ensemble parameter for all models in hyperparameters.
If a key in ag_args_ensemble is already specified for a model in hyperparameters, it will not be altered through this argument.

ds_argsdict, see below for default
Keyword arguments for dynamic stacking, only used if dynamic_stacking=True. These keyword arguments control the behavior of dynamic stacking
and determine how AutoGluon tries to detect stacked overfitting. To detect stacked overfitting, AutoGluon will fit itself (so called sub-fits)
on a subset (for holdout) or multiple subsets (for repeated cross-validation) and use the predictions of AutoGluon on the validation data to
detect stacked overfitting. The sub-fits stop and stacking will be disabled if any sub-fit shows stacked overfitting.
Allowed keys and values are:

detection_time_fracfloat in (0,1), default = 1/4
Determines how much of the original training time is used for detecting stacked overfitting.
When using (repeated) cross-validation, each sub-fit will be fit for 1/n_splits * detection_time_frac * time_limit.
If no time limit is given to AutoGluon, this parameter is ignored and AutoGluon is fit without a time limit in the sub-fit.

validation_procedure: str, default = ‘holdout’
Determines the validation procedure used to detect stacked overfitting. Can be either cv or holdout.
If validation_procedure=’holdout’ and holdout_data is not specified (default), then holdout_frac determines the holdout data.
If validation_procedure=’holdout’ and holdout_data is specified, then the provided holdout_data is used for validation.
If validation_procedure=’cv’, n_folds and n_repeats determine the kind cross-validation procedure.

holdout_fracfloat in (0,1), default = 1/9
Determines how much of the original training data is used for the holdout data during holdout validation.
Ignored if holdout_data is not None.

n_foldsint in [2, +inf), default = 2
Number of folds to use for cross-validation.

n_repeatsint [1, +inf), default = 1
Number of repeats to use for repeated cross-validation. If set to 1, performs 1-repeated cross-validation which is equivalent to
cross-validation without repeats.

memory_safe_fitsbool, default = True
If True, AutoGluon runs each sub-fit in a ray-based subprocess to avoid memory leakage that exist due to Python’s lackluster
garbage collector.

clean_up_fitsbool, default = True
If True, AutoGluon will remove all saved information from sub-fits from disk.
If False, the sub-fits are kept on disk and self._sub_fits will store paths to the sub-fits, which can be loaded just like any other
predictor from disk using TabularPredictor.load().

enable_ray_loggingbool, default = True
If True, will log the dynamic stacking sub-fit when ray is used (memory_safe_fits=True).
Note that because of how ray works, this may cause extra unwanted logging in the main fit process after dynamic stacking completes.

enable_callbacksbool, default = False
If True, will perform a deepcopy on the specified user callbacks and enable them during the DyStack call.
If False, will not include callbacks in the DyStack call.

holdout_data: str or pd.DataFrame, default = None
Another dataset containing validation data reserved for detecting stacked overfitting. This dataset should be in the same format as
train_data. If str is passed, holdout_data will be loaded using the str value as the file path.
If holdout_data is not None, the sub-fit is fit on all of train_data and the full fit is fit on all of train_data and
holdout_data combined.

included_model_typeslist, default = None
To only include listed model types for training during fit().
Models that are listed in included_model_types but not in hyperparameters will be ignored.
Reference hyperparameters documentation for what models correspond to each value.
Useful when only a subset of model needs to be trained and the hyperparameters dictionary is difficult or time-consuming.

Example: To include both ‘GBM’ and ‘FASTAI’ models, specify included_model_types=[‘GBM’, ‘FASTAI’].

excluded_model_typeslist, default = None
Banned subset of model types to avoid training during fit(), even if present in hyperparameters.
Reference hyperparameters documentation for what models correspond to each value.
Useful when a particular model type such as ‘KNN’ or ‘custom’ is not desired but altering the hyperparameters dictionary is difficult or time-consuming.

Example: To exclude both ‘KNN’ and ‘custom’ models, specify excluded_model_types=[‘KNN’, ‘custom’].

refit_fullbool or str, default = False
Whether to retrain all models on all of the data (training + validation) after the normal training procedure.
This is equivalent to calling predictor.refit_full(model=refit_full) after fit.
If refit_full=True, it will be treated as refit_full=’all’.
If refit_full=False, refitting will not occur.
Valid str values:

all: refits all models.
best: refits only the best model (and its ancestors if it is a stacker model).
{model_name}: refits only the specified model (and its ancestors if it is a stacker model).

For bagged models:
Reduces a model’s inference time by collapsing bagged ensembles into a single model fit on all of the training data.
This process will typically result in a slight accuracy reduction and a large inference speedup.
The inference speedup will generally be between 10-200x faster than the original bagged ensemble model.

The inference speedup factor is equivalent to (k * n), where k is the number of folds (num_bag_folds) and n is the number of finished repeats (num_bag_sets) in the bagged ensemble.

The runtime is generally 10% or less of the original fit runtime.
The runtime can be roughly estimated as 1 / (k * n) of the original fit runtime, with k and n defined above.

For non-bagged models:
Optimizes a model’s accuracy by retraining on 100% of the data without using a validation set.
Will typically result in a slight accuracy increase and no change to inference time.
The runtime will be approximately equal to the original fit runtime.

This process does not alter the original models, but instead adds additional models.
If stacker models are refit by this process, they will use the refit_full versions of the ancestor models during inference.
Models produced by this process will not have validation scores, as they use all of the data for training.

Therefore, it is up to the user to determine if the models are of sufficient quality by including test data in predictor.leaderboard(test_data).
If the user does not have additional test data, they should reference the original model’s score for an estimate of the performance of the refit_full model.

Warning: Be aware that utilizing refit_full models without separately verifying on test data means that the model is untested, and has no guarantee of being consistent with the original model.

The time taken by this process is not enforced by time_limit.

save_bag_foldsbool, default = True
If True, will save the bagged fold models to disk.
If False, will not save the bagged fold models, only keeping their metadata and out-of-fold predictions.

Note: The bagged models will not be available for prediction, only use this if you intend to call refit_full.
The purpose of setting it to False is that it greatly decreases the peak disk usage of the predictor during the fit call when bagging.
Note that this makes refit_full slightly more likely to crash in scenarios where the dataset is large relative to available system memory.
This is because by default, refit_full will fall back to cloning the first fold of the bagged model in case it lacks memory to refit.
However, if save_bag_folds=False, this fallback isn’t possible, as there is not fold model to clone because it wasn’t saved.
In this scenario, refit will raise an exception for save_bag_folds=False, but will succeed if save_bag_folds=True.

Final disk usage of predictor will be identical regardless of the setting after predictor.delete_models(models_to_keep=”best”, dry_run=False) is called post-fit.

set_best_to_refit_fullbool, default = False
If True, will change the default model that Predictor uses for prediction when model is not specified to the refit_full version of the model that exhibited the highest validation score.
Only valid if refit_full is set.

keep_only_bestbool, default = False
If True, only the best model and its ancestor models are saved in the outputted predictor. All other models are deleted.
If you only care about deploying the most accurate predictor with the smallest file-size and no longer need any of the other trained models or functionality beyond prediction on new data, then set: keep_only_best=True, save_space=True.
This is equivalent to calling predictor.delete_models(models_to_keep=’best’, dry_run=False) directly after fit().

If used with refit_full and set_best_to_refit_full, the best model will be the refit_full model, and the original bagged best model will be deleted.
refit_full will be automatically set to ‘best’ in this case to avoid training models which will be later deleted.

save_spacebool, default = False
If True, reduces the memory and disk size of predictor by deleting auxiliary model files that aren’t needed for prediction on new data.
This is equivalent to calling predictor.save_space() directly after fit().

This has NO impact on inference accuracy.
It is recommended if the only goal is to use the trained model for prediction.
Certain advanced functionality may no longer be available if save_space=True. Refer to predictor.save_space() documentation for more details.

feature_generatorautogluon.features.generators.AbstractFeatureGenerator, default = autogluon.features.generators.AutoMLPipelineFeatureGenerator
The feature generator used by AutoGluon to process the input data to the form sent to the models. This often includes automated feature generation and data cleaning.
It is generally recommended to keep the default feature generator unless handling an advanced use-case.
To control aspects of the default feature generation process, you can pass in an AutoMLPipelineFeatureGenerator object constructed using some of these kwargs:

enable_numeric_featuresbool, default True
Whether to keep features of ‘int’ and ‘float’ raw types.
These features are passed without alteration to the models.
Appends IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[‘int’, ‘float’]))) to the generator group.

enable_categorical_featuresbool, default True
Whether to keep features of ‘object’ and ‘category’ raw types.
These features are processed into memory optimized ‘category’ features.
Appends CategoryFeatureGenerator() to the generator group.

enable_datetime_featuresbool, default True
Whether to keep features of ‘datetime’ raw type and ‘object’ features identified as ‘datetime_as_object’ features.
These features will be converted to ‘int’ features representing milliseconds since epoch.
Appends DatetimeFeatureGenerator() to the generator group.

enable_text_special_featuresbool, default True
Whether to use ‘object’ features identified as ‘text’ features to generate ‘text_special’ features such as word count, capital letter ratio, and symbol counts.
Appends TextSpecialFeatureGenerator() to the generator group.

enable_text_ngram_featuresbool, default True
Whether to use ‘object’ features identified as ‘text’ features to generate ‘text_ngram’ features.
Appends TextNgramFeatureGenerator(vectorizer=vectorizer) to the generator group.

enable_raw_text_featuresbool, default False
Whether to keep the raw text features.
Appends IdentityFeatureGenerator(infer_features_in_args=dict(required_special_types=[‘text’])) to the generator group.

vectorizerCountVectorizer, default CountVectorizer(min_df=30, ngram_range=(1, 3), max_features=10000, dtype=np.uint8)
sklearn CountVectorizer object to use in TextNgramFeatureGenerator.
Only used if enable_text_ngram_features=True.

unlabeled_datapd.DataFrame, default = None
[Experimental Parameter] UNUSED.
Collection of data without labels that we can use to pretrain on.
This is the same schema as train_data, except without the labels.
Currently, unlabeled_data is not used by any model.

verbosityint
If specified, overrides the existing predictor.verbosity value.

raise_on_model_failure: bool, default = False
If True, will raise on any exception during model training.
This is useful when using a debugger during development to identify the cause of model failures.
This should only be used for debugging.

If False, will try to skip to the next model if an exception occurred during model training.
This is the default logic and is a core principle of AutoGluon’s design.

Added in version 1.3.0.

raise_on_no_models_fitted: bool, default = True
If True, will raise a RuntimeError if no models were successfully fit during fit().

calibrate: bool or str, default = ‘auto’
Note: It is recommended to use [‘auto’, False] as the values and avoid True.
If ‘auto’ will automatically set to True if the problem_type and eval_metric are suitable for calibration.
If True and the problem_type is classification, temperature scaling will be used to calibrate the Predictor’s estimated class probabilities
(which may improve metrics like log_loss) and will train a scalar parameter on the validation set.
If True and the problem_type is quantile regression, conformalization will be used to calibrate the Predictor’s estimated quantiles
(which may improve the prediction interval coverage, and bagging could further improve it) and will compute a set of scalar parameters on the validation set.

test_datastr or pd.DataFrame, default = None
Table of the test data.
If str is passed, test_data will be loaded using the str value as the file path.
NOTE: This test_data is NEVER SEEN by the model during training and, if specified, is only used for logging purposes (i.e. for learning curve generation).
This test_data should be treated the same way test data is used in predictor.leaderboard.

learning_curvesbool or dict, default = None
If bool and is True, default learning curve hyperparameter ag_args will be initialized for each of the models included in the ensemble.
By default, learning curves will include eval_metric scores specified in fit call arguments.
This can be overwritten as shown below.

If dict, user can pass learning_curves parameters to be initialized as ag_args in the following format:
learning_curves = {
“metrics”: str or list(str) or Scorer or list(Scorer):
autogluon metric scorer(s) to be calculated at each iteration, represented as Scorer object(s) or scorer name(s) (str)

”use_error”: bool : whether to use error or score format for metrics listed above

}

Return type:

TabularPredictor object. Returns self.

Examples

>>> from autogluon.tabular import TabularDataset, TabularPredictor
>>> train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
>>> label = 'class'
>>> predictor = TabularPredictor(label=label).fit(train_data)
>>> test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
>>> leaderboard = predictor.leaderboard(test_data)
>>> y_test = test_data[label]
>>> test_data = test_data.drop(columns=[label])
>>> y_pred = predictor.predict(test_data)
>>> perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred)

To maximize predictive performance, use the following:

>>> eval_metric = 'roc_auc'  # set this to the metric you ultimately care about
>>> time_limit = 3600  # set as long as you are willing to wait (in sec)
>>> predictor = TabularPredictor(label=label, eval_metric=eval_metric).fit(train_data, presets=['best_quality'], time_limit=time_limit)

        

      

    

      

          
            On this page

        

            

TabularPredictor.fit

TabularPredictor.fit()

          

      

      

    

    

    

    

    
    
