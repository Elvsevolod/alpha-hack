# refit
Source: https://auto.gluon.ai/1.3.1/api/autogluon.tabular.TabularPredictor.refit_full.html



    

    

    

        TabularPredictor.refit_full - AutoGluon 1.3.1 documentation

    

    

    

  

    

  
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

          

        

TabularPredictor.refit_full¶

TabularPredictor.refit_full(model: str | list[str] = 'all', set_best_to_refit_full: bool = True, train_data_extra: DataFrame = None, num_cpus: int | str = 'auto', num_gpus: int | str = 'auto', fit_strategy: Literal['auto', 'sequential', 'parallel'] = 'auto', **kwargs) → dict[str, str][source]¶

Retrain model on all of the data (training + validation).
For bagged models:

Optimizes a model’s inference time by collapsing bagged ensembles into a single model fit on all of the training data.
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

cache_data must have been set to True during the original training to enable this functionality.

Parameters:

model (str | list[str], default = 'all') – 
Model name of model(s) to refit.
If ‘all’ then all models are refitted.
If ‘best’ then the model with the highest validation score is refit.

All ancestor models will also be refit in the case that the selected model is a weighted or stacker ensemble.
Valid models are listed in this predictor by calling predictor.model_names().

set_best_to_refit_full (bool | str, default = True) – If True, sets best model to the refit_full version of the prior best model.
This means the model used when predictor.predict(data) is called will be the refit_full version instead of the original version of the model.
Ignored if model is not the best model.
If str, interprets as a model name and sets best model to the refit_full version of the model set_best_to_refit_full.

train_data_extra (pd.DataFrame, default = None) – If specified, will be used as additional rows of training data when refitting models.
Requires label column. Will only be used for L1 models.

num_cpus (int | str, default = "auto") – The total amount of cpus you want AutoGluon predictor to use.
Auto means AutoGluon will make the decision based on the total number of cpus available and the model requirement for best performance.
Users generally don’t need to set this value

num_gpus (int | str, default = "auto") – The total amount of gpus you want AutoGluon predictor to use.
Auto means AutoGluon will make the decision based on the total number of gpus available and the model requirement for best performance.
Users generally don’t need to set this value

fit_strategy (Literal["auto", "sequential", "parallel"], default = "auto") – 
The strategy used to fit models.
If “auto”, uses the same fit_strategy as used in the original TabularPredictor.fit() call.
If “sequential”, models will be fit sequentially. This is the most stable option with the most readable logging.
If “parallel”, models will be fit in parallel with ray, splitting available compute between them.

Note: “parallel” is experimental and may run into issues. It was first added in version 1.2.0.

For machines with 16 or more CPU cores, it is likely that “parallel” will be faster than “sequential”.

Added in version 1.2.0.

**kwargs – [Advanced] Developer debugging arguments.

Return type:

Dictionary of original model names -> refit_full model names.

        

      

    

      

          
            On this page

        

            

TabularPredictor.refit_full

TabularPredictor.refit_full()

          

      

      

    

    

    

    

    
    
