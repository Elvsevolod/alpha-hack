# autogluon.tabular.TabularPredictor.refit_full
Version: 1.3.1

## autogluon.tabular.TabularPredictor.refit_full(self, model: 'str | list[str]' = 'all', set_best_to_refit_full: 'bool' = True, train_data_extra: 'pd.DataFrame' = None, num_cpus: 'int | str' = 'auto', num_gpus: 'int | str' = 'auto', fit_strategy: "Literal['auto', 'sequential', 'parallel']" = 'auto', **kwargs) -> 'dict[str, str]'

Retrain model on all of the data (training + validation).
For bagged models:
    Optimizes a model's inference time by collapsing bagged ensembles into a single model fit on all of the training data.
    This process will typically result in a slight accuracy reduction and a large inference speedup.
    The inference speedup will generally be between 10-200x faster than the original bagged ensemble model.
        The inference speedup factor is equivalent to (k * n), where k is the number of folds (`num_bag_folds`) and n is the number of finished repeats (`num_bag_sets`) in the bagged ensemble.
    The runtime is generally 10% or less of the original fit runtime.
        The runtime can be roughly estimated as 1 / (k * n) of the original fit runtime, with k and n defined above.
For non-bagged models:
    Optimizes a model's accuracy by retraining on 100% of the data without using a validation set.
    Will typically result in a slight accuracy increase and no change to inference time.
    The runtime will be approximately equal to the original fit runtime.
This process does not alter the original models, but instead adds additional models.
If stacker models are refit by this process, they will use the refit_full versions of the ancestor models during inference.
Models produced by this process will not have validation scores, as they use all of the data for training.
    Therefore, it is up to the user to determine if the models are of sufficient quality by including test data in `predictor.leaderboard(test_data)`.
    If the user does not have additional test data, they should reference the original model's score for an estimate of the performance of the refit_full model.
        Warning: Be aware that utilizing refit_full models without separately verifying on test data means that the model is untested, and has no guarantee of being consistent with the original model.
`cache_data` must have been set to `True` during the original training to enable this functionality.

Parameters
----------
model : str | list[str], default = 'all'
    Model name of model(s) to refit.
        If 'all' then all models are refitted.
        If 'best' then the model with the highest validation score is refit.
    All ancestor models will also be refit in the case that the selected model is a weighted or stacker ensemble.
    Valid models are listed in this `predictor` by calling `predictor.model_names()`.
set_best_to_refit_full : bool | str, default = True
    If True, sets best model to the refit_full version of the prior best model.
    This means the model used when `predictor.predict(data)` is called will be the refit_full version instead of the original version of the model.
    Ignored if `model` is not the best model.
    If str, interprets as a model name and sets best model to the refit_full version of the model `set_best_to_refit_full`.
train_data_extra : pd.DataFrame, default = None
    If specified, will be used as additional rows of training data when refitting models.
    Requires label column. Will only be used for L1 models.
num_cpus: int | str, default = "auto"
    The total amount of cpus you want AutoGluon predictor to use.
    Auto means AutoGluon will make the decision based on the total number of cpus available and the model requirement for best performance.
    Users generally don't need to set this value
num_gpus: int | str, default = "auto"
    The total amount of gpus you want AutoGluon predictor to use.
    Auto means AutoGluon will make the decision based on the total number of gpus available and the model requirement for best performance.
    Users generally don't need to set this value
fit_strategy: Literal["auto", "sequential", "parallel"], default = "auto"
    The strategy used to fit models.
    If "auto", uses the same fit_strategy as used in the original :meth:`TabularPredictor.fit` call.
    If "sequential", models will be fit sequentially. This is the most stable option with the most readable logging.
    If "parallel", models will be fit in parallel with ray, splitting available compute between them.
        Note: "parallel" is experimental and may run into issues. It was first added in version 1.2.0.
    For machines with 16 or more CPU cores, it is likely that "parallel" will be faster than "sequential".

    .. versionadded:: 1.2.0

**kwargs
    [Advanced] Developer debugging arguments.

Returns
-------
Dictionary of original model names -> refit_full model names.
