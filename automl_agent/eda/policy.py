"""Fit policy shared by autonomous dispatch and its worker entry."""
RESOURCE_PARAMETERS = {"n_jobs", "num_threads", "num_thread", "nthread", "nthreads", "thread_count",
    "device", "device_type", "task_type", "gpu_device_id", "num_cpus", "num_gpus"}

DIRECT_MODELS = frozenset({"lightgbm", "ridge", "extra_trees", "random_forest", "hist_gradient_boosting", "logistic"})


def parameter_capabilities():
    from ..pipelines import PARAMETERS
    return {model:sorted(PARAMETERS[model] - RESOURCE_PARAMETERS) for model in sorted(DIRECT_MODELS)}


def validate_pipeline(pipeline):
    if (pipeline.engine != "direct" or pipeline.custom_module or pipeline.members or
            pipeline.search_owner != "none" or pipeline.fit_diagnostics != "none" or
            pipeline.postprocess_fit_policy != "none" or pipeline.prediction_policy != "default" or
            pipeline.imputation.numeric not in {"median", "native", "knn"} or pipeline.imputation.categorical == "model" or
            pipeline.model not in DIRECT_MODELS or
            RESOURCE_PARAMETERS & set(pipeline.hyperparameters)):
        raise ValueError("EDA permits one direct CPU fit without nested training or resource overrides")

    from ..pipelines import PARAMETERS, effective_parameters
    unknown = set(effective_parameters(pipeline)) - PARAMETERS[pipeline.model]
    if unknown:
        raise ValueError(f"Unsupported parameters for {pipeline.model}: {sorted(unknown)}. Allowed: {sorted(PARAMETERS[pipeline.model] - RESOURCE_PARAMETERS)}")


def constrain_advertised_pipeline(definitions):
    """Describe the enforced EDA subset without changing production contracts."""
    properties = definitions.get('PipelineSpec', {}).get('properties', {})
    for name, value in {'engine':'direct', 'fit_diagnostics':'none', 'search_owner':'none',
            'postprocess_fit_policy':'none', 'prediction_policy':'default'}.items():
        if name in properties:
            properties[name] = {'type':'string', 'const':value, 'default':value}
    if 'model' in properties:
        properties['model'] = {**properties['model'], 'enum':sorted(DIRECT_MODELS)}
