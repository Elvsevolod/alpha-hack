"""Known executable contracts shared by catalog, controller and worker."""
from __future__ import annotations

from .util import digest

COMPATIBILITY_VERSION = 1
FRAME = {"datetime", "math", "log1p"}
NUMERIC = {"standard_scale", "robust_scale", "yeo_johnson", "spline"}
SELECT = {"drop_constant", "drop_correlated", "select_kbest"}


def native_branch(spec, task):
    return spec.model in {"catboost", "lightgbm", "lgbm_ranker"} and spec.representation in {"native", "tabular"} and not task.text_columns


def incompatibilities(spec, task, strategy=None, capabilities=None):
    reasons = []
    for operation in spec.transformations:
        if operation.name in {"datetime", "log1p"} and not operation.parameters.get("columns"):
            reasons.append("frame_transformation_requires_columns: " + operation.name)
    if spec.engine == "direct" and (spec.search_space or spec.learners):
        reasons.append("direct_pipeline_does_not_execute_search_space_or_learners")
    if spec.engine in {"direct", "optuna"} or spec.engine == "custom" and spec.custom_component == "transformer":
        stages = [0 if op.name in FRAME else 1 if op.name in NUMERIC else 2 for op in spec.transformations]
        if stages != sorted(stages):
            reasons.append("transformation_stage_order: frame operations must precede numeric transforms and selection")
        unsupported = [op.name for op in spec.transformations if op.name in NUMERIC | SELECT and native_branch(spec, task)]
        if unsupported:
            reasons.append("native_transformations_unsupported: " + ", ".join(unsupported))
        if spec.imputation.numeric == "native" and any(op.name in NUMERIC for op in spec.transformations):
            reasons.append("numeric_transformations_require_imputation")
        if spec.imputation.numeric == "native" and spec.model not in {"catboost", "lightgbm", "lgbm_ranker", "xgboost", "hist_gradient_boosting"}:
            reasons.append("native_missing_values_unsupported_by_model")
        if spec.model in {"ridge", "elastic_net", "spline"} and task.classes:
            reasons.append("numeric_model_requires_regression")
        if spec.model == "logistic" and not task.classes:
            reasons.append("logistic_requires_classification")
        if spec.model == "lgbm_ranker" and task.kind != "ranking":
            reasons.append("ranker_requires_ranking")
        if spec.resampling != "none" and task.kind != "classification":
            reasons.append("resampling_requires_single_classification")
        if spec.resampling == "smote" and native_branch(spec, task):
            reasons.append("smote_requires_encoded_numeric_representation")
        if spec.prediction_policy != "default":
            if task.kind != "classification" or len(task.classes.get(task.targets[0], [])) != 2:
                reasons.append("postprocessing_requires_binary_classification")
            if spec.prediction_policy == "threshold" and (task.metric.name not in {"f1", "accuracy"} or task.output.prediction_type != "labels"):
                reasons.append("threshold_requires_label_metric_and_output")
        if spec.target_transform != "none" and task.classes:
            reasons.append("classification_target_transform_unsupported")
    if spec.engine in {"flaml", "autogluon"}:
        if spec.hyperparameters or spec.search_space or spec.training_objective is not None:
            reasons.append("automl_hyperparameters_unsupported: this adapter uses the engine's built-in search and confirmed task metric; select learners or use direct/Optuna for explicit parameters and search spaces")
        if task.kind in {"forecasting", "ranking"} or task.text_columns:
            reasons.append("automl_requires_tabular_nontext_task")
        if spec.representation not in {"tabular", "native"}:
            reasons.append("automl_representation_unsupported")
        if spec.transformations or spec.imputation.numeric not in {"median", "native"} or spec.imputation.categorical != "missing_category":
            reasons.append("automl_preprocessing_unsupported: transformations must be empty; numeric imputation median/native; categorical missing_category; use direct or Optuna for model-based imputation")
        if spec.prediction_policy != "default" or spec.target_transform != "none" or spec.resampling != "none":
            reasons.append("automl_postprocessing_unsupported")
        allowed = {"lgbm", "rf", "extra_tree", "xgboost", "xgb_limitdepth", "catboost"} if spec.engine == "flaml" else {"GBM", "CAT", "XGB"}
        if not set(spec.learners) <= allowed:
            reasons.append("automl_learner_unverified")
        if spec.engine == "autogluon" and task.metric.name in {"roc_auc", "average_precision"} and any(len(labels) != 2 for labels in task.classes.values()):
            reasons.append("autogluon_probability_metric_requires_binary_targets")
    needs_inner = spec.engine in {"flaml", "autogluon", "optuna"} or spec.prediction_policy != "default"
    if needs_inner and (strategy == "temporal" and not task.time_column or strategy == "group" and not task.group_column):
        reasons.append("inner_split_missing_structural_role")
    if capabilities:
        try:
            capabilities.require_available(spec)
        except (ValueError, ImportError) as error:
            reasons.append(str(error))
        packages = {"scikit-learn", "numpy", "pandas", "scipy", "cloudpickle"}
        if any(op.name in {"math", "drop_correlated"} for op in spec.transformations):
            packages.add("feature-engine")
        if spec.resampling != "none":
            packages.add("imbalanced-learn")
        for package in packages:
            expected = capabilities.expected.get(package)
            if expected and capabilities.versions.get(package) != expected:
                reasons.append("incompatible_dependency_version: " + package)
    return reasons


def require_compatible(spec, task, strategy=None, capabilities=None):
    reasons = incompatibilities(spec, task, strategy, capabilities)
    if reasons:
        raise ValueError("; ".join(reasons))


def registry_fingerprint(capabilities):
    return digest({"version": COMPATIBILITY_VERSION, "environment": capabilities.versions if capabilities else {},
        "expected": capabilities.expected if capabilities else {}, "records": capabilities.records if capabilities else {}})
