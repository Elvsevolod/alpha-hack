"""Known executable contracts shared by catalog, controller and worker."""
from __future__ import annotations

import numpy as np

from .util import digest

COMPATIBILITY_VERSION = 3
FRAME = {"datetime", "math", "log1p", "aggregate", "history", "target_encode"}
NUMERIC = {"standard_scale", "robust_scale", "yeo_johnson", "spline"}
SELECT = {"drop_constant", "drop_correlated", "select_kbest"}


def native_branch(spec, task):
    return spec.model in {"catboost", "lightgbm", "lgbm_ranker"} and spec.representation in {"native", "tabular"} and not task.text_columns


def ranking_label_issue(values, label_gain=None):
    """Return why labels cannot be used by LightGBM's integer relevance objective."""
    try:
        labels = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return "relevance labels must be numeric"
    if labels.ndim != 1 or labels.size == 0:
        return "relevance labels must be a nonempty one-dimensional target"
    if not np.isfinite(labels).all():
        return "relevance labels must be finite"
    if (labels < 0).any():
        return "relevance labels must be nonnegative"
    if not np.equal(labels, np.floor(labels)).all():
        return "relevance labels must be integer-valued"
    if label_gain is None:
        gain_count = 31  # LightGBM's default label_gain contains indices 0 through 30.
    else:
        try:
            gain_count = len(label_gain.split(",")) if isinstance(label_gain, str) else len(label_gain)
        except TypeError:
            return "label_gain must be a sequence of gains"
    if gain_count < 1 or int(labels.max()) >= gain_count:
        return "relevance labels exceed the configured label_gain indices"
    return None


def ranking_bootstrap_issue(task, train, plan):
    """Limit the automatic ranker candidate to the NDCG label contract screened offline."""
    if task.kind != "ranking":
        return "task is not ranking"
    if task.metric.name != "ndcg":
        return "automatic ranker candidate is screened only for NDCG"
    if len(task.targets) != 1:
        return "automatic ranker candidate is screened only for one relevance target"
    if task.text_columns:
        return "native LightGBM ranker does not support text features"
    if task.group_column in task.feature_columns:
        return "query group identifiers must not be model features"
    if plan.strategy not in {"group", "temporal_group"}:
        return "automatic ranker candidate requires group-preserving validation"
    if not task.group_column or task.group_column not in train:
        return "automatic ranker candidate requires an established query group column"
    groups = train[task.group_column]
    if groups.isna().any():
        return "query groups must be complete across validation partitions"
    partition_groups = [set(groups.iloc[rows]) for rows in (plan.fit, plan.validation, plan.control)]
    if any(partition_groups[left] & partition_groups[right]
            for left, right in ((0, 1), (0, 2), (1, 2))):
        return "query groups must remain disjoint across fit, validation and control"
    for target in task.targets:
        issue = ranking_label_issue(train[target].to_numpy())
        if issue:
            return f"{target}: {issue}"
    return None


def incompatibilities(spec, task, strategy=None, capabilities=None):
    reasons = []
    from .preprocessing import selected_features
    try:
        selected_features(spec, task)
    except ValueError as error:
        reasons.append(str(error))
    if spec.model == "knn" and (task.kind not in {"classification", "regression", "multioutput_classification", "multioutput_regression"} or spec.shift_strategy != "none"):
        reasons.append("knn_requires_tabular_numeric_or_classification_output_without_importance_weights")
    if spec.log_inverse != "naive" and (spec.target_transform != "log1p" or task.classes or spec.engine != "direct"):
        reasons.append("oof_smearing_requires_direct_log1p_regression")
    if spec.shift_strategy == "importance_weighting" and (spec.postprocess_fit_policy == "pooled_oof" or spec.log_inverse == "oof_smearing"):
        reasons.append("domain_weights_for_nested_oof_require_an_independent_fold_refit_adapter")
    for column, constraints in spec.imputation.constraints.items():
        confirmed = task.feature_constraints.get(column, {})
        if any(key not in confirmed or value != confirmed[key] for key, value in constraints.items()):
            reasons.append("imputation_constraints_require_confirmed_feature_semantics")
    if spec.engine == "direct":
        from .pipelines import validate_parameters
        try:
            validate_parameters(spec, task)
        except ValueError as error:
            reasons.append(str(error))
    if task.kind == "forecasting" and spec.early_stopping_rounds:
        reasons.append("forecast_early_stopping_requires_causal_inner_adapter; use fixed iterations until verified")
    if spec.shift_strategy == "pseudo_label" and task.kind != "classification":
        reasons.append("pseudo_label_requires_single_output_classification")
    if spec.model == "lightgbm" and spec.engine == "direct":
        from .pipelines import effective_parameters
        parameters = effective_parameters(spec)
        fraction = parameters.get("bagging_fraction", parameters.get("subsample", 1))
        frequency = parameters.get("bagging_freq", parameters.get("subsample_freq", 0))
        if fraction < 1 and frequency <= 0:
            reasons.append("inactive_row_bagging: subsample/bagging_fraction < 1 requires positive subsample_freq/bagging_freq")
    if spec.model == "mlp":
        reasons.append("model_outside_classical_ml_scope")
    if spec.engine == "optuna" and task.kind == "forecasting":
        reasons.append("optuna_forecasting_requires_causal_forecast_adapter; use direct until verified")
    if spec.engine == "optuna":
        from .search import normalized_search_space
        try:
            normalized_search_space(spec)
        except ValueError as error:
            reasons.append(str(error))
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
            if task.kind != "classification" or (spec.prediction_policy != "class_offsets" and len(task.classes.get(task.targets[0], [])) != 2):
                reasons.append("postprocessing_requires_binary_classification")
            if spec.prediction_policy == "class_offsets" and (task.metric.name not in {"f1", "accuracy", "balanced_accuracy"}
                    or task.output.prediction_type != "labels" or len(task.classes.get(task.targets[0], [])) < 3):
                reasons.append("class_offsets_require_multiclass_label_metric_and_output")
            if spec.prediction_policy == "threshold" and (task.metric.name not in {"f1", "accuracy"} or task.output.prediction_type != "labels"):
                reasons.append("threshold_requires_label_metric_and_output")
        if spec.target_transform != "none" and task.classes:
            reasons.append("classification_target_transform_unsupported")
    if spec.shift_strategy != "none" and (spec.engine != "direct" or spec.model in {"mlp", "lgbm_ranker"} or spec.resampling != "none"):
        reasons.append("shift_interventions_require_weight_compatible_direct_tabular_models_without_resampling")
    if spec.early_stopping_rounds and (spec.model not in {"lightgbm", "xgboost", "catboost", "hist_gradient_boosting"} or spec.engine not in {"direct", "optuna"}):
        reasons.append("early_stopping_requires_direct_or_optuna_boosting")
    if spec.engine in {"flaml", "autogluon", "lightautoml"}:
        if spec.engine == "flaml" and spec.n_trials is not None and spec.n_trials < 2:
            reasons.append("flaml_replayable_search_requires_n_trials_at_least_two")
        if spec.engine == "autogluon" and spec.autogluon_configurations:
            from .search import autogluon_portfolio
            try:
                autogluon_portfolio(spec, task, spec.learners or ["GBM", "CAT", "XGB"])
            except ValueError as error:
                reasons.append(str(error))
        if spec.include_features is not None or spec.exclude_features:
            reasons.append("automl_subset_unverified; use direct/Optuna")
        if spec.autogluon_configurations and spec.engine != "autogluon":
            reasons.append("explicit_configurations_require_autogluon")
        if spec.engine == "autogluon" and spec.autogluon_profile != "selection" and strategy in {"temporal", "temporal_group"}:
            reasons.append("autogluon_random_bagging_forbidden_on_time; use selection or split-aware OOF blend")
        if strategy == "temporal_group":
            reasons.append("automl_joint_temporal_group_adapter_unverified; use direct or Optuna")
        if task.metric.name in {"mape", "balanced_accuracy"}:
            reasons.append("metric_adapter_unverified: use direct or Optuna for mape/balanced_accuracy")
        if spec.hyperparameters or spec.search_space or spec.training_objective is not None:
            reasons.append("automl_hyperparameters_unsupported: this adapter uses the engine's built-in search and confirmed task metric; select learners or use direct/Optuna for explicit parameters and search spaces")
        if task.kind in {"forecasting", "ranking"} or task.text_columns:
            reasons.append("automl_requires_tabular_nontext_task")
        if spec.representation not in {"tabular", "native"}:
            reasons.append("automl_representation_unsupported")
        if spec.transformations or spec.imputation.numeric not in {"median", "native"} or spec.imputation.categorical != "missing_category":
            reasons.append("automl_preprocessing_unsupported: transformations must be empty; numeric imputation median/native; categorical missing_category; use direct or Optuna for model-based imputation")
        supported_decision = spec.engine == "autogluon" and spec.prediction_policy == "class_offsets" and spec.postprocess_fit_policy == "pooled_oof" and task.kind == "classification" and task.output.prediction_type == "labels"
        if spec.prediction_policy != "default" and not supported_decision or spec.target_transform != "none" or spec.resampling != "none":
            reasons.append("automl_postprocessing_unsupported")
        allowed = {"lgbm", "rf", "extra_tree", "xgboost", "xgb_limitdepth", "catboost"} if spec.engine == "flaml" else {"lgb"} if spec.engine == "lightautoml" else {"GBM", "CAT", "XGB"}
        if not set(spec.learners) <= allowed:
            reasons.append("automl_learner_unverified")
        if spec.engine == "autogluon" and task.metric.name in {"roc_auc", "average_precision"} and any(len(labels) != 2 for labels in task.classes.values()):
            reasons.append("autogluon_probability_metric_requires_binary_targets")
    needs_inner = spec.engine in {"flaml", "autogluon", "optuna"} or spec.prediction_policy != "default"
    if spec.fit_diagnostics != "none" and (spec.engine not in {"direct", "optuna"} or task.kind != "classification"):
        reasons.append("fit_diagnostics_require_direct_or_optuna_single_classification")
    if spec.prediction_policy == "class_offsets" and spec.engine not in {"direct", "optuna", "autogluon"}:
        reasons.append("class_offsets_require_direct_or_optuna_frozen_fit_path")
    if spec.engine == "custom" and spec.prediction_policy != "default":
        reasons.append("custom_postprocessing_has_no_verified_inner_holdout; use direct or Optuna")
    if (needs_inner or spec.fit_diagnostics != "none") and strategy == "temporal_group" and (not task.time_column or not task.group_column):
        reasons.append("joint_split_requires_time_and_group_roles")
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
