from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.preprocessing import LabelEncoder

from .api import prediction_bundle
from .contracts import PipelineSpec, TaskSpec, TrainingSummary
from .evaluation import score_predictions
from .preprocessing import FeatureMatrix
from .validation import inner_split
from .compatibility import ranking_label_issue, require_compatible
from .util import digest

PARAMETERS = {
    "knn": {"n_neighbors", "weights", "metric", "p", "leaf_size"},
    "constant": set(),
    "ridge": {"alpha", "fit_intercept", "solver", "tol", "max_iter", "positive"},
    "spline": {"alpha", "fit_intercept", "solver", "tol", "max_iter", "positive"},
    "elastic_net": {"alpha", "l1_ratio", "fit_intercept", "max_iter", "tol", "positive", "selection"},
    "logistic": {"C", "penalty", "solver", "l1_ratio", "max_iter", "tol", "class_weight", "fit_intercept", "intercept_scaling", "n_jobs"},
    "sgd": {"loss", "penalty", "alpha", "max_iter", "tol", "l1_ratio", "epsilon", "average", "learning_rate", "eta0", "power_t", "class_weight", "fit_intercept"},
    "random_forest": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "bootstrap", "max_samples", "criterion", "ccp_alpha", "max_leaf_nodes", "class_weight", "n_jobs"},
    "extra_trees": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "bootstrap", "max_samples", "criterion", "ccp_alpha", "max_leaf_nodes", "class_weight", "n_jobs"},
    "hist_gradient_boosting": {"max_iter", "learning_rate", "max_leaf_nodes", "max_depth", "min_samples_leaf", "l2_regularization", "max_bins", "loss", "early_stopping", "max_features"},
    "catboost": {"iterations", "depth", "learning_rate", "l2_leaf_reg", "loss_function", "eval_metric", "border_count", "bagging_temperature", "random_strength", "bootstrap_type", "subsample", "rsm", "grow_policy", "min_data_in_leaf", "max_leaves", "class_weights", "auto_class_weights", "one_hot_max_size", "nan_mode", "leaf_estimation_iterations", "boost_from_average"},
    "lightgbm": {"n_estimators", "num_leaves", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "min_child_samples", "min_child_weight", "subsample_freq", "max_bin", "min_split_gain", "class_weight", "objective", "importance_type", "deterministic", "force_col_wise", "force_row_wise", "extra_trees", "linear_tree"},
    "lgbm_ranker": {"n_estimators", "num_leaves", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "min_child_samples", "min_child_weight", "subsample_freq", "max_bin", "min_split_gain", "objective", "label_gain", "lambdarank_truncation_level"},
    "xgboost": {"n_estimators", "max_depth", "max_leaves", "learning_rate", "subsample", "colsample_bytree", "colsample_bylevel", "reg_alpha", "reg_lambda", "min_child_weight", "gamma", "max_bin", "grow_policy", "objective", "eval_metric", "scale_pos_weight", "monotone_constraints", "interaction_constraints", "booster"},
}


CONTROLLED_PARAMETERS = {
    "logistic": {"n_jobs", "random_state"},
    "random_forest": {"n_jobs", "random_state"},
    "extra_trees": {"n_jobs", "random_state"},
    "catboost": {"thread_count", "task_type", "devices", "train_dir", "allow_writing_files", "random_seed", "verbose"},
    "lightgbm": {"n_jobs", "num_threads", "device", "device_type", "random_state", "verbosity", "verbose"},
    "lgbm_ranker": {"n_jobs", "num_threads", "device", "device_type", "random_state", "verbosity", "verbose"},
    "xgboost": {"n_jobs", "nthread", "device", "random_state", "tree_method"},
}


def effective_parameters(spec: PipelineSpec):
    parameters = {key: value for key, value in spec.hyperparameters.items()
        if key not in CONTROLLED_PARAMETERS.get(spec.model, set())}
    if spec.model == "xgboost":
        for alias, canonical in (("lambda", "reg_lambda"), ("alpha", "reg_alpha"),
                ("eta", "learning_rate"), ("min_split_loss", "gamma")):
            if alias in parameters:
                if canonical in parameters and parameters[canonical] != parameters[alias]:
                    raise ValueError(f"Conflicting XGBoost aliases: {alias}/{canonical}")
                parameters[canonical] = parameters.pop(alias)
    if spec.model in {"lightgbm", "lgbm_ranker"}:
        # LightGBM's primary name wins over its sklearn alias, including when
        # an Optuna trial overrides a normalized base value.
        for primary, alias in (("feature_fraction", "colsample_bytree"),
                ("bagging_fraction", "subsample"), ("bagging_freq", "subsample_freq"),
                ("lambda_l2", "reg_lambda"), ("lambda_l1", "reg_alpha"),
                ("min_data_in_leaf", "min_child_samples"), ("num_iterations", "n_estimators")):
            if primary in parameters:
                if alias in parameters and parameters[alias] != parameters[primary]:
                    raise ValueError(f"Conflicting LightGBM aliases: {primary}/{alias}")
                parameters[alias] = parameters.pop(primary)
    return parameters


PARAMETERS["lightgbm"].update({"cat_smooth", "cat_l2", "min_data_per_group", "alpha", "fair_c", "tweedie_variance_power"})


def validate_parameters(spec, task):
    if set(spec.hyperparameters) & CONTROLLED_PARAMETERS.get(spec.model, set()):
        raise ValueError("Model resource/seed parameters are controller-owned; use the declared task limits and seed")
    p = effective_parameters(spec)
    if spec.model in PARAMETERS and not set(p) <= PARAMETERS[spec.model]:
        raise ValueError(f"Unsupported parameters for {spec.model}: {sorted(set(p) - PARAMETERS[spec.model])}")
    if spec.model == "knn":
        if p.get("metric", "minkowski") not in {"minkowski", "euclidean", "manhattan", "cosine"} or p.get("weights", "uniform") not in {"uniform", "distance"}:
            raise ValueError("KNN requires a verified metric and weighting rule")
        for key in ("n_neighbors", "leaf_size", "p"):
            value = p.get(key, 5 if key == "n_neighbors" else 30 if key == "leaf_size" else 2)
            if type(value) is not int or value < 1 or value > (1024 if key == "n_neighbors" else 100):
                raise ValueError("Invalid bounded KNN parameter: " + key)
    if spec.model == "lightgbm":
        objective = spec.training_objective or p.get("objective") or ("multiclass" if task.classes else "regression")
        classification = {"binary", "multiclass", "multiclassova", "cross_entropy", "cross_entropy_lambda"}
        regression = {"regression", "regression_l2", "l2", "mse", "rmse", "regression_l1", "l1", "mae", "huber", "fair", "poisson", "quantile", "mape", "gamma", "tweedie"}
        if objective not in (classification if task.classes else regression):
            raise ValueError("LightGBM objective is incompatible with the confirmed target family")
        if spec.training_objective and "objective" in p and p["objective"] != spec.training_objective:
            raise ValueError("Conflicting training objectives")
        for key in ("cat_smooth", "cat_l2", "min_data_per_group", "alpha", "fair_c", "tweedie_variance_power"):
            if key not in p:
                continue
            value = p[key]
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError("Expected a finite numeric LightGBM parameter: " + key)
            valid = value >= 0 if key in {"cat_smooth", "cat_l2"} else value > 0
            if key == "min_data_per_group":
                valid = type(value) is int and value > 0
            if key == "alpha":
                valid = 0 < value <= 1 and objective in {"huber", "quantile"}
            if key == "fair_c":
                valid = value > 0 and objective == "fair"
            if key == "tweedie_variance_power":
                valid = 1 <= value < 2 and objective == "tweedie"
            if key in {"cat_smooth", "cat_l2", "min_data_per_group"}:
                valid = valid and bool(task.categorical_columns)
            if not valid:
                raise ValueError(f"LightGBM {key} is outside its range or incompatible with objective/features")
    return p


class BoundedNeighbors:
    """Bound every prediction batch; do not allocate test-by-fit distances."""
    def __init__(self, parameters, classification, context):
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        self.model = (KNeighborsClassifier if classification else KNeighborsRegressor)(
            algorithm="brute", n_jobs=context["cpu"], **parameters)
        self.memory_mb = max(1., min(128., context["memory_mb"] * .05))

    def fit(self, X, y):
        if X.shape[0] > 100000 or self.model.n_neighbors > X.shape[0]:
            raise ValueError("KNN fit population exceeds its bound or has fewer rows than neighbors")
        from scipy import sparse
        size = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes if sparse.issparse(X) else X.nbytes
        if size > self.memory_mb * 1024 ** 2 * 8:
            raise MemoryError("KNN donor matrix exceeds the configured memory bound")
        self.model.fit(X, y)
        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        self.rows = X.shape[0]
        return self

    def _predict(self, X, method):
        from sklearn import config_context
        if not X.shape[0]:
            return np.empty((0, len(self.classes_))) if method == "predict_proba" else np.empty(0)
        batch = max(1, min(512, int(self.memory_mb * 1024 ** 2 / max(8, self.rows * 8))))
        with config_context(working_memory=self.memory_mb):
            return np.concatenate([getattr(self.model, method)(X[i:i + batch]) for i in range(0, X.shape[0], batch)])

    def predict(self, X):
        return self._predict(X, "predict")

    def predict_proba(self, X):
        return self._predict(X, "predict_proba")


def refit_iterations(spec: PipelineSpec, iterations, rows: int, mode: str | None):
    """Scale a frozen boosting budget to a larger refit population (sqrt, bounded)."""
    if not iterations or not spec.iteration_scale_rows or mode != "refit" or spec.refit_iteration_policy != "sqrt_rows":
        return iterations
    ratio = max(1.0, rows / max(1, spec.iteration_scale_rows))
    return min(10000, max(int(iterations), round(int(iterations) * min(2.0, ratio ** 0.5))))


def make_estimator(spec: PipelineSpec, classification: bool, context):
    if spec.model == "mlp":
        raise ValueError("model_outside_classical_ml_scope")
    p = effective_parameters(spec)
    if spec.model in PARAMETERS and not set(p) <= PARAMETERS[spec.model]:
        raise ValueError(f"Unsupported parameters for {spec.model}: {sorted(set(p) - PARAMETERS[spec.model])}")
    objective = spec.training_objective
    if objective and spec.model not in {"catboost", "lightgbm", "lgbm_ranker", "xgboost"}:
        if spec.model in {"sgd", "hist_gradient_boosting"}:
            if "loss" in p and p["loss"] != objective:
                raise ValueError("Conflicting training objectives")
            p["loss"] = objective
        else:
            expected = "log_loss" if classification else "squared_error"
            if objective != expected:
                raise ValueError("This model has a fixed training objective; choose a compatible family")
    cpu, seed = context["cpu"], spec.seed
    if spec.model == "knn":
        return BoundedNeighbors(p, classification, context)
    if spec.model == "constant":
        return DummyClassifier(strategy="prior") if classification else DummyRegressor(strategy="mean")
    if spec.model in {"ridge", "spline"}:
        if classification:
            raise ValueError("Ridge/spline recipe is for numeric targets; use logistic for classification")
        return Ridge(alpha=p.pop("alpha", 1.0), solver=p.pop("solver", "lsqr"), **p)
    if spec.model == "elastic_net":
        if classification:
            raise ValueError("ElasticNet requires numeric targets")
        return ElasticNet(random_state=seed, max_iter=p.pop("max_iter", 1000), **p)
    if spec.model == "logistic":
        if not classification:
            raise ValueError("Logistic regression requires classification")
        return LogisticRegression(random_state=seed, n_jobs=1, max_iter=p.pop("max_iter", 500), **p)
    if spec.model == "sgd":
        return SGDClassifier(random_state=seed, loss=p.pop("loss", "log_loss"), **p) if classification else SGDRegressor(random_state=seed, **p)
    if spec.model in {"extra_trees", "random_forest"}:
        classes = {"extra_trees": (ExtraTreesClassifier, ExtraTreesRegressor), "random_forest": (RandomForestClassifier, RandomForestRegressor)}
        estimator = classes[spec.model][0 if classification else 1]
        return estimator(n_estimators=p.pop("n_estimators", 100), n_jobs=cpu, random_state=seed, **p)
    if spec.model == "hist_gradient_boosting":
        # Automatic early stopping would create an uncontrolled random split.
        p["early_stopping"] = False
        return (HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor)(random_state=seed, max_iter=p.pop("max_iter", 100), **p)
    if spec.model == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        if spec.training_objective:
            p["loss_function"] = spec.training_objective
        return (CatBoostClassifier if classification else CatBoostRegressor)(iterations=p.pop("iterations", 200),
            random_seed=seed, thread_count=cpu, task_type="CPU", verbose=False, allow_writing_files=False, **p)
    if spec.model in {"lightgbm", "lgbm_ranker"}:
        from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
        if spec.training_objective:
            p["objective"] = spec.training_objective
        model = LGBMRanker if spec.model == "lgbm_ranker" else LGBMClassifier if classification else LGBMRegressor
        return model(n_estimators=p.pop("n_estimators", 150), n_jobs=cpu, random_state=seed, verbosity=-1, **p)
    if spec.model == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        if spec.training_objective:
            p["objective"] = spec.training_objective
        return (XGBClassifier if classification else XGBRegressor)(n_estimators=p.pop("n_estimators", 150),
            n_jobs=cpu, random_state=seed, tree_method="hist", device="cpu", **p)
    raise ValueError(f"Unknown model family: {spec.model}")


class DirectPipeline:
    def __init__(self, spec, task):
        self.spec, self.task = spec, task
        self.models, self.encoders, self.postprocessors = {}, {}, {}

    def fit(self, X, y, context):
        started = time.monotonic()
        require_compatible(self.spec, self.task, context.get("strategy"))
        self.training_rows = list(map(str, X.index))
        classification = bool(self.task.classes)
        make_estimator(self.spec, classification, context)
        if self.spec.target_transform == "log1p" and classification:
            raise ValueError("Classification targets cannot be log transformed")
        self.inner_diagnostics = None
        self.inner_auxiliary_fits = 0
        self.schedules = {}
        from .training_schedule import ITERATION_PARAMETER, fit_with_schedule
        final_output_schedule = self.spec.prediction_policy == "class_offsets" and self.spec.early_stopping_rounds
        if (self.spec.checkpoint_counts or final_output_schedule) and not self.spec.iteration_schedule and not self.spec.frozen_iterations:
            maximum = int(effective_parameters(self.spec).get(ITERATION_PARAMETER.get(self.spec.model), 150))
            counts = self.spec.checkpoint_counts or sorted({max(1, maximum // 4), max(1, maximum // 2), maximum})
            self._select_checkpoints(X, y, context, counts)
        elif self.spec.early_stopping_rounds and not self.spec.iteration_schedule and not self.spec.frozen_iterations and not context.get("schedule_selection"):
            a, b = inner_split(X, y, self.task, context["strategy"], self.spec.seed)
            inner_spec = self.spec.model_copy(update={"prediction_policy": "default", "postprocess_fit_policy": "none", "fit_diagnostics": "none"})
            inner = DirectPipeline(inner_spec, self.task)
            inner.fit(X.iloc[a], y.iloc[a], {**context, "schedule_selection": True,
                "skip_history_diagnostics": True, "schedule_validation": (X.iloc[b], y.iloc[b])})
            self.schedules = inner.schedules
            self.inner_auxiliary_fits += 1 + inner.summary.auxiliary_fits
        if self.spec.prediction_policy != "default" or self.spec.fit_diagnostics != "none":
            self._fit_postprocessing(X, y, context)
        self.inverse_state = dict(self.spec.learned_inverse)
        if self.spec.log_inverse == "oof_smearing" and not self.inverse_state:
            self._fit_inverse(X, y, context)
        if (any(op.name == "history" for op in self.spec.transformations)
                and self.task.kind in {"regression", "classification", "multioutput_regression"}
                and not context.get("skip_history_diagnostics") and context.get("execution_mode") != "refit"):
            self._fit_history_diagnostics(X, y, context)
        selected_started = time.monotonic()
        self.preprocessor = FeatureMatrix(self.spec, self.task, context["cpu"], context["memory_mb"], context.get("strategy", "random"))
        matrix = self.preprocessor.fit_transform(X, y)
        for target in self.task.targets:
            values = y[target].to_numpy()
            if self.spec.target_transform == "log1p":
                if (values < 0).any():
                    raise ValueError("Target log1p requires nonnegative observed targets")
                values = np.log1p(values)
            if self.spec.model == "lgbm_ranker" and self.task.kind == "ranking":
                issue = ranking_label_issue(values, self.spec.hyperparameters.get("label_gain"))
                if issue:
                    raise ValueError("LightGBM ranking labels are incompatible: " + issue)
                values = values.astype(np.int64)
            model_matrix = matrix
            if self.spec.resampling != "none":
                if self.task.kind != "classification":
                    raise ValueError("Resampling is verified for single-output classification only")
                counts = pd.Series(values).value_counts()
                if len(counts) * int(counts.max()) > len(values) * 5:
                    raise MemoryError("Resampling expansion exceeds the bounded fit matrix")
                if self.spec.resampling == "random_over":
                    from imblearn.over_sampling import RandomOverSampler
                    sampler = RandomOverSampler(random_state=self.spec.seed)
                else:
                    if self.preprocessor.native or counts.min() < 2:
                        raise ValueError("SMOTE requires encoded numeric features and at least two observations per class")
                    from imblearn.over_sampling import SMOTE
                    sampler = SMOTE(random_state=self.spec.seed, k_neighbors=min(5, int(counts.min()) - 1))
                model_matrix, values = sampler.fit_resample(matrix, values)
            iterations = self.spec.iteration_schedule.get(target, self.spec.frozen_iterations) or self.schedules.get(target, {}).get("iterations")
            iterations = refit_iterations(self.spec, iterations, len(X), context.get("execution_mode"))
            model_spec = self.spec
            if iterations and self.spec.model in ITERATION_PARAMETER:
                model_spec = self.spec.model_copy(update={"hyperparameters": {**self.spec.hyperparameters, ITERATION_PARAMETER[self.spec.model]: iterations}})
            model = make_estimator(model_spec, classification, context)
            if classification and len(pd.unique(values)) == 1:
                model = DummyClassifier(strategy="prior")
            if classification and self.spec.model == "xgboost" and not isinstance(model, DummyClassifier):
                encoder = LabelEncoder().fit(values)
                self.encoders[target] = encoder
                values = encoder.transform(values)
            fit_kwargs = {}
            if self.spec.model in {"lightgbm", "lgbm_ranker"} and not isinstance(model, DummyClassifier) and isinstance(model_matrix, pd.DataFrame) and any(
                    any(char in str(column) for char in '[]{}":,') for column in model_matrix.columns):
                aliases = [f"feature_{i}" for i in range(model_matrix.shape[1])]
                self.model_feature_names = dict(zip(aliases, map(str, model_matrix.columns)))
                fit_kwargs["feature_name"] = aliases
                fit_kwargs["categorical_feature"] = [i for i, dtype in enumerate(model_matrix.dtypes)
                    if isinstance(dtype, pd.CategoricalDtype)]
                self.preprocessor.processing_summary.update(model_feature_name_policy="positional_safe_aliases",
                    feature_name_mapping=self.model_feature_names, feature_name_mapping_hash=digest(self.model_feature_names))
            if context.get("sample_weights") is not None:
                fit_kwargs["sample_weight"] = context["sample_weights"].reindex(X.index).to_numpy()
            if self.spec.model == "catboost" and self.preprocessor.native:
                fit_kwargs["cat_features"] = self.preprocessor.categorical_columns
            if self.spec.model == "lgbm_ranker":
                if self.task.kind != "ranking":
                    raise ValueError("The ranking model requires a ranking contract")
                group = X[self.task.group_column].astype(str).to_numpy()
                order = np.argsort(group, kind="stable")
                _, sizes = np.unique(group[order], return_counts=True)
                ordered_matrix = model_matrix.iloc[order] if isinstance(model_matrix, pd.DataFrame) else model_matrix[order]
                if "sample_weight" in fit_kwargs:
                    fit_kwargs["sample_weight"] = fit_kwargs["sample_weight"][order]
                model.fit(ordered_matrix, values[order], group=sizes.tolist(), **fit_kwargs)
            elif context.get("schedule_selection") and self.spec.early_stopping_rounds and not isinstance(model, DummyClassifier):
                valid_X, valid_y = context["schedule_validation"]
                valid_values = valid_y[target].to_numpy()
                if self.spec.target_transform == "log1p":
                    valid_values = np.log1p(valid_values)
                if target in self.encoders:
                    valid_values = self.encoders[target].transform(valid_values)
                order = self.encoders[target].classes_.tolist() if target in self.encoders else sorted(y[target].unique().tolist()) if classification else []
                self.schedules[target] = fit_with_schedule(model, self.spec, self.task, target, model_matrix,
                    values, (self.preprocessor.transform(valid_X), valid_values), context, fit_kwargs, order)
            else:
                model.fit(model_matrix, values, **fit_kwargs)
            self.models[target] = model
        self.summary = TrainingSummary(rows=len(X), seconds=time.monotonic() - started,
            selected_parameters={"model": self.spec.model, "hyperparameters": self.spec.hyperparameters},
            imputation=self.preprocessor.summary, preprocessing=self.preprocessor.processing_summary,
            auxiliary_fits=self.preprocessor.summary["auxiliary_fits"] + self.inner_auxiliary_fits,
            fit_diagnostics=self.inner_diagnostics,
            budget={"selected_fit_actual_seconds": time.monotonic() - selected_started,
                "selected_fit_rows": len(X), "auxiliary_fit_actual_seconds": selected_started - started,
                "iteration_schedule": {target: refit_iterations(self.spec,
                    self.spec.iteration_schedule.get(target, self.spec.frozen_iterations) or self.schedules.get(target, {}).get("iterations"),
                    len(X), context.get("execution_mode")) for target in self.task.targets},
                "refit_iteration_policy": self.spec.refit_iteration_policy},
            learning_curve=self.schedules)
        return self.summary

    def _select_checkpoints(self, X, y, context, counts=None):
        from .training_schedule import ITERATION_PARAMETER
        from .evaluation import align_predictions
        if self.spec.model not in ITERATION_PARAMETER:
            raise ValueError("Checkpoint selection requires a supported boosting family")
        a, b = inner_split(X, y, self.task, context["strategy"], self.spec.seed)
        comparisons = []
        for iterations in sorted(set(counts or self.spec.checkpoint_counts)):
            candidate = self.spec.model_copy(update={"checkpoint_counts": [], "early_stopping_rounds": None,
                "iteration_schedule": {}, "frozen_iterations": iterations, "class_offsets": None, "learned_decision": {},
                "hyperparameters": {**self.spec.hyperparameters, ITERATION_PARAMETER[self.spec.model]: iterations}})
            inner = DirectPipeline(candidate, self.task)
            local = {**context, "split_id": "checkpoint", "skip_history_diagnostics": True}
            inner.fit(X.iloc[a], y.iloc[a], local)
            bundle = inner.predict(X.iloc[b], local)
            values = align_predictions(bundle, X.index[b].tolist(), self.task, context["protocol_hash"], "checkpoint")
            score = score_predictions(self.task, y.iloc[b], values, bundle.prediction_type)
            comparisons.append({"iterations": iterations, "score": score})
            self.inner_auxiliary_fits += 1 + inner.summary.auxiliary_fits
        winner = (max if self.task.metric.maximize else min)(comparisons, key=lambda item: item["score"])
        self.schedules = {target: {"iterations": winner["iterations"], "curve": comparisons,
            "scope": "fit_inner_final_output_selection", "decision_fit": "strictly_inside_checkpoint_fit",
            "fit_keys_hash": digest(list(X.index[a])), "evaluation_keys_hash": digest(list(X.index[b]))} for target in self.task.targets}

    def _fit_inverse(self, X, y, context):
        from .research_evaluation import oof_predictions
        base = self.spec.model_copy(update={"log_inverse": "naive", "learned_inverse": {}})
        report = oof_predictions(base, self.task, X, y, {**context, 'minimum_oof_folds': 2})
        positions = {str(key): i for i, key in enumerate(X.index)}
        actual = y.iloc[[positions[key] for key in report["row_keys"]]]
        for target in self.task.targets:
            predicted = np.asarray(report["values"][target])
            residual = np.log1p(actual[target].to_numpy()) - np.log1p(np.maximum(predicted, -1 + 1e-12))
            raw = float(np.mean(np.exp(np.clip(residual, -2, 2))))
            fold_values = []
            for fold in report["folds"]:
                mask = np.isin(report["row_keys"], list(map(str, fold["evaluation_keys"])))
                fold_values.append(float(np.mean(np.exp(np.clip(residual[mask], -2, 2)))))
            stable = max(fold_values) / max(1e-12, min(fold_values)) <= 1.5
            factor = 1 + len(residual) / (len(residual) + 100) * (np.clip(raw, .5, 2) - 1) if stable else 1.
            self.inverse_state[target] = {"factor": float(factor), "status": "fitted" if stable else "rejected_unstable_folds",
                "scope": "fit_only_oof_residuals", "folds_hash": report["folds_hash"], "row_keys_hash": digest(report["row_keys"]),
                "fold_factors": fold_values, "quality_claimed": False}
        self.inner_auxiliary_fits += len(report["folds"])

    def _fit_history_diagnostics(self, X, y, context):
        a, b = inner_split(X, y, self.task, context["strategy"], self.spec.seed)
        spec = self.spec.model_copy(update={"prediction_policy":"default", "postprocess_fit_policy":"none", "fit_diagnostics":"none"})
        inner = DirectPipeline(spec, self.task)
        local = {**context, "skip_history_diagnostics":True, "split_id":"history_inner"}
        inner.fit(X.iloc[a], y.iloc[a], local)
        bundle = inner.predict(X.iloc[b], local)
        features = inner.preprocessor.feature_frame(X.iloc[b])
        reports = {}
        for column in features:
            if not column.startswith("__history_") or not column.endswith("_count"):
                continue
            reports[column] = {}
            for name, mask in (("cold",features[column].to_numpy()==0),("known",features[column].to_numpy()>0)):
                size = int(mask.sum())
                record = {"rows":size,"uncertain":size<30,"status":"available" if size else "empty"}
                if size:
                    values = {target:np.asarray(bundle.values[target])[mask] for target in self.task.targets}
                    record["score"] = score_predictions(self.task, y.iloc[b].iloc[np.flatnonzero(mask)], values, bundle.prediction_type)
                    if not self.task.classes:
                        record["bias"] = {target:float(np.mean(y.iloc[b][target].to_numpy(dtype=float)[mask]-values[target]))
                            for target in self.task.targets}
                reports[column][name] = record
        self.inner_auxiliary_fits += 1 + inner.summary.auxiliary_fits
        self.inner_diagnostics = {**(self.inner_diagnostics or {}), "entity_history": {
            "scope":"fit_inner_holdout", "fit_rows":len(a), "evaluation_rows":len(b), "reports":reports,
            "entity_identity_is_hypothesis":True, "not_external_quality_evidence":True}}

    def _fit_postprocessing(self, X, y, context):
        if self.task.kind != "classification":
            raise ValueError("Inner diagnostics/postprocessing require single-target classification")
        target = self.task.targets[0]
        classes = self.task.classes[target]
        if target in self.spec.learned_decision:
            self.postprocessors[target] = self.spec.learned_decision[target]
            return
        if self.spec.class_offsets is not None:
            if len(self.spec.class_offsets) != len(classes):
                raise ValueError("Frozen offsets and declared class order differ")
            self.postprocessors[target] = {"offsets": np.asarray(self.spec.class_offsets)}
            return
        if self.spec.prediction_policy == "class_offsets" and self.spec.postprocess_fit_policy == "pooled_oof":
            from .research_evaluation import oof_predictions
            from .decision import fit_decision
            base = self.spec.model_copy(update={"prediction_policy": "default", "postprocess_fit_policy": "none",
                "fit_diagnostics": "none", "class_offsets": None, "learned_decision": {}})
            report = oof_predictions(base, self.task, X, y, {**context, 'minimum_oof_folds': 2})
            positions = {str(key): i for i, key in enumerate(X.index)}
            actual = y.iloc[[positions[key] for key in report["row_keys"]]]
            self.postprocessors[target] = fit_decision(self.task, actual, report["values"][target],
                report["row_keys"], report["folds_hash"], self.spec.decision_method)
            self.inner_auxiliary_fits += len(report["folds"])
            return
        a, b = inner_split(X, y, self.task, context["strategy"], self.spec.seed)
        inner_spec = self.spec.model_copy(update={"prediction_policy": "default", "postprocess_fit_policy": "none", "fit_diagnostics": "none"})
        inner = DirectPipeline(inner_spec, self.task)
        inner_context = {key: value for key, value in context.items() if key not in {"schedule_selection", "schedule_validation", "optuna_trial", "rolling_folds"}}
        inner.fit(X.iloc[a], y.iloc[a], {**inner_context, "skip_history_diagnostics":True})
        prediction_context = {**context, "split_id": "inner"}
        bundle = inner.predict(X.iloc[b], prediction_context)
        self.inner_auxiliary_fits += 1 + inner.summary.auxiliary_fits
        raw = np.asarray(bundle.values[target])
        if self.spec.fit_diagnostics != "none":
            from .evaluation import confusion_details
            predicted = np.asarray(classes)[np.argmax(raw, axis=1)]
            detail = confusion_details(y.iloc[b][target], predicted, classes)
            self.inner_diagnostics = {"scope": "fit_inner_holdout", "semantics": "full_confusion_per_class",
                "fit_rows": len(a), "evaluation_rows": len(b), "fit_keys_hash": digest(list(X.iloc[a].index)),
                "evaluation_keys_hash": digest(list(X.iloc[b].index)), **detail,
                "score": score_predictions(self.task, y.iloc[b], {target: raw}, "probabilities"),
                "not_external_validation": True, "not_unbiased_for_tuned_postprocessing": True}
        if self.spec.prediction_policy == "default":
            return
        if self.spec.prediction_policy == "class_offsets":
            from .decision import fit_decision
            self.postprocessors[target] = fit_decision(self.task, y.iloc[b], raw,
                X.iloc[b].index, None, self.spec.decision_method,
                scope="fit_only_inner_holdout_decision_training")
            return
        positive = self.task.metric.positive_label
        classes = self.task.classes[target]
        if positive is None:
            positive = classes[1]
        position = classes.index(positive)
        probabilities = np.asarray(bundle.values[target])[:, position]
        if self.spec.prediction_policy == "threshold":
            if self.task.metric.name not in {"f1", "accuracy"} or self.task.output.prediction_type == "probabilities":
                raise ValueError("Threshold search requires a label metric and label output")
            best, threshold = -np.inf, 0.5
            for candidate in np.linspace(0.05, 0.95, 37):
                labels = np.where(probabilities >= candidate, positive, classes[1 - position])
                value = score_predictions(self.task, y.iloc[b], {target: labels}, "labels")
                if value > best:
                    best, threshold = value, float(candidate)
            self.postprocessors[target] = {"threshold": threshold, "position": position}
        else:
            actual = (y.iloc[b][target].to_numpy() == positive).astype(int)
            if len(np.unique(actual)) < 2:
                raise ValueError("Calibration inner holdout lacks both classes")
            calibrator = LogisticRegression(C=1.0, random_state=self.spec.seed)
            logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))
            calibrator.fit(logits[:, None], actual)
            self.postprocessors[target] = {"calibration_coef": float(calibrator.coef_[0, 0]),
                "calibration_intercept": float(calibrator.intercept_[0]), "position": position}

    def predict(self, X, context):
        matrix = self.preprocessor.transform(X)
        predictions, raw_probabilities, calibrated_probabilities, decisions = {}, {}, {}, {}
        kind = "probabilities" if self.task.classes else "scores" if self.task.kind == "ranking" else "values"
        for target, model in self.models.items():
            if self.task.classes:
                if not hasattr(model, "predict_proba"):
                    raise ValueError("This classifier does not produce probabilities")
                raw = np.asarray(model.predict_proba(matrix))
                labels = model.classes_.tolist()
                if target in self.encoders:
                    labels = self.encoders[target].inverse_transform(np.asarray(labels, dtype=int)).tolist()
                output = np.zeros((len(X), len(self.task.classes[target])))
                for i, label in enumerate(labels):
                    output[:, self.task.classes[target].index(label)] = raw[:, i]
                raw_probabilities[target] = output.copy().tolist()
                if target in self.postprocessors:
                    post = self.postprocessors[target]
                    if "offsets" in post:
                        from .decision import decide
                        if self.task.output.prediction_type != "probabilities":
                            decisions[target] = decide(output, self.task.classes[target], post).tolist()
                    elif "calibration_coef" in post:
                        position = post["position"]
                        p = np.clip(output[:, position], 1e-6, 1 - 1e-6)
                        z = post["calibration_coef"] * np.log(p / (1 - p)) + post["calibration_intercept"]
                        calibrated = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
                        output[:, position], output[:, 1 - position] = calibrated, 1 - calibrated
                        calibrated_probabilities[target] = output.copy().tolist()
                    elif "threshold" in post:
                        position = post["position"]
                        label_mask = output[:, position] >= post["threshold"]
                        decisions[target] = np.asarray(self.task.classes[target])[np.where(label_mask, position, 1 - position)].tolist()
                predictions[target] = decisions.get(target, output)
            else:
                predicted = np.asarray(model.predict(matrix)).reshape(-1)
                if self.spec.target_transform == "log1p":
                    factor = self.inverse_state.get(target, {}).get("factor", 1.)
                    predicted = np.exp(np.clip(predicted + np.log(factor), -700, 700)) - 1
                predictions[target] = predicted
        if decisions:
            kind = "labels"
        return prediction_bundle(predictions, X, context, kind, self.task.classes,
            raw_probabilities=raw_probabilities, calibrated_probabilities=calibrated_probabilities, decision_values=decisions)

    def frozen_spec(self):
        schedule = self.spec.iteration_schedule or {target: row["iterations"] for target, row in self.schedules.items()}
        updates = {"iteration_schedule": schedule, "fit_diagnostics": "none", "learned_inverse": self.inverse_state}
        # Scale refit iterations only for a genuinely frozen boosting schedule.
        # Fixed-iteration or non-boosting models keep the field unset so their
        # frozen representation stays compatible with earlier runs.
        if schedule or self.spec.iteration_scale_rows:
            updates["iteration_scale_rows"] = self.spec.iteration_scale_rows or len(self.training_rows)
        if self.spec.prediction_policy == "class_offsets":
            offsets = np.asarray(self.postprocessors[self.task.targets[0]]["offsets"]).tolist()
            updates["class_offsets"] = offsets
            updates["learned_decision"] = {target: {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in state.items()} for target, state in self.postprocessors.items()}
        elif self.postprocessors:
            updates["learned_decision"] = self.postprocessors
        return self.spec.model_copy(update=updates)


def build_pipeline(spec: PipelineSpec, task: TaskSpec, context):
    require_compatible(spec, task, context.get("strategy"))
    if context.get("rolling_folds"):
        from .rolling import RollingPipeline
        return RollingPipeline(spec, task)
    if spec.engine == "autogluon" and spec.prediction_policy == "class_offsets":
        from .automl_decision import AutoMLDecisionPipeline
        return AutoMLDecisionPipeline(spec, task)
    if spec.engine == "custom":
        from pathlib import Path
        from .custom import load_module, validate_code, TransformerPipeline
        from .util import file_hash
        from .dependency_policy import install_dependency_guard
        path = Path(context["workdir"]) / "candidate.py"
        if not spec.custom_oof or spec.custom_component not in {"pipeline", "transformer"} or file_hash(path) != spec.custom_hash:
            raise ValueError("Custom OOF requires an opted-in verified factory and matching code hash")
        validate_code(path.read_text(), spec.custom_component)
        install_dependency_guard(path, Path(context["workdir"]))
        module = load_module(path)
        local = {**context, "seed": spec.seed, "pipeline": spec.model_dump(mode="json")}
        return TransformerPipeline(module.build_transformer(local), spec, task) if spec.custom_component == "transformer" else module.build_pipeline(local)
    if spec.shift_strategy != "none":
        from .shift import ShiftPipeline
        return ShiftPipeline(spec, task)
    if spec.engine == "lightautoml":
        from .lightautoml_adapter import LightAutoMLPipeline
        return LightAutoMLPipeline(spec, task)
    if spec.engine in {"flaml", "autogluon", "optuna"}:
        from .search import SearchPipeline
        return SearchPipeline(spec, task)
    if task.kind == "forecasting":
        from .forecasting import ForecastPipeline
        return ForecastPipeline(spec, task)
    return DirectPipeline(spec, task)
