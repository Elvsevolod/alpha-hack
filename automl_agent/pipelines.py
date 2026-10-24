from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder

from .api import prediction_bundle
from .contracts import PipelineSpec, TaskSpec, TrainingSummary
from .evaluation import score_predictions
from .preprocessing import FeatureMatrix
from .validation import inner_split
from .compatibility import ranking_label_issue, require_compatible
from .util import digest

PARAMETERS = {
    "constant": set(),
    "ridge": {"alpha", "fit_intercept", "solver", "tol", "max_iter", "positive"},
    "spline": {"alpha", "fit_intercept", "solver", "tol", "max_iter", "positive"},
    "elastic_net": {"alpha", "l1_ratio", "fit_intercept", "max_iter", "tol", "positive", "selection"},
    "logistic": {"C", "penalty", "solver", "l1_ratio", "max_iter", "tol", "class_weight", "fit_intercept", "intercept_scaling", "n_jobs"},
    "sgd": {"loss", "penalty", "alpha", "max_iter", "tol", "l1_ratio", "epsilon", "average", "learning_rate", "eta0", "power_t", "class_weight", "fit_intercept"},
    "random_forest": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "bootstrap", "max_samples", "criterion", "ccp_alpha", "max_leaf_nodes", "class_weight", "n_jobs"},
    "extra_trees": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "bootstrap", "max_samples", "criterion", "ccp_alpha", "max_leaf_nodes", "class_weight", "n_jobs"},
    "hist_gradient_boosting": {"max_iter", "learning_rate", "max_leaf_nodes", "max_depth", "min_samples_leaf", "l2_regularization", "max_bins", "loss", "early_stopping", "max_features"},
    "mlp": {"hidden_layer_sizes", "activation", "solver", "alpha", "batch_size", "learning_rate", "learning_rate_init", "power_t", "max_iter", "shuffle", "tol", "momentum", "nesterovs_momentum", "early_stopping", "beta_1", "beta_2", "epsilon", "n_iter_no_change", "max_fun"},
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
    if spec.model in {"lightgbm", "lgbm_ranker"}:
        # LightGBM's primary name wins over its sklearn alias, including when
        # an Optuna trial overrides a normalized base value.
        for primary, alias in (("feature_fraction", "colsample_bytree"),
                ("bagging_fraction", "subsample"), ("bagging_freq", "subsample_freq"),
                ("lambda_l2", "reg_lambda")):
            if primary in parameters:
                parameters[alias] = parameters.pop(primary)
    return parameters


def make_estimator(spec: PipelineSpec, classification: bool, context):
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
    if spec.model == "mlp":
        if p.get("early_stopping"):
            raise ValueError("MLP automatic random early stopping is disabled; use fixed epochs or inner Optuna")
        p["hidden_layer_sizes"] = tuple(p.get("hidden_layer_sizes", (64,)))
        p.setdefault("max_iter", 100)
        p["early_stopping"] = False
        return (MLPClassifier if classification else MLPRegressor)(random_state=seed, **p)
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
        if self.spec.prediction_policy != "default" or self.spec.fit_diagnostics != "none":
            self._fit_postprocessing(X, y, context)
        self.preprocessor = FeatureMatrix(self.spec, self.task, context["cpu"], context["memory_mb"])
        matrix = self.preprocessor.fit_transform(X, y)
        if self.spec.model == "mlp":
            hidden = self.spec.hyperparameters.get("hidden_layer_sizes", (64,))
            widths = [matrix.shape[1], *hidden, max(1, max(map(len, self.task.classes.values()), default=1))]
            parameters = sum((a + 1) * b for a, b in zip(widths, widths[1:]))
            if parameters > 5_000_000 or parameters * 8 * 10 > context["memory_mb"] * 1024 * 1024 * 0.2:
                raise MemoryError("MLP parameter and optimizer memory bound exceeded")
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
            model = make_estimator(self.spec, classification, context)
            if classification and len(pd.unique(values)) == 1:
                model = DummyClassifier(strategy="prior")
            if classification and self.spec.model == "xgboost" and not isinstance(model, DummyClassifier):
                encoder = LabelEncoder().fit(values)
                self.encoders[target] = encoder
                values = encoder.transform(values)
            fit_kwargs = {}
            if self.spec.model in {"lightgbm", "lgbm_ranker"} and isinstance(model_matrix, pd.DataFrame) and any(
                    any(char in str(column) for char in '[]{}":,') for column in model_matrix.columns):
                # Keep task/source names unchanged. LightGBM's JSON model format
                # requires safe, collision-free internal names, including indicators.
                aliases = [f"feature_{i}" for i in range(model_matrix.shape[1])]
                self.model_feature_names = dict(zip(aliases, map(str, model_matrix.columns)))
                fit_kwargs["feature_name"] = aliases
                fit_kwargs["categorical_feature"] = [i for i, dtype in enumerate(model_matrix.dtypes)
                    if isinstance(dtype, pd.CategoricalDtype)]
                self.preprocessor.processing_summary["model_feature_name_policy"] = "positional_safe_aliases"
                self.preprocessor.processing_summary["feature_name_mapping_hash"] = digest(self.model_feature_names)
            if self.spec.model == "catboost" and self.preprocessor.native:
                fit_kwargs["cat_features"] = self.preprocessor.categorical_columns
            if self.spec.model == "lgbm_ranker":
                if self.task.kind != "ranking":
                    raise ValueError("The ranking model requires a ranking contract")
                group = X[self.task.group_column].astype(str).to_numpy()
                order = np.argsort(group, kind="stable")
                _, sizes = np.unique(group[order], return_counts=True)
                ordered_matrix = model_matrix.iloc[order] if isinstance(model_matrix, pd.DataFrame) else model_matrix[order]
                model.fit(ordered_matrix, values[order], group=sizes.tolist(), **fit_kwargs)
            else:
                model.fit(model_matrix, values, **fit_kwargs)
            self.models[target] = model
        self.summary = TrainingSummary(rows=len(X), seconds=time.monotonic() - started,
            selected_parameters={"model": self.spec.model, "hyperparameters": self.spec.hyperparameters},
            imputation=self.preprocessor.summary, preprocessing=self.preprocessor.processing_summary,
            auxiliary_fits=self.preprocessor.summary["auxiliary_fits"] + self.inner_auxiliary_fits,
            fit_diagnostics=self.inner_diagnostics)
        return self.summary

    def _fit_postprocessing(self, X, y, context):
        if self.task.kind != "classification":
            raise ValueError("Inner diagnostics/postprocessing require single-target classification")
        target = self.task.targets[0]
        classes = self.task.classes[target]
        if self.spec.class_offsets is not None:
            if len(self.spec.class_offsets) != len(classes):
                raise ValueError("Frozen offsets and declared class order differ")
            self.postprocessors[target] = {"offsets": np.asarray(self.spec.class_offsets)}
            return
        a, b = inner_split(X, y, self.task, context["strategy"], self.spec.seed)
        inner_spec = self.spec.model_copy(update={"prediction_policy": "default", "postprocess_fit_policy": "none", "fit_diagnostics": "none"})
        inner = DirectPipeline(inner_spec, self.task)
        inner.fit(X.iloc[a], y.iloc[a], context)
        prediction_context = {**context, "split_id": "inner"}
        bundle = inner.predict(X.iloc[b], prediction_context)
        self.inner_auxiliary_fits = 1 + inner.summary.auxiliary_fits
        raw = np.asarray(bundle.values[target])
        if self.spec.fit_diagnostics != "none":
            from sklearn.metrics import precision_recall_fscore_support
            predicted = np.asarray(classes)[np.argmax(raw, axis=1)]
            precision, recall, f1, support = precision_recall_fscore_support(y.iloc[b][target], predicted,
                labels=classes, average=None, zero_division=0)
            rows = []
            for i, label in enumerate(classes):
                available = support[i] >= 30 and int((predicted == label).sum()) >= 30
                rows.append({"class": label, "status": "available" if available else "suppressed",
                    **({"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])} if available else {"reason": "minimum_aggregate_support"})})
            self.inner_diagnostics = {"scope": "fit_inner_holdout", "semantics": "full_confusion_per_class",
                "fit_rows": len(a), "evaluation_rows": len(b), "fit_keys_hash": digest(list(X.iloc[a].index)),
                "evaluation_keys_hash": digest(list(X.iloc[b].index)), "classes": rows,
                "score": score_predictions(self.task, y.iloc[b], {target: raw}, "probabilities") if len(b) >= 30 else None,
                "not_external_validation": True, "not_unbiased_for_tuned_postprocessing": True}
        if self.spec.prediction_policy == "default":
            return
        if self.spec.prediction_policy == "class_offsets":
            logits = np.log(np.clip(raw, 1e-12, 1))
            offsets = np.zeros(len(classes))
            def objective(values):
                labels = np.asarray(classes)[np.argmax(logits + values, axis=1)]
                return score_predictions(self.task, y.iloc[b], {target: labels}, "labels") - .001 * float(values @ values)
            best = objective(offsets)
            for _ in range(2):
                for column in range(len(classes)):
                    selected = offsets.copy()
                    for value in (-.5, -.25, 0., .25, .5):
                        trial = offsets.copy(); trial[column] = value
                        score = objective(trial)
                        if score > best + 1e-12:
                            best, selected = score, trial
                    offsets = selected
            self.postprocessors[target] = {"offsets": offsets}
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
            self.postprocessors[target] = {"calibrator": calibrator, "position": position}

    def predict(self, X, context):
        matrix = self.preprocessor.transform(X)
        predictions = {}
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
                if target in self.postprocessors:
                    post = self.postprocessors[target]
                    if "offsets" in post:
                        decision = np.argmax(np.log(np.clip(output, 1e-12, 1)) + post["offsets"], axis=1)
                        output = np.eye(len(self.task.classes[target]))[decision]
                    elif "calibrator" in post:
                        position = post["position"]
                        p = np.clip(output[:, position], 1e-6, 1 - 1e-6)
                        calibrated = post["calibrator"].predict_proba(np.log(p / (1 - p))[:, None])[:, 1]
                        output[:, position], output[:, 1 - position] = calibrated, 1 - calibrated
                    elif "threshold" in post:
                        position = post["position"]
                        label_mask = output[:, position] >= post["threshold"]
                        output[:, position], output[:, 1 - position] = label_mask.astype(float), (~label_mask).astype(float)
                predictions[target] = output
            else:
                predicted = np.asarray(model.predict(matrix)).reshape(-1)
                if self.spec.target_transform == "log1p":
                    predicted = np.expm1(predicted)
                predictions[target] = predicted
        return prediction_bundle(predictions, X, context, kind, self.task.classes)

    def frozen_spec(self):
        if self.spec.prediction_policy == "class_offsets":
            offsets = self.postprocessors[self.task.targets[0]]["offsets"].tolist()
            return self.spec.model_copy(update={"class_offsets": offsets, "fit_diagnostics": "none"})
        if self.spec.fit_diagnostics != "none":
            return self.spec.model_copy(update={"fit_diagnostics": "none"})
        return self.spec


def build_pipeline(spec: PipelineSpec, task: TaskSpec, context):
    require_compatible(spec, task, context.get("strategy"))
    if spec.engine in {"flaml", "autogluon", "optuna"}:
        from .search import SearchPipeline
        return SearchPipeline(spec, task)
    if task.kind == "forecasting":
        from .forecasting import ForecastPipeline
        return ForecastPipeline(spec, task)
    return DirectPipeline(spec, task)
