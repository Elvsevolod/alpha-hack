from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .api import prediction_bundle
from .contracts import PipelineSpec, TrainingSummary
from .evaluation import align_predictions, score_predictions
from .pipelines import DirectPipeline, validate_parameters
from .util import atomic_json, read_json
from .validation import inner_split
from .compatibility import require_compatible
from .diagnostics import trial_event


def suggest_parameters(trial, space):
    values, pending = {}, dict(space)
    while pending:
        changed = False
        for name, rule in list(pending.items()):
            condition = rule.get("when")
            if condition and condition["parameter"] in pending:
                continue
            del pending[name]
            changed = True
            if condition and values.get(condition["parameter"]) not in condition["in"]:
                continue
            kind = rule["type"]
            if kind == "categorical":
                values[name] = trial.suggest_categorical(name, rule["choices"])
            elif kind == "int":
                values[name] = trial.suggest_int(name, rule["low"], rule["high"], step=rule.get("step", 1), log=rule.get("log", False))
            elif kind == "float":
                values[name] = trial.suggest_float(name, rule["low"], rule["high"], log=rule.get("log", False), step=rule.get("step"))
            else:
                raise ValueError("Unknown search-space parameter type")
        if not changed:
            raise ValueError("Cyclic conditional search space")
    return values


def normalized_search_space(spec):
    from .pipelines import effective_parameters, PARAMETERS
    result, names = {}, {}
    for name, rule in spec.search_space.items():
        mapped = effective_parameters(spec.model_copy(update={"hyperparameters": {name: 0}}))
        if len(mapped) != 1 or next(iter(mapped)) not in PARAMETERS.get(spec.model, set()):
            raise ValueError("Unknown or controller-managed search parameter: " + name)
        canonical = next(iter(mapped))
        if canonical in result:
            raise ValueError("Conflicting parameter aliases in the search space")
        if not isinstance(rule, dict) or set(rule) - {"type", "low", "high", "step", "log", "choices", "when"}:
            raise ValueError("Search rules must use the supported typed schema")
        kind = rule.get("type")
        if kind == "categorical":
            if not isinstance(rule.get("choices"), list) or not rule["choices"] or any(value is not None and type(value) not in (str, int, float, bool) for value in rule["choices"]):
                raise ValueError("Categorical search requires nonempty scalar choices")
        elif kind in {"int", "float"}:
            low, high = rule.get("low"), rule.get("high")
            if type(low) not in (int, float) or type(high) not in (int, float) or not np.isfinite([low, high]).all() or low > high:
                raise ValueError("Search bounds must be ordered finite numbers")
            if kind == "int" and (type(low) is not int or type(high) is not int or type(rule.get("step", 1)) is not int):
                raise ValueError("Integer search requires integer bounds/step")
            step = rule.get("step", 1 if kind == "int" else None)
            if step is not None and (type(step) not in (int, float) or not np.isfinite(step) or step <= 0):
                raise ValueError("Search step must be finite and positive")
            if rule.get("log") and (low <= 0 or step not in (None, 1) or kind == "float" and step is not None):
                raise ValueError("Log search requires positive bounds and a compatible step")
        else:
            raise ValueError("Unknown search-space parameter type")
        result[canonical], names[name] = dict(rule), canonical
    for rule in result.values():
        if rule.get("when"):
            condition = rule["when"]
            if not isinstance(condition, dict) or set(condition) != {"parameter", "in"} or condition["parameter"] not in names or not isinstance(condition["in"], list):
                raise ValueError("Conditional search requires a known dependency and allowed values")
            rule["when"] = {**condition, "parameter": names[condition["parameter"]]}
    pending = set(result)
    while pending:
        ready = {name for name in pending if not result[name].get("when") or result[name]["when"]["parameter"] not in pending}
        if not ready:
            raise ValueError("Cyclic conditional search space")
        pending -= ready
    return result


def autogluon_portfolio(spec, task, learners):
    configured = spec.autogluon_configurations
    explicit = bool(configured)
    if not explicit:
        configured = {"GBM": [{"num_leaves": 31}, {"num_leaves": 63, "reg_lambda": 1.0}],
            "CAT": [{"depth": 6}, {"depth": 8, "l2_leaf_reg": 5.0}],
            "XGB": [{"max_depth": 4}, {"max_depth": 8, "reg_lambda": 2.0}]}
    if set(configured) - {"GBM", "CAT", "XGB"} or sum(map(len, configured.values())) > 6 or any(not values for values in configured.values()):
        raise ValueError("AutoGluon explicit portfolio requires at most six configurations of allowed tree families")
    families = {"GBM": "lightgbm", "CAT": "catboost", "XGB": "xgboost"}
    seeds = {"GBM": {"seed_value": spec.seed}, "CAT": {"random_seed": spec.seed}, "XGB": {"random_state": spec.seed}}
    result = {}
    for learner in learners:
        entries = configured.get(learner)
        if entries is None:
            result[learner] = seeds[learner]
            continue
        result[learner] = []
        for index, parameters in enumerate(entries):
            direct = PipelineSpec(model=families[learner], hyperparameters=parameters)
            if set(parameters) & {"ag_args", "ag_args_fit", "ag_args_ensemble"}:
                raise ValueError("AutoGluon internal/resource configuration is controller-owned")
            from .pipelines import CONTROLLED_PARAMETERS
            if set(parameters) & CONTROLLED_PARAMETERS.get(direct.model, set()):
                raise ValueError("AutoGluon configuration cannot override managed resources")
            normalized = validate_parameters(direct, task)
            suffix = "explicit" if explicit else "portfolio"
            result[learner].append({**normalized, **seeds[learner], "ag_args": {"name_suffix": f"_{suffix}_{index}"}})
    return result


class SearchPipeline:
    def __init__(self, spec, task):
        self.spec, self.task = spec, task
        self.models, self.training_keys = {}, []
        self.autogluon_train_keys = {}

    def fit(self, X, y, context):
        if self.spec.engine == "optuna":
            self.spec = self.spec.model_copy(update={"search_space": normalized_search_space(self.spec)})
        require_compatible(self.spec, self.task, context.get("strategy"))
        self.training_keys = list(map(str, X.index))
        if self.spec.engine == "optuna":
            return self._tune(X, y, context)
        if self.task.kind in {"forecasting", "ranking"} or self.task.text_columns:
            raise ValueError("This AutoML adapter is limited to verified tabular outputs; use a direct specialized pipeline")
        if self.spec.transformations or self.spec.imputation.numeric not in {"median", "native"} or self.spec.imputation.categorical != "missing_category":
            raise ValueError("Custom preprocessing with AutoML is not fit-safe in this adapter; use direct + Optuna")
        if self.spec.prediction_policy != "default" or self.spec.target_transform != "none":
            raise ValueError("AutoML postprocessing requires a separately verified adapter")
        started = time.monotonic()
        fit, val = inner_split(X, y, self.task, context["strategy"], self.spec.seed)
        if self.spec.engine == "autogluon":
            # Final refit_full is called on the original selected model, so
            # remember only rows that model itself saw during predictor.fit.
            self.training_keys = list(map(str, X.index[fit]))
        self.inner_split = {"fit_keys": X.index[fit].tolist(), "validation_keys": X.index[val].tolist(), "scope": "external_fit_only", "strategy": context["strategy"]}
        self.selected, self.inner_scores = {}, {}
        self.raw_feature_columns = list(self.task.feature_columns)
        if self.spec.engine == "autogluon" and self.spec.autogluon_profile != "selection" and context["strategy"] == "group":
            self.raw_feature_columns = list(dict.fromkeys([*self.raw_feature_columns, self.task.group_column]))
        estimates = context.get("phase_estimates", {})
        selected_fit = max(.2, estimates.get("selected_fit_seconds", 2))
        prediction = max(.1, estimates.get("prediction_seconds", 1))
        self.budget_evidence = {"mode": context.get("budget_mode", "research"),
            "requested_seconds": context.get("requested_budget_seconds", context["budget_seconds"]),
            "allocated_seconds": context["budget_seconds"], "n_trials": self.spec.n_trials,
            "selected_fit_rows": len(fit) if self.spec.engine == "autogluon" else len(X),
            "phase_reserves": {"selected_fit_seconds": selected_fit, "prediction_seconds": prediction,
                "publication_seconds": estimates.get("publication_seconds", 1)}, "targets": {}}
        for i, target in enumerate(self.task.targets):
            remaining_targets = len(self.task.targets) - i
            allocation = (context["budget_seconds"] - (time.monotonic() - started)
                - remaining_targets * selected_fit - prediction - estimates.get("publication_seconds", 1)) / remaining_targets
            if self.task.classes and y[target].nunique() == 1:
                task = self.task.model_copy(update={"targets": [target], "classes": {target: self.task.classes[target]}})
                model = DirectPipeline(PipelineSpec(model="constant"), task)
                model.fit(X, y[[target]], context)
                self.models[target] = ("constant", model)
                self.selected[target] = {"model": "constant"}
            else:
                if allocation < .2:
                    raise ValueError("AutoML phase reserves exceed the allocated budget; increase this candidate's budget or use a cheaper configuration")
                if self.spec.engine == "flaml":
                    self._fit_flaml(target, X, y, fit, val, context, allocation)
                else:
                    self._fit_autogluon(target, X, y, fit, val, context, allocation)
        self.summary = TrainingSummary(rows=len(X), seconds=time.monotonic() - started,
            selected_parameters=self.selected, inner_scores={"scores": self.inner_scores, "split": self.inner_split},
            imputation={"implementation": "engine_native_inside_inner_fit", "scope": "fit_only"},
            budget={**self.budget_evidence, "actual_seconds": time.monotonic() - started})
        return self.summary

    def _flaml_metric(self, target, original_labels):
        metric = self.task.metric.model_copy(update={"aggregation": "uniform", "target_weights": None})
        task = self.task.model_copy(update={"targets": [target], "classes": {target: self.task.classes[target]} if self.task.classes else {},
            "metric": metric})

        def metric(X_val, y_val, estimator, labels=None, X_train=None, y_train=None, *args, **kwargs):
            if task.classes:
                raw = np.asarray(estimator.predict_proba(X_val))
                local = estimator.classes_.tolist()
                probability = np.zeros((len(X_val), len(task.classes[target])))
                for i, encoded_label in enumerate(local):
                    label = original_labels[int(encoded_label)]
                    probability[:, task.classes[target].index(label)] = raw[:, i]
                actual = np.asarray(original_labels)[np.asarray(y_val, dtype=int)]
                predictions, kind = {target: probability}, "probabilities"
            else:
                actual = np.asarray(y_val)
                predictions, kind = {target: np.asarray(estimator.predict(X_val))}, "values"
            value = score_predictions(task, pd.DataFrame({target: actual}), predictions, kind)
            return (-value if task.metric.maximize else value), {"primary": value, "scope": "inner_fit_holdout"}

        return metric

    def _fit_flaml(self, target, X, y, fit, val, context, allocation):
        group_values = X[self.task.group_column].to_numpy() if context["strategy"] == "group" else None
        X = X[self.task.feature_columns]
        from flaml import AutoML
        allowed = {"lgbm", "rf", "extra_tree", "xgboost", "xgb_limitdepth", "catboost"}
        learners = self.spec.learners or ["lgbm", "rf", "extra_tree"]
        if not set(learners) <= allowed:
            raise ValueError("FLAML learners include an unverified or download-requiring mode")
        if not self.task.classes and set(learners) & {"lrl1", "lrl2"}:
            raise ValueError("Logistic FLAML learners require classification")
        log = Path(context["workdir"]) / f"flaml_{target}.jsonl"
        model = AutoML()
        labels = sorted(y[target].unique().tolist()) if self.task.classes else []
        trial_arguments = {} if self.spec.n_trials is None else {"max_iter": self.spec.n_trials}
        search_started = time.monotonic()
        from .diagnostic_automl import flaml_trials
        with flaml_trials(context):
            model.fit(X_train=X.iloc[fit].reset_index(drop=True), y_train=y.iloc[fit][target].reset_index(drop=True),
                X_val=X.iloc[val].reset_index(drop=True), y_val=y.iloc[val][target].reset_index(drop=True),
                task="classification" if self.task.classes else "regression", estimator_list=learners,
                time_budget=allocation, early_stop=False, metric=self._flaml_metric(target, labels), **trial_arguments,
                n_jobs=context["cpu"], n_concurrent_trials=1, use_ray=False, use_spark=False,
                sample=False, ensemble=False, eval_method="holdout", retrain_full=False,
                auto_augment=False, hpo_method="cfo", seed=self.spec.seed, log_file_name=str(log), log_type="all", verbose=0)
        if model.model is None:
            raise ValueError("FLAML completed no model within its budget")
        from flaml.automl.training_log import training_log_reader
        with training_log_reader(str(log)) as reader:
            entries = list(reader.records())
        if not entries:
            raise ValueError("FLAML did not preserve a replayable winning configuration")
        best = min(entries, key=lambda record: (record.validation_loss, -record.sample_size))
        elapsed = time.monotonic() - search_started
        trials = int(getattr(model, "_track_iter", len(entries) - 1)) + 1
        effective_limit = getattr(model, "_max_iter", None)
        active = getattr(model, "_active_estimators", None)
        if effective_limit is not None and trials >= effective_limit:
            reason = "trial_limit_reached"
        elif active == []:
            reason = "engine_space_exhausted"
        elif elapsed >= allocation - max(.1, float(best.trial_time)):
            reason = "calculated_time_budget_exhausted"
        else:
            reason = "engine_stopped_unclassified"
        self.budget_evidence["targets"][target] = {"engine_time_budget": allocation, "max_iter_argument_present": "max_iter" in trial_arguments,
            "max_iter_argument": trial_arguments.get("max_iter"), "effective_engine_max_iter": effective_limit,
            "early_stop": False, "completed_trials": len(entries), "attempted_trials": trials,
            "search_seconds": elapsed, "remaining_engine_seconds": max(0, allocation - elapsed),
            "next_fit_estimate_seconds": float(best.trial_time), "stop_reason": reason}
        record_id = best.record_id
        self.selected[target] = {"learner": best.learner, "configuration": best.config, "record_id": record_id,
            "log_name": log.name, "search_owner": "flaml"}
        self.inner_scores[target] = {"loss": float(best.validation_loss), "scope": "inner_fit_holdout"}
        # Retraining one logged record executes no HPO or model selection.
        selected_fit_started = time.monotonic()
        model = AutoML()
        model.retrain_from_log(str(log), X_train=X.reset_index(drop=True), y_train=y[target].reset_index(drop=True),
            task="classification" if self.task.classes else "regression", train_full=True, record_id=record_id,
            n_jobs=context["cpu"], auto_augment=False, eval_method="holdout",
            split_type="time" if context["strategy"] == "temporal" else "group" if context["strategy"] == "group" else "auto",
            groups=group_values)
        self.budget_evidence["targets"][target]["selected_fit_actual_seconds"] = time.monotonic() - selected_fit_started
        self.models[target] = ("flaml", model)
        self.log_content = getattr(self, "log_content", {})
        self.log_content[target] = log.read_text()

    def _ag_metric(self):
        name = self.task.metric.name
        if name == "f1":
            return "f1" if self.task.metric.average == "binary" else "f1_" + self.task.metric.average
        if name == "average_precision":
            return "average_precision"
        if name in {"rmse", "mse", "mae", "r2", "accuracy", "roc_auc", "log_loss"}:
            return {"rmse": "root_mean_squared_error", "mse": "mean_squared_error", "mae": "mean_absolute_error"}.get(name, name)
        raise ValueError("Metric has no verified AutoGluon mapping")

    def _fit_autogluon(self, target, X, y, fit, val, context, allocation):
        from autogluon.tabular import TabularPredictor
        started = time.monotonic()
        requested_learners = self.spec.learners or ["GBM", "CAT", "XGB"]
        learners = requested_learners if self.spec.n_trials is None else requested_learners[:self.spec.n_trials]
        if not set(learners) <= {"GBM", "CAT", "XGB"}:
            raise ValueError("AutoGluon accepts only the verified CPU tree allowlist")
        if self.task.classes and self.task.metric.name in {"roc_auc", "average_precision"} and len(self.task.classes[target]) != 2:
            raise ValueError("This AutoGluon probability-metric mapping is verified for binary classification only")
        self.autogluon_group_column = self.task.group_column if context["strategy"] == "group" else None
        if self.spec.autogluon_profile != "selection" and context["strategy"] == "exact_group":
            self.autogluon_group_column = "__agent_exact_group__"
            while self.autogluon_group_column in X or self.autogluon_group_column in self.task.targets:
                self.autogluon_group_column += "_"
        train = self._autogluon_frame(X, y, target, context)
        path = Path(context["workdir"]) / f"autogluon_{target}"
        problem = "binary" if self.task.classes and len(self.task.classes[target]) == 2 else "multiclass" if self.task.classes else "regression"
        kwargs = {"positive_class": self.task.metric.positive_label} if problem == "binary" and self.task.metric.positive_label is not None else {}
        bagging = self.spec.autogluon_profile != "selection"
        if bagging and context["strategy"] in {"temporal", "temporal_group"}:
            raise ValueError("AutoGluon random bagging is incompatible with temporal validation; use selection or split-aware OOF blending")
        if bagging and self.autogluon_group_column:
            kwargs["groups"] = self.autogluon_group_column
        predictor = TabularPredictor(label=target, path=str(path), problem_type=problem, eval_metric=self._ag_metric(),
            verbosity=0, log_to_file=False, learner_kwargs={"random_state": self.spec.seed}, **kwargs)
        seed_parameters = {"GBM": {"seed_value": self.spec.seed}, "CAT": {"random_seed": self.spec.seed}, "XGB": {"random_state": self.spec.seed}}
        portfolio = autogluon_portfolio(self.spec, self.task, learners)
        training, tuning = train.iloc[fit].copy(), train.iloc[val].copy()
        self.autogluon_train_keys[target] = list(map(str, training.index))
        if bagging and self.autogluon_group_column:
            training = self._bounded_group_codes(training)
            # The pinned predictor treats groups as a train-only role.
            tuning = tuning.drop(columns=[self.autogluon_group_column])
        from .diagnostic_automl import autogluon_trials
        with autogluon_trials(context):
            predictor.fit(train_data=training, tuning_data=tuning, time_limit=allocation,
                hyperparameters=portfolio,
                num_bag_folds=self.spec.bag_folds if bagging else 0,
                use_bag_holdout=bagging,
                num_stack_levels=1 if self.spec.autogluon_profile == "stacking" else 0,
                fit_weighted_ensemble=bagging, fit_strategy="sequential", num_cpus=context["cpu"], num_gpus=0,
                memory_limit=max(.1, context["memory_mb"] / 1024 * .7),
                dynamic_stacking=False, calibrate_decision_threshold=False,
                ag_args_fit={"num_cpus": context["cpu"], "num_gpus": 0},
                ag_args_ensemble={"fold_fitting_strategy": "sequential_local", "max_base_models": 0, "max_base_models_per_type": 0})
        best = predictor.model_best
        if best is None:
            raise ValueError("AutoGluon completed no model")
        leader = predictor.leaderboard(silent=True, extra_info=True)
        failures = predictor.model_failures()
        atomic_json(Path(context['workdir'])/'autogluon_diagnostic_failures.json',json.loads(failures.to_json(orient='records')))
        atomic_json(Path(context['workdir'])/'autogluon_diagnostic_leaderboard.json',json.loads(leader.to_json(orient='records')))
        model_types = sorted(set(leader["model_type"].map(str)))
        expected_types = {"GBM": "LGBModel", "CAT": "CatBoostModel", "XGB": "XGBoostModel"}
        completed_learners = [learner for learner in learners if any(expected_types[learner] in value for value in model_types)]
        failure_columns = [column for column in ("model", "model_type", "exc_type", "total_time") if column in failures]
        elapsed = time.monotonic() - started
        self.budget_evidence["targets"][target] = {"engine_time_budget": allocation,
            "max_iter_argument_present": False, "max_iter_argument": None, "early_stop": "engine_model_policy",
            "completed_trials": len(leader), "search_seconds": elapsed, "remaining_engine_seconds": max(0, allocation - elapsed),
            "completed_models": leader["model"].tolist(), "configured_learners": learners,
            "configured_portfolio": portfolio, "configured_count": sum(len(value) if isinstance(value, list) else 1 for value in portfolio.values()),
            "search_kind": "explicit_portfolio" if self.spec.autogluon_configurations else "bounded_default_portfolio",
            "builtin_hpo_performed": False, "n_trials_semantics": "legacy_family_limit",
            "requested_learners": requested_learners, "training_seed": self.spec.seed,
            "model_seed_parameters": {learner: seed_parameters[learner] for learner in learners},
            "completed_model_types": model_types, "completed_learners": completed_learners,
            "model_failures": json.loads(failures[failure_columns].to_json(orient="records")),
            "stop_reason": ("trial_limit_reached" if len(learners) < len(requested_learners) else "engine_space_exhausted")
                if set(completed_learners) == set(learners) else "calculated_time_budget_exhausted" if elapsed >= allocation * .95 else "engine_stopped_unclassified"}
        if "fit_time" in leader:
            self.budget_evidence["targets"][target]["selected_fit_actual_seconds"] = float(leader.loc[leader["model"] == best, "fit_time"].iloc[0])
        self.inner_scores[target] = {"leaderboard": json.loads(leader[["model", "score_val"]].to_json(orient="records")), "scope": "inner_fit_holdout"}
        refit_started = time.monotonic()
        if bagging:
            # Freeze architecture, trained iteration counts and ensemble weights.
            # Rebuild OOF features on each permitted training population: the
            # pinned engine's train_data_extra path silently omits stacked rows.
            configurations, graph = {}, {}
            information = predictor.info()['model_info']
            model = predictor._trainer.load_model(best)
            weights = model._get_model_weights() if 'WeightedEnsemble' in best else {best:1.}
            weights = {name:float(weight) for name,weight in weights.items() if weight > 0}
            required, pending = set(), list(weights)
            while pending:
                name = pending.pop()
                if name not in required:
                    required.add(name)
                    pending.extend(information[name].get('stacker_info', {}).get('base_model_names', []))
            by_type = {"LGBModel": "GBM", "CatBoostModel": "CAT", "XGBoostModel": "XGB"}
            for node_name in predictor.model_names():
                if node_name not in required:
                    continue
                item = leader.loc[leader["model"] == node_name].iloc[0]
                info = information[node_name]
                bag = info.get("bagged_info", {})
                learner = by_type.get(bag.get("child_model_type"))
                if learner:
                    level = int(item["stack_level"])
                    suffix = f"_BAG_L{level}"
                    name = str(item["model"])
                    if not name.endswith(suffix):
                        raise ValueError("AutoGluon node name has an unsupported bagging suffix: " + name)
                    parameters = {**bag["child_hyperparameters"], **bag["child_hyperparameters_fit"],
                        "ag_args": {"name": name[:-len(suffix)]},
                        "ag_args_fit": {"early_stop": None}}
                    configurations.setdefault(level, {}).setdefault(learner, []).append(parameters)
                    graph[name] = list(info.get("stacker_info", {}).get("base_model_names", []))
            self.profile_configs = getattr(self, "profile_configs", {})
            self.profile_configs[target] = {"hyperparameters": configurations, "weights": weights, "selected_model": best, "graph": graph}
            self.models[target] = ("autogluon", predictor)
            self.selected[target] = {"model": best, "path": str(path), "search_owner": "autogluon",
                "bagging": True, "stacking": self.spec.autogluon_profile == "stacking", "profile": self.spec.autogluon_profile,
                'selected_stack_levels':max(configurations)}
            self._refit_profile(target, X, y, context)
            elapsed_refit = time.monotonic() - refit_started
            self.budget_evidence["targets"][target]["selected_fit_actual_seconds"] = elapsed_refit
            self.budget_evidence["selected_fit_rows"] = len(X)
            self.training_keys = list(map(str, X.index))
            return
        # Score the selected recipe after it has seen every permitted fit row.
        # Iterations are already fixed by the engine, so merging the tuning rows
        # is a deterministic refit, not a second search. refit_full already
        # concatenates the saved inner validation split, so no train_data_extra
        # is passed here (the pinned engine would silently drop it anyway).
        mapping = predictor.refit_full(model=best,
            set_best_to_refit_full=True, num_cpus=context["cpu"], num_gpus=0, fit_strategy="sequential")
        refit_seconds = time.monotonic() - refit_started
        if not mapping or best not in mapping:
            raise ValueError("AutoGluon did not produce the selected full-data refit")
        information = predictor.info()["model_info"][best]
        learner = {"LGBModel": "GBM", "CatBoostModel": "CAT", "XGBoostModel": "XGB"}.get(information.get("model_type"))
        if learner is None:
            raise ValueError("AutoGluon selected model has no verified frozen learner")
        frozen_parameters = {**information.get("hyperparameters", {}), **information.get("hyperparameters_fit", {})}
        self.budget_evidence["targets"][target]["selected_fit_actual_seconds"] = refit_seconds
        self.budget_evidence["targets"][target]["inner_holdout_refit_seconds"] = refit_seconds
        self.budget_evidence["selected_fit_rows"] = len(X)
        self.training_keys = list(map(str, X.index))
        self.selected[target] = {"model": best, "path": str(path), "search_owner": "autogluon", "bagging": bagging,
            "stacking": self.spec.autogluon_profile == "stacking", "profile": self.spec.autogluon_profile,
            "refitted_model": mapping[best], "learner": learner, "model_type": information.get("model_type"),
            "parameters": frozen_parameters}
        self.models[target] = ("autogluon", predictor)
        self.selected[target]["refit_policy"] = "frozen_selected_configuration_refit_on_all_permitted_rows"

    def _bounded_group_codes(self, data):
        from sklearn.model_selection import GroupKFold
        column = getattr(self, "autogluon_group_column", None) or self.task.group_column
        groups = data[column].astype(str)
        count = min(self.spec.bag_folds, groups.nunique())
        if count < 2:
            raise ValueError("AutoGluon group bagging needs at least two training groups")
        result = data.copy()
        codes = np.zeros(len(data), dtype=int)
        for number, (_, heldout) in enumerate(GroupKFold(count).split(data, groups=groups)):
            codes[heldout] = number
        result[column] = codes
        return result

    def _autogluon_frame(self, X, y, target, context):
        data = X[self.raw_feature_columns].copy()
        column = getattr(self, "autogluon_group_column", None)
        if column and context["strategy"] == "exact_group":
            from .identity import feature_groups
            data[column] = feature_groups(X[self.task.feature_columns])
        data[target] = y[target]
        return data

    def _refit_profile(self, target, X, y, context):
        from autogluon.tabular import TabularPredictor
        self.profile_refit_serial = getattr(self, "profile_refit_serial", 0) + 1
        path = Path(context["workdir"]) / f"autogluon_{target}_frozen_{self.profile_refit_serial}"
        configuration = self.profile_configs[target]
        data = self._autogluon_frame(X, y, target, context)
        if getattr(self, "autogluon_group_column", None):
            data = self._bounded_group_codes(data)
        problem = "binary" if self.task.classes and len(self.task.classes[target]) == 2 else "multiclass" if self.task.classes else "regression"
        kwargs = {"groups": self.autogluon_group_column} if getattr(self, "autogluon_group_column", None) else {}
        if problem == "binary" and self.task.metric.positive_label is not None:
            kwargs["positive_class"] = self.task.metric.positive_label
        predictor = TabularPredictor(label=target, path=str(path), problem_type=problem, eval_metric=self._ag_metric(),
            verbosity=0, log_to_file=False, learner_kwargs={"random_state": self.spec.seed}, **kwargs)
        predictor.fit(data, hyperparameters=configuration["hyperparameters"],
            num_bag_folds=self.spec.bag_folds, num_stack_levels=max(configuration["hyperparameters"])-1,
            fit_weighted_ensemble=False, dynamic_stacking=False, calibrate_decision_threshold=False,
            num_cpus=context["cpu"], num_gpus=0, fit_strategy="sequential",
            time_limit=max(1, context["budget_seconds"]), memory_limit=max(.1, context["memory_mb"]/1024*.7),
            ag_args_fit={"num_cpus": context["cpu"], "num_gpus": 0},
                ag_args_ensemble={"fold_fitting_strategy": "sequential_local", "max_base_models": 0, "max_base_models_per_type": 0})
        parents = list(configuration["weights"])
        if not set(parents) <= set(predictor.model_names()):
            raise ValueError("Frozen AutoGluon profile graph did not complete")
        for name, dependencies in configuration.get("graph", {}).items():
            replayed = predictor.info()["model_info"].get(name, {})
            if sorted(replayed.get("stacker_info", {}).get("base_model_names", [])) != sorted(dependencies):
                raise ValueError("Frozen AutoGluon stacking dependencies changed: " + name)
        mapping = predictor.refit_full(model=parents, num_cpus=context["cpu"], num_gpus=0, fit_strategy="sequential")
        if not set(parents) <= set(mapping):
            raise ValueError("Frozen AutoGluon profile full refit is incomplete")
        self.selected[target].update(path=str(path),
            refitted_weights={mapping[name]:weight for name,weight in configuration["weights"].items()},
            full_refit_rows=len(X), refit_policy="frozen_architecture_parameters_weights_rebuilt_oof_all_rows")
        self.models[target] = ("autogluon", predictor)

    def _refit_selection(self, target, X, y, context):
        """Refit the selected model template without a new validation search."""
        frozen = self.selected[target]
        loaded = self.models[target][1]
        selected = frozen["model"]
        if selected not in loaded.model_names():
            # v090 replaced the original predictor during refit. Its current
            # measured model still identifies the frozen parent to preserve.
            selected = loaded.model_refit_map(inverse=True).get(loaded.model_best, loaded.model_best)
        self.selection_refit_serial = getattr(self, "selection_refit_serial", 0) + 1
        path = Path(context["workdir"]) / f"autogluon_{target}_frozen_{self.selection_refit_serial}"
        predictor = loaded.clone(str(path), return_clone=True)
        predictor.set_model_best(selected, save_trainer=True)
        previous_full = [name for name in predictor.model_names() if name.endswith("_FULL")]
        if previous_full:
            predictor.delete_models(models_to_delete=previous_full, allow_delete_cascade=True, dry_run=False)
        features = predictor.transform_features(X[self.raw_feature_columns])
        labels = predictor.transform_labels(y[target])
        # In pinned AutoGluon 1.3.1 these forwarded inputs replace the cached
        # inner split. Empty validation overrides are intentional: None reloads
        # the old holdout. refit_full then trains the template's fixed rounds on
        # every supplied row without early stopping or ignored pseudo-rows.
        mapping = predictor.refit_full(model=selected, X=features, y=labels,
            X_val=features.iloc[:0], y_val=labels.iloc[:0], set_best_to_refit_full=True,
            num_cpus=context["cpu"], num_gpus=0, fit_strategy="sequential")
        if selected not in mapping:
            raise ValueError("AutoGluon frozen selection full refit did not complete")
        inner_rows = len(getattr(self, "autogluon_train_keys", {}).get(target, []))
        frozen.update(model=selected, path=str(path), refitted_model=mapping[selected], full_refit_rows=len(X),
            full_refit_added_rows=max(0, len(X) - inner_rows),
            refit_policy="frozen_selected_configuration_refit_on_all_permitted_rows")
        self.models[target] = ("autogluon", predictor)

    def _tune(self, X, y, context):
        from .tuning import tune
        return tune(self, X, y, context)

    def predict(self, X, context):
        if hasattr(self, "pipeline"):
            return self.pipeline.predict(X, context)
        predictions = {}
        for target, (engine, model) in self.models.items():
            if engine == "constant":
                predictions[target] = model.predict(X, context).values[target]
                continue
            if self.task.classes:
                if engine == "flaml":
                    raw = np.asarray(model.predict_proba(X.reset_index(drop=True)))
                    labels = model.classes_.tolist()
                else:
                    weights = self.selected[target].get("refitted_weights")
                    if weights:
                        frames = [(weight, model.predict_proba(X, model=name, as_pandas=True, as_multiclass=True)) for name, weight in weights.items()]
                        labels = list(frames[0][1])
                        if any(list(frame) != labels for _, frame in frames):
                            raise ValueError("Frozen AutoGluon class order differs across members")
                        raw = sum(weight * frame.to_numpy() for weight, frame in frames) / sum(weights.values())
                    else:
                        raw_frame = model.predict_proba(X, as_pandas=True, as_multiclass=True)
                        labels, raw = list(raw_frame), raw_frame.to_numpy()
                arr = np.zeros((len(X), len(self.task.classes[target])))
                for i, label in enumerate(labels):
                    arr[:, self.task.classes[target].index(label)] = raw[:, i]
                predictions[target] = arr
            else:
                weights = self.selected[target].get("refitted_weights")
                if weights:
                    predictions[target] = sum(weight * np.asarray(model.predict(X, model=name)).reshape(-1) for name, weight in weights.items()) / sum(weights.values())
                else:
                    predictions[target] = np.asarray(model.predict(X.reset_index(drop=True) if engine == "flaml" else X)).reshape(-1)
        return prediction_bundle(predictions, X, context, "probabilities" if self.task.classes else "values")

    def refit(self, X, y, context):
        if self.spec.engine == "optuna":
            self.pipeline = DirectPipeline(self.frozen, self.task)
            return self.pipeline.fit(X, y, context)
        started = time.monotonic()
        for target, (engine, old) in list(self.models.items()):
            if engine == "constant":
                old.fit(X, y[[target]], context)
                continue
            if engine == "flaml":
                from flaml import AutoML
                log = Path(context["workdir"]) / f"flaml_{target}.jsonl"
                log.write_text(self.log_content[target])
                model = AutoML()
                model.retrain_from_log(str(log), X_train=X[self.task.feature_columns].reset_index(drop=True), y_train=y[target].reset_index(drop=True),
                    task="classification" if self.task.classes else "regression", train_full=True,
                    record_id=self.selected[target]["record_id"], n_jobs=context["cpu"], auto_augment=False,
                    eval_method="holdout", split_type="time" if context["strategy"] == "temporal" else "group" if context["strategy"] == "group" else "auto",
                    groups=X[self.task.group_column].to_numpy() if context["strategy"] == "group" else None)
            elif self.spec.autogluon_profile != "selection":
                self._refit_profile(target, X, y, context)
                continue
            else:
                self._refit_selection(target, X, y, context)
                continue
            self.models[target] = (engine, model)
        self.training_keys = list(map(str, X.index))
        return TrainingSummary(rows=len(X), seconds=time.monotonic() - started, selected_parameters=self.selected,
            inner_scores={"scope": "frozen_refit_no_search"}, imputation={"scope": "allowed_rows_only"})

    def frozen_spec(self):
        return self.frozen if hasattr(self, "frozen") else self.spec
