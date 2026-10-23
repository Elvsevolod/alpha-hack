from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .api import prediction_bundle
from .contracts import PipelineSpec, TrainingSummary
from .evaluation import align_predictions, score_predictions
from .pipelines import DirectPipeline
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
                values[name] = trial.suggest_float(name, rule["low"], rule["high"], log=rule.get("log", False))
            else:
                raise ValueError("Unknown search-space parameter type")
        if not changed:
            raise ValueError("Cyclic conditional search space")
    return values


class SearchPipeline:
    def __init__(self, spec, task):
        self.spec, self.task = spec, task
        self.models, self.training_keys = {}, []

    def fit(self, X, y, context):
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
        self.raw_feature_columns = list(X)
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
            groups=X[self.task.group_column].to_numpy() if context["strategy"] == "group" else None)
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
        train = X.copy()
        train[target] = y[target]
        path = Path(context["workdir"]) / f"autogluon_{target}"
        problem = "binary" if self.task.classes and len(self.task.classes[target]) == 2 else "multiclass" if self.task.classes else "regression"
        kwargs = {"positive_class": self.task.metric.positive_label} if problem == "binary" and self.task.metric.positive_label is not None else {}
        predictor = TabularPredictor(label=target, path=str(path), problem_type=problem, eval_metric=self._ag_metric(),
            verbosity=0, log_to_file=False, learner_kwargs={"random_state": self.spec.seed}, **kwargs)
        seed_parameters = {"GBM": {"seed_value": self.spec.seed}, "CAT": {"random_seed": self.spec.seed}, "XGB": {"random_state": self.spec.seed}}
        from .diagnostic_automl import autogluon_trials
        with autogluon_trials(context):
            predictor.fit(train_data=train.iloc[fit], tuning_data=train.iloc[val], time_limit=allocation,
                hyperparameters={learner: seed_parameters[learner] for learner in learners}, num_bag_folds=0, num_stack_levels=0,
                fit_weighted_ensemble=False, fit_strategy="sequential", num_cpus=context["cpu"], num_gpus=0,
                dynamic_stacking=False, calibrate_decision_threshold=False,
                ag_args_fit={"num_cpus": context["cpu"], "num_gpus": 0})
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
        mapping = predictor.refit_full(model=best, train_data_extra=train.iloc[val].copy(),
            set_best_to_refit_full=True, num_cpus=context["cpu"], num_gpus=0, fit_strategy="sequential")
        refit_seconds = time.monotonic() - refit_started
        if not mapping or best not in mapping:
            raise ValueError("AutoGluon did not produce the selected full-data refit")
        self.budget_evidence["targets"][target]["selected_fit_actual_seconds"] = refit_seconds
        self.budget_evidence["targets"][target]["inner_holdout_refit_seconds"] = refit_seconds
        self.budget_evidence["selected_fit_rows"] = len(X)
        self.selected[target] = {"model": best, "path": str(path), "search_owner": "autogluon", "bagging": False, "stacking": False,
            "refitted_model": mapping[best], "parameters": predictor.info()["model_info"][best].get("hyperparameters", {})}
        self.models[target] = ("autogluon", predictor)
        self.selected[target]["refit_policy"] = "fixed_original_model_on_train_tuning_and_permitted_extra_rows"

    def _tune(self, X, y, context):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if not self.spec.search_space:
            raise ValueError("Optuna requires a nonempty typed search space")
        if self.spec.model in {"flaml", "autogluon"}:
            raise ValueError("Nested search owners are forbidden")
        started = time.monotonic()
        fit, val = inner_split(X, y, self.task, context["strategy"], self.spec.seed)
        study = optuna.create_study(direction="maximize" if self.task.metric.maximize else "minimize",
            sampler=optuna.samplers.TPESampler(seed=self.spec.seed))
        trials = []
        estimates = context.get("phase_estimates", {})
        selected_fit = max(.2, estimates.get("selected_fit_seconds", 2))
        prediction = max(.1, estimates.get("prediction_seconds", 1))
        publication = max(.1, estimates.get("publication_seconds", 1))
        next_trial = selected_fit * len(fit) / max(1, len(X))
        stop_reason = None
        remaining = lambda: context["budget_seconds"] - (time.monotonic() - started)
        if remaining() < next_trial + selected_fit + prediction + publication:
            raise ValueError("Optuna first trial and selected fit exceed the phase reserve")

        def protect_selected_fit(study, trial):
            nonlocal stop_reason
            if remaining() < next_trial + selected_fit + prediction + publication:
                stop_reason = "selected_fit_reserve"
                study.stop()

        def objective(trial):
            nonlocal next_trial
            parameters = {**self.spec.hyperparameters, **suggest_parameters(trial, self.spec.search_space)}
            trial_started = time.monotonic()
            trial_event(context, 'trial_started', trial_id=f'optuna_{trial.number}', engine='optuna', parameters=parameters)
            try:
                value = evaluate_trial(trial, parameters)
                trial_event(context, 'trial_finished', trial_id=f'optuna_{trial.number}', engine='optuna',
                    status='completed', metric=value, seconds=time.monotonic()-trial_started)
                return value
            except BaseException:
                import traceback
                trial_event(context, 'trial_finished', trial_id=f'optuna_{trial.number}', engine='optuna',
                    status='failed', seconds=time.monotonic()-trial_started, traceback=traceback.format_exc())
                raise
            finally:
                next_trial = max(next_trial, (time.monotonic() - trial_started) * 1.25)

        def evaluate_trial(trial, parameters):
            nonlocal selected_fit
            direct = self.spec.model_copy(update={"engine": "direct", "search_owner": "none", "search_space": {}, "hyperparameters": parameters})
            pipeline = DirectPipeline(direct, self.task)
            fit_started = time.monotonic()
            pipeline.fit(X.iloc[fit], y.iloc[fit], context)
            selected_fit = max(selected_fit, (time.monotonic() - fit_started) * len(X) / max(1, len(fit)) * 1.25)
            inner_context = {**context, "task": self.task.model_dump(mode="json"), "split_id": "inner"}
            bundle = pipeline.predict(X.iloc[val], inner_context)
            predicted = align_predictions(bundle, X.index[val].tolist(), self.task, context["protocol_hash"], "inner")
            groups = X.iloc[val][self.task.group_column].to_numpy() if self.task.kind == "ranking" else None
            value = score_predictions(self.task, y.iloc[val], predicted, bundle.prediction_type, groups)
            trials.append({"number": trial.number, "parameters": parameters, "value": value, "seconds": time.monotonic() - started, "scope": "inner_fit_only"})
            atomic_json(Path(context["workdir"]) / "optuna_trials.json", trials)
            return value

        study.optimize(objective, n_trials=self.spec.n_trials, timeout=max(0.1, context["budget_seconds"] * 0.6), n_jobs=1,
            catch=(ValueError, RuntimeError), show_progress_bar=False, callbacks=[protect_selected_fit])
        completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            raise ValueError("Optuna completed no valid inner trial")
        winner = study.best_trial
        parameters = next(t["parameters"] for t in trials if t["number"] == winner.number)
        if remaining() < selected_fit + prediction + publication:
            raise ValueError("Optuna selected fit exceeds the remaining phase reserve after measured trials")
        self.frozen = self.spec.model_copy(update={"engine": "direct", "search_owner": "none", "search_space": {}, "hyperparameters": parameters})
        self.pipeline = DirectPipeline(self.frozen, self.task)
        summary = self.pipeline.fit(X, y, context)
        self.frozen = self.pipeline.frozen_spec()
        summary.budget = {"selected_fit_actual_seconds": summary.seconds,
            "selected_fit_rows": len(X), "search_seconds": time.monotonic() - started - summary.seconds, "owner": "optuna",
            "phase_reserves": {"selected_fit_seconds": selected_fit, "prediction_seconds": prediction,
                "publication_seconds": publication}, "next_trial_estimate_seconds": next_trial,
            "stop_reason": stop_reason or "trial_limit_or_search_timeout"}
        summary.inner_scores = {"trials": trials, "best": float(winner.value), "scope": "inner_fit_only"}
        summary.seconds = time.monotonic() - started
        self.summary = summary
        return summary

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
                    raw_frame = model.predict_proba(X, as_pandas=True, as_multiclass=True)
                    labels, raw = list(raw_frame), raw_frame.to_numpy()
                arr = np.zeros((len(X), len(self.task.classes[target])))
                for i, label in enumerate(labels):
                    arr[:, self.task.classes[target].index(label)] = raw[:, i]
                predictions[target] = arr
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
                model.retrain_from_log(str(log), X_train=X.reset_index(drop=True), y_train=y[target].reset_index(drop=True),
                    task="classification" if self.task.classes else "regression", train_full=True,
                    record_id=self.selected[target]["record_id"], n_jobs=context["cpu"], auto_augment=False,
                    eval_method="holdout", split_type="time" if context["strategy"] == "temporal" else "group" if context["strategy"] == "group" else "auto",
                    groups=X[self.task.group_column].to_numpy() if context["strategy"] == "group" else None)
            else:
                from autogluon.tabular import TabularPredictor
                original_path = self.selected[target]["path"]
                copied = Path(context["workdir"]) / f"autogluon_{target}"
                if Path(original_path).resolve() != copied.resolve():
                    shutil.copytree(original_path, copied, dirs_exist_ok=True)
                model = TabularPredictor.load(str(copied), verbosity=0)
                extra = X.loc[~X.index.isin(self.training_keys)].copy()
                extra[target] = y.loc[extra.index, target]
                best = self.selected[target]["model"]
                mapping = model.refit_full(model=best, train_data_extra=extra if len(extra) else None,
                    set_best_to_refit_full=True, num_cpus=context["cpu"], num_gpus=0, fit_strategy="sequential")
                self.selected[target]["path"] = str(copied)
                self.selected[target]["refitted_model"] = mapping[best]
            self.models[target] = (engine, model)
        self.training_keys = list(map(str, X.index))
        return TrainingSummary(rows=len(X), seconds=time.monotonic() - started, selected_parameters=self.selected,
            inner_scores={"scope": "frozen_refit_no_search"}, imputation={"scope": "allowed_rows_only"})

    def frozen_spec(self):
        return self.frozen if hasattr(self, "frozen") else self.spec
