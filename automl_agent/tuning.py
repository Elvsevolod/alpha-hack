"""Bounded task-local HPO with durable trial reconciliation and honest samplers."""
import time
from pathlib import Path

import numpy as np

from .evaluation import align_predictions, score_predictions
from .diagnostics import trial_event
from .util import atomic_json, digest
from .validation import inner_split


def tune(owner, X, y, context):
    import optuna
    from .search import suggest_parameters
    from .pipelines import DirectPipeline, effective_parameters
    from .training_schedule import ITERATION_PARAMETER
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spec, task = owner.spec, owner.task
    if not spec.search_space:
        raise ValueError("Optuna requires a nonempty typed search space")
    parts = context.get("rolling_folds") or [dict(zip(("fit", "validation"), inner_split(X, y, task, context["strategy"], spec.seed)))]
    if not context.get("rolling_folds") and not task.classes and context["strategy"] in {"temporal", "temporal_group"}:
        from .research_evaluation import folds
        transitions = folds(X, y, task, context["strategy"], spec.seed, count=2, min_folds=1)
        supported = []
        for a, b in transitions:
            if (spec.prediction_policy != "default" or spec.checkpoint_counts or spec.log_inverse == "oof_smearing"
                    or any(op.name == 'target_encode' for op in spec.transformations)):
                try:
                    inner_split(X.iloc[a], y.iloc[a], task, context["strategy"], spec.seed)
                except ValueError:
                    continue
            supported.append({"fit": a, "validation": b})
        if not supported:
            raise ValueError("Insufficient temporal history for the complete HPO recipe")
        parts = supported
    parts = [{name: np.asarray(indices, dtype=int) for name, indices in part.items()} for part in parts]
    recipe = spec.research_recipe()
    for field in ("n_trials", "continue_study", "hpo_sampler"):
        recipe.pop(field, None)
    identity = digest({"task": task.hash, "data": context.get("data_hash"), "rows": list(X.index),
        "folds": [{k: X.index[v].tolist() for k, v in part.items()} for part in parts], "metric": task.metric.model_dump(),
        "protocol": context["protocol_hash"], "recipe": recipe, "versions": context.get("library_versions", {})})
    history = list(context.get("hpo_histories", {}).get(identity, [])) if spec.continue_study else []
    history = [{**item, "state": "outcome_unknown" if item["state"] == "pending" else item["state"]} for item in history]
    completed = [item for item in history if item["state"] == "completed"]
    total = spec.n_trials or 8
    sampler_name = ("tpe" if total >= 4 else "random") if spec.hpo_sampler == "auto" else spec.hpo_sampler
    startup = max(2, min(4, total // 3))
    sampler_seed = spec.seed + len(history)
    sampler = optuna.samplers.RandomSampler(seed=sampler_seed) if sampler_name == "random" else optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=startup)
    callbacks = bool(spec.early_stopping_rounds and spec.model in ITERATION_PARAMETER and len(task.targets) == 1 and len(parts) == 1
        and spec.prediction_policy == "default" and spec.log_inverse != "oof_smearing")
    study = optuna.create_study(direction="maximize" if task.metric.maximize else "minimize", sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=20) if callbacks else optuna.pruners.NopPruner())
    for item in completed:
        study.add_trial(optuna.trial.create_trial(value=item["value"], params=item["params"],
            distributions={key: optuna.distributions.json_to_distribution(value) for key, value in item["distributions"].items()},
            user_attrs={"record_id": item["id"]}))
    base_parameters = effective_parameters(spec)
    incumbent = {}
    def allowed_incumbent():
        pending = dict(spec.search_space)
        while pending:
            ready = [name for name, rule in pending.items() if not rule.get("when") or rule["when"]["parameter"] not in pending]
            if not ready:
                return False
            name = ready[0]
            rule = pending.pop(name)
            condition = rule.get("when")
            if condition and incumbent.get(condition["parameter"]) not in condition["in"]:
                continue
            if name not in base_parameters:
                return False
            value = incumbent[name] = base_parameters[name]
            if rule["type"] == "categorical":
                if value not in rule["choices"]:
                    return False
            elif (type(value) not in (int, float) or not np.isfinite(value)
                    or not rule["low"] <= value <= rule["high"]):
                return False
            elif rule.get("step") and not np.isclose((value - rule["low"]) / rule["step"], round((value - rule["low"]) / rule["step"])):
                return False
        return True

    if not history and allowed_incumbent():
        study.enqueue_trial(incumbent)
    directory = Path(context["workdir"])
    started = time.monotonic()
    estimates = context.get("phase_estimates", {})
    selected_fit = max(.2, estimates.get("selected_fit_seconds", 2))
    # Rolling's final OOF replay and full development fit both belong to this budget.
    fit_multiplier = 1 + sum(len(part["fit"]) / max(1, len(X)) for part in parts) if context.get("rolling_folds") else 1
    prediction = max(.1, estimates.get("prediction_seconds", 1))
    publication = max(.1, estimates.get("publication_seconds", 1))
    next_trial = selected_fit * sum(len(part["fit"]) for part in parts) / max(1, len(X))
    remaining = lambda: context["budget_seconds"] - (time.monotonic() - started)
    stop_reason = None
    duplicates = []
    seen = {item["parameters_hash"] for item in history}

    def save():
        atomic_json(directory / "hpo_history.json", {"identity": identity, "trials": history})
        atomic_json(directory / "optuna_trials.json", history)

    def direct_recipe(parameters):
        return spec.model_copy(update={"engine": "direct", "search_owner": "none",
            "search_space": {}, "hyperparameters": parameters})

    def evaluate(parameters, trial=None):
        nonlocal selected_fit
        scores, truth, predictions, schedules = [], [], {}, []
        for number, part in enumerate(parts):
            a, b = part["fit"], part["validation"]
            direct = direct_recipe(parameters)
            pipeline = DirectPipeline(direct, task)
            local = {**context, "split_id": f"inner_{number}", "rolling_folds": None,
                "fit_row_keys": X.index[a].tolist(), "schedule_selection": True,
                "schedule_validation": (X.iloc[b], y.iloc[b]), "optuna_trial": trial if callbacks else None}
            fit_started = time.monotonic()
            pipeline.fit(X.iloc[a], y.iloc[a], local)
            selected_fit = max(selected_fit, (time.monotonic() - fit_started) * len(X) / max(1, len(a)) * 1.25)
            bundle = pipeline.predict(X.iloc[b], local)
            values = align_predictions(bundle, X.index[b].tolist(), task, context["protocol_hash"], local["split_id"])
            truth.extend(b.tolist())
            for target in task.targets:
                predictions.setdefault(target, []).extend(np.asarray(values[target]).tolist())
            schedules.append(pipeline.frozen_spec().iteration_schedule)
            scores.append({"fit_rows": len(a), "evaluation_rows": len(b), "prediction_role": "search_validation"})
        actual = y.iloc[truth]
        groups = X.iloc[truth][task.group_column].to_numpy() if task.kind == "ranking" else None
        metric = score_predictions(task, actual, {k: np.asarray(v) for k, v in predictions.items()}, bundle.prediction_type, groups)
        schedule = {target: max(1, round(np.median([value[target] for value in schedules if target in value])))
            for target in task.targets if any(target in value for value in schedules)}
        return metric, schedule, {"row_keys": X.index[truth].tolist(), "values": predictions, "prediction_type": bundle.prediction_type}, scores

    def objective(trial):
        nonlocal next_trial
        suggested = suggest_parameters(trial, spec.search_space)
        # Normalize trial aliases separately so an explicit override has one name.
        parameters = {**{key: value for key, value in base_parameters.items() if key not in spec.search_space},
            **effective_parameters(spec.model_copy(update={"hyperparameters": suggested}))}
        key = digest(parameters)
        if key in seen:
            trial.set_user_attr("duplicate_skipped", True)
            duplicates.append({"trial": trial.number, "parameters_hash": key, "state": "skipped_already_measured_or_unknown"})
            raise optuna.TrialPruned("Already measured or unresolved configuration; no repeated fit")
        seen.add(key)
        record = {"id": digest({"identity": identity, "index": len(history)}), "number": trial.number,
            "state": "pending", "parameters": parameters, "parameters_hash": key, "params": trial.params,
            "distributions": {k: optuna.distributions.distribution_to_json(v) for k, v in trial.distributions.items()},
            "scope": "search_validation_inside_allowed_fit"}
        trial.set_user_attr("record_id", record["id"])
        history.append(record)
        save()
        trial_event(context, "trial_started", trial_id=f"optuna_{trial.number}", engine="optuna", parameters=parameters)
        fit_started = time.monotonic()
        try:
            # These checks inspect only the public recipe and task contract,
            # never X/y values. Preserve their reasons if a smoke search has no
            # valid trial; arbitrary backend exceptions stay private.
            from .compatibility import incompatibilities
            issues = incompatibilities(direct_recipe(parameters), task, context['strategy'])
            if issues:
                record['recipe_issues'] = issues[:8]
                raise ValueError('; '.join(issues))
            metric, schedule, _, boundaries = evaluate(parameters, trial)
            record.update(state="completed", value=metric, iteration_schedule=schedule, folds=boundaries)
            return metric
        except optuna.TrialPruned:
            record["state"] = "pruned"
            raise
        except BaseException as error:
            record["state"] = "failed"
            record['failure_type'] = type(error).__name__
            raise
        finally:
            record["seconds"] = time.monotonic() - fit_started
            next_trial = max(next_trial, record["seconds"] * 1.25)
            save()
            trial_event(context, "trial_finished", trial_id=f"optuna_{trial.number}", engine="optuna",
                status=record["state"], metric=record.get("value"), seconds=record["seconds"])

    def protect(study, trial):
        nonlocal stop_reason
        if remaining() < next_trial + selected_fit * fit_multiplier + prediction + publication:
            stop_reason = "selected_fit_reserve"
            study.stop()

    if remaining() < next_trial + selected_fit * fit_multiplier + prediction + publication:
        raise ValueError("Optuna first trial and selected fit exceed the phase reserve")
    errors = (ValueError, RuntimeError)
    if spec.model in {"lightgbm", "lgbm_ranker"}:
        from lightgbm.basic import LightGBMError
        errors += (LightGBMError,)
    elif spec.model == "catboost":
        from catboost import CatBoostError
        errors += (CatBoostError,)
    elif spec.model == "xgboost":
        from xgboost.core import XGBoostError
        errors += (XGBoostError,)
    study.optimize(objective, n_trials=spec.n_trials, timeout=max(.1, min(context["budget_seconds"] * .6,
        remaining() - selected_fit * fit_multiplier - prediction - publication)), n_jobs=1,
        catch=errors, show_progress_bar=False, callbacks=[protect])
    if not any(item["state"] == "completed" for item in history):
        from .util import canonical
        reasons = sorted({reason for item in history for reason in item.get('recipe_issues', [])})[:8]
        types = sorted({item['failure_type'] for item in history if item.get('failure_type')})
        raise ValueError('Optuna completed no valid inner trial; failure types=' + canonical(types)
            + '; public recipe constraints=' + canonical(reasons)[:1800])
    winner = next(item for item in history if item["id"] == study.best_trial.user_attrs["record_id"])
    if remaining() < selected_fit * fit_multiplier + prediction + publication:
        raise ValueError("Optuna selected fit exceeds its measured phase reserve")
    if context.get("rolling_folds"):
        _, _, owner.search_predictions, _ = evaluate(winner["parameters"])
    owner.frozen = spec.model_copy(update={"engine": "direct", "search_owner": "none", "search_space": {},
        "hyperparameters": winner["parameters"], "iteration_schedule": winner["iteration_schedule"]})
    owner.pipeline = DirectPipeline(owner.frozen, task)
    summary = owner.pipeline.fit(X, y, {**context, "rolling_folds": None, "optuna_trial": None})
    owner.frozen = owner.pipeline.frozen_spec()
    summary.budget = {**(summary.budget or {}), "search_seconds": time.monotonic() - started - summary.seconds, "owner": "optuna",
        "selected_fit_rows": len(X), "stop_reason": stop_reason or "trial_limit_or_search_timeout",
        "sampler": sampler_name, "n_startup_trials": startup if sampler_name == "tpe" else 0,
        "history_identity": identity, "continued_completed_trials": len(completed), "pruning_callbacks": callbacks}
    summary.budget['temporal_transitions'] = [{k:len(v) for k,v in part.items()} for part in parts] if context['strategy'] in {'temporal','temporal_group'} else []
    summary.budget['temporal_transfer_quality_proven'] = False
    summary.inner_scores = {"trials": history, "best": winner["value"], "scope": "search_validation_inside_fit",
        "unbiased_quality_claimed": False, "duplicate_proposals": duplicates}
    summary.seconds = time.monotonic() - started
    owner.summary = summary
    return summary
