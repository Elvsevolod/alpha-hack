"""Fit-only fold identities and feature-only transfer diagnostics."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold

from .util import digest
from .validation import ordered_periods
from .identity import feature_groups, artifact_scope


def stability_design(strategy):
    return {"split_strategy": strategy or "unknown", "varied_axis": "rows_not_features",
        "temporal_order_required": strategy in {"temporal", "temporal_group"} if strategy else None,
        "group_basis": "declared_group" if strategy in {"group", "temporal_group"}
            else "exact_feature_duplicates_not_entities" if strategy == "random" else None}


def folds(X, y, task, strategy, seed=42, count=3, min_folds=2):
    if strategy in {"temporal", "temporal_group"}:
        periods = ordered_periods(X[task.time_column])
        unique = sorted(periods.unique())
        result = []
        horizon = task.horizon if task.kind == "forecasting" else 1
        starts = list(range(max(1, len(unique) - count * horizon), len(unique) - horizon + 1, horizon))
        for start in starts:
            b = np.flatnonzero(periods.isin(unique[start:start + horizon]))
            a = np.flatnonzero(periods < unique[start])
            if strategy == "temporal_group":
                held = set(X.iloc[b][task.group_column])
                a = np.asarray([i for i in a if X.iloc[i][task.group_column] not in held], dtype=int)
            if len(a) and len(b):
                result.append((a, b))
    elif strategy in {"group", "exact_group"}:
        groups = X[task.group_column] if strategy == "group" else pd.Series(feature_groups(X[task.feature_columns]), index=X.index)
        n = min(count, groups.nunique())
        result = list(GroupKFold(n).split(X, groups=groups)) if n >= 2 else []
    else:
        n = min(count, len(X))
        if task.kind == "classification" and y.iloc[:, 0].value_counts().min() >= n:
            result = list(StratifiedKFold(n, shuffle=True, random_state=seed).split(X, y.iloc[:, 0]))
        else:
            result = list(KFold(n, shuffle=True, random_state=seed).split(X)) if n >= 2 else []
    if len(result) < min_folds:
        raise ValueError(f"At least {min_folds} nonempty split-aware folds are required")
    for a, b in result:
        if set(a) & set(b):
            raise ValueError("OOF training and evaluation rows overlap")
    return result


def oof_plan(spec, task, X, y, strategy, seed=42):
    """Declare fold support and label-free warmup schedules before training."""
    temporal = strategy in {"temporal", "temporal_group"}
    parts = folds(X, y, task, strategy, seed, min_folds=1 if temporal else 2)
    result, skipped = [], []
    for a, b in parts:
        fresh = spec.model_copy(update={"class_offsets": None, "learned_decision": {}, "learned_inverse": {},
            "frozen_iterations": None, "iteration_schedule": {}, "iteration_scale_rows": None, "fit_diagnostics": "none"})
        # Consumers learn their own decision on these probabilities. Individual
        # label offsets need no nested fit and never alter raw OOF probabilities.
        if fresh.prediction_policy == "class_offsets":
            fresh = fresh.model_copy(update={"prediction_policy": "default", "postprocess_fit_policy": "none"})
        warmup = None
        if temporal:
            from .validation import inner_split, pseudo_label_split, pilot_population
            if fresh.shift_strategy == "pseudo_label":
                try:
                    teacher, _, _ = pseudo_label_split(X.iloc[a], y.iloc[a], task, strategy, spec.seed)
                    base = type(fresh).model_validate({**fresh.research_recipe(), "shift_strategy": "none"})
                    pilot_population(X.iloc[a].iloc[teacher], y.iloc[a].iloc[teacher], task,
                        strategy, len(teacher), spec.seed, base)
                except ValueError as error:
                    skipped.append({"fit_rows": len(a), "evaluation_rows": len(b), "reason": str(error)})
                    continue
            try:
                inner_split(X.iloc[a], y.iloc[a], task, strategy, spec.seed)
            except ValueError as error:
                needs_inner = (fresh.engine != "direct" or fresh.prediction_policy != "default"
                    or fresh.log_inverse == "oof_smearing" or fresh.checkpoint_counts
                    or fresh.early_stopping_rounds and any(op.name == 'target_encode' for op in fresh.transformations))
                # Target encoding itself can use a neutral first period. Keep
                # the existing label-free stopping warmup for plain recipes.
                if needs_inner:
                    skipped.append({"fit_rows": len(a), "evaluation_rows": len(b), "reason": str(error)})
                    continue
                fresh = fresh.model_copy(update={"early_stopping_rounds": None, "checkpoint_counts": []})
                warmup = "declared_fixed_iteration_cap; no evaluation labels used for stopping"
        result.append((a, b, fresh, warmup))
    if not result:
        raise ValueError("insufficient_oof_support: no fold can fit this complete recipe inside its own training history")
    return result, skipped


def oof_predictions(spec, task, X, y, context):
    from .pipelines import build_pipeline
    from .evaluation import align_predictions, score_predictions
    started = time.monotonic()
    parts, skipped = oof_plan(spec, task, X, y, context["strategy"], context.get("oof_seed", 42))
    if len(parts) < context.get('minimum_oof_folds', 1):
        raise ValueError('insufficient_oof_support: at least two independently fitted folds are required for learned postprocessing')
    predictions = {target: {} for target in task.targets}
    identities, scores = [], []
    for index, (a, b, fresh, warmup) in enumerate(parts):
        remaining = context["budget_seconds"] - (time.monotonic() - started)
        if remaining <= .2:
            raise TimeoutError("OOF evaluation exhausted its allocated budget")
        local = {**context, "budget_seconds": remaining / (len(parts) - index),
            "fit_row_keys": X.iloc[a].index.tolist(), "split_id": f"oof_{index}"}
        for field in ("schedule_selection", "schedule_validation", "optuna_trial", "rolling_folds"):
            local.pop(field, None)
        if context.get("sample_weights") is not None:
            raise ValueError("Strict OOF requires independently rebuilt domain weights; parent-fitted weights cannot be inherited")
        # Every fold selects its iterations/postprocessing inside its own fit.
        model = build_pipeline(fresh, task, local)
        model.fit(X.iloc[a].copy(), y.iloc[a].copy(), local)
        bundle = model.predict(X.iloc[b].copy(), local)
        values = align_predictions(bundle, X.iloc[b].index.tolist(), task, context["protocol_hash"], local["split_id"])
        probabilities = getattr(bundle, "calibrated_probabilities", {}) or getattr(bundle, "raw_probabilities", {})
        if task.classes:
            if bundle.classes != task.classes or not probabilities and bundle.prediction_type != "probabilities":
                raise ValueError("OOF composition requires probabilities and an explicit complete class order")
            if probabilities:
                probability_bundle = bundle.model_copy(update={"values": probabilities, "prediction_type": "probabilities"})
                probabilities = align_predictions(probability_bundle, X.iloc[b].index.tolist(), task, context["protocol_hash"], local["split_id"])
        for target in task.targets:
            source = np.asarray(probabilities.get(target, values[target]))
            for key, value in zip(X.iloc[b].index, source):
                predictions[target][str(key)] = value.tolist() if np.ndim(value) else value.item()
        groups = X.iloc[b][task.group_column].to_numpy() if task.kind == "ranking" else None
        scores.append(score_predictions(task, y.iloc[b], values, bundle.prediction_type, groups))
        identities.append({"fold": index, "fit_keys": X.iloc[a].index.tolist(), "evaluation_keys": X.iloc[b].index.tolist(),
            "warmup_policy": warmup, "fresh_recipe_hash": fresh.hash,
            "scope": artifact_scope(local, fresh, task, X.iloc[a].index, X.iloc[b].index)})
    keys = [str(key) for key in X.index if all(str(key) in predictions[target] for target in task.targets)]
    return {"row_keys": keys, "values": {target: [predictions[target][key] for key in keys] for target in task.targets},
        "classes": task.classes, "prediction_type": "probabilities" if task.classes else "values",
        "folds": identities, "folds_hash": digest([{k: part[k] for k in ("fold", "fit_keys", "evaluation_keys")} for part in identities]), "coverage": len(keys) / len(X),
        "uncovered_reason": "earliest temporal rows lack an earlier training population" if len(keys) < len(X) else None,
        "scores": scores, "seconds": time.monotonic() - started, "scope": "fit_only_out_of_fold",
        "support_status": "limited_temporal_history" if len(parts) < 2 or skipped else "multiple_folds",
        "skipped_folds": skipped, "individual_label_offsets_applied": False,
        "prediction_role": "strict_oof", "selection_requires_outer_evaluation": True}


def transfer_profile(fit, y, validation, train, test, task):
    fit, validation, train, test = [frame[task.feature_columns] for frame in (fit, validation, train, test)]
    fit_groups, full_groups = feature_groups(fit), feature_groups(train)
    seen = {"validation_vs_fit": np.isin(feature_groups(validation), fit_groups),
        "test_vs_train": np.isin(feature_groups(test), full_groups)}
    conflicts = {}
    for target in y:
        frame = pd.DataFrame({"group": fit_groups, "target": y[target].to_numpy()})
        groups = frame.groupby("group")["target"].nunique()
        mask = frame["group"].isin(groups[groups > 1].index)
        conflicts[target] = {"groups": int((groups > 1).sum()), "rows": int(mask.sum()),
            "by_class": {str(label): int((mask & (frame["target"] == label)).sum()) for label in task.classes.get(target, [])}}
    differences = {}
    for column in task.feature_columns:
        a, b = train[column], test[column]
        row = {"train_missing_fraction": float(a.isna().mean()), "test_missing_fraction": float(b.isna().mean())}
        if pd.api.types.is_numeric_dtype(a) and column not in task.categorical_columns:
            av, bv = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
            row.update(train_quantiles={str(q): float(av.quantile(q)) if av.notna().any() else None for q in (.1, .5, .9)},
                test_quantiles={str(q): float(bv.quantile(q)) if bv.notna().any() else None for q in (.1, .5, .9)})
        else:
            known = set(a.dropna().astype(str))
            row["test_unseen_category_fraction"] = float((b.notna() & ~b.astype(str).isin(known)).mean())
        differences[column] = row
    return {"scope": "features_only_except_fit_label_conflicts", "exact_feature_groups_are_not_entities": True,
        "overlap": {name: {"rows": len(mask), "matched": int(mask.sum()), "fraction": float(mask.mean()) if len(mask) else 0.}
            for name, mask in seen.items()}, "fit_conflicts": conflicts, "features": differences,
        "target_quality_inferred_from_shift": False}


def domain_diagnostic(train, test, cpu=1, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OrdinalEncoder
    X = pd.concat([train, test], ignore_index=True)
    source = np.r_[np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]
    groups = feature_groups(X)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed).split(X, source, groups))
    if len(np.unique(source[a])) < 2 or len(np.unique(source[b])) < 2:
        raise ValueError("Feature groups do not support a two-domain held-out evaluation")
    numeric = [column for column in X if pd.api.types.is_numeric_dtype(X[column])]
    categorical = [column for column in X if column not in numeric]
    prepared = X.copy()
    prepared[categorical] = prepared[categorical].astype("string").fillna("__missing__").astype(str)
    preprocessor = ColumnTransformer([("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical)])
    model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=80, max_depth=8,
        min_samples_leaf=10, n_jobs=cpu, random_state=seed))
    model.fit(prepared.iloc[a], source[a])
    probabilities = model.predict_proba(prepared.iloc[b])[:, 1]
    importance = model[-1].feature_importances_
    return {"scope": "unlabelled_train_vs_test_features", "auc": float(roc_auc_score(source[b], probabilities)),
        "fit_rows": len(a), "evaluation_rows": len(b), "exact_groups_disjoint": not bool(set(groups[a]) & set(groups[b])),
        "feature_importance": sorted([{"column": column, "importance": float(value)}
            for column, value in zip(numeric + categorical, importance)], key=lambda row: -row["importance"]),
        "task_labels_used": False, "target_quality_inferred": False, "warning": "P(X) separation does not establish harmful P(y|X) shift"}
