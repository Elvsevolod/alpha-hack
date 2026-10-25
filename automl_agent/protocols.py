"""Opt-in transfer protocols; the original disjoint holdout remains intact."""
from collections import Counter

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from .contracts import ValidationPlan
from .identity import feature_groups
from .validation import ordered_periods


def configure(dataset, plan, mode, initial=False):
    if mode == "holdout":
        return ValidationPlan.model_validate({**plan.model_dump(), "evaluation_mode": mode, "development_folds": []})
    task, train = dataset.task, dataset.train
    pool = sorted(plan.fit + plan.validation)
    observations = list(plan.observations)
    if mode == "rolling":
        if plan.strategy != "temporal" or task.classes or not task.time_column:
            raise ValueError("Rolling mode currently supports chronological numeric tasks without entity exclusions")
        time = ordered_periods(train[task.time_column])
        periods = sorted(time.iloc[pool].unique())
        horizon = task.horizon if task.kind == "forecasting" else 1
        if len(periods) < 2 * horizon + 2:
            raise ValueError("Insufficient development history for two rolling transitions; retain chronological holdout")
        parts = []
        for start in (len(periods) - 2 * horizon, len(periods) - horizon):
            a = [i for i in pool if time.iloc[i] < periods[start]]
            b = [i for i in pool if time.iloc[i] in periods[start:start + horizon]]
            parts.append({"fit": a, "validation": b})
        b = sorted(i for part in parts for i in part["validation"])
        a = sorted(set(pool) - set(b))
        observations.append({"type": "rolling_history", "fold_rows": [{k: len(v) for k, v in part.items()} for part in parts],
            "development_rows": len(pool), "control_rows": len(plan.control), "prediction_role": "search_validation"})
        return ValidationPlan.model_validate({**plan.model_dump(), "evaluation_mode": mode, "fit": a, "validation": b,
            "development_folds": parts, "observations": observations})
    if task.kind != "classification" or plan.strategy != "random" or task.group_column or task.time_column:
        raise ValueError("Exact-X protocols cannot replace declared time/entity boundaries")
    control = list(plan.control)
    if initial:
        pool = sorted(pool + control)
        all_groups = feature_groups(train.iloc[pool][task.feature_columns])
        y_all = train.iloc[pool][task.targets[0]].to_numpy()
        options = []
        for a, b in GroupShuffleSplit(n_splits=24, test_size=.15, random_state=plan.seed + 2).split(train.iloc[pool], groups=all_groups):
            missing = sum(not np.any(y_all[a] == label) or not np.any(y_all[b] == label) for label in task.classes[task.targets[0]])
            options.append((missing, abs(len(b) / len(pool) - .15), a, b))
        _, _, development, held = min(options, key=lambda item: item[:2])
        control = [pool[i] for i in held]
        pool = [pool[i] for i in development]
    X = train.iloc[pool][task.feature_columns]
    groups = feature_groups(X)
    if len(np.unique(groups)) < 4:
        raise ValueError("Too few independent exact-X groups; retain the holdout and report uncertainty")
    target = task.targets[0]
    classes = task.classes[target]
    y = train.iloc[pool][target].to_numpy()
    alternatives = []
    for a, b in GroupShuffleSplit(n_splits=24, test_size=.2, random_state=plan.seed).split(X, groups=groups):
        missing = sum(not np.any(y[a] == label) for label in classes) + sum(not np.any(y[b] == label) for label in classes)
        alternatives.append((missing, abs(len(b) / len(pool) - .2), a, b))
    _, _, a, b = min(alternatives, key=lambda item: item[:2])
    stress = [pool[i] for i in b]
    desired = float(np.isin(feature_groups(dataset.test[task.feature_columns]), feature_groups(train[task.feature_columns])).mean())
    if mode == "low_overlap" and desired < .5:
        # Select using features/support only, never a candidate's metric.
        rng = np.random.default_rng(plan.seed + 1)
        shuffled = rng.permutation(a)
        move, remaining = [], set(a.tolist())
        group_counts = Counter(groups[a].tolist())
        label_counts = Counter(y[a].tolist())
        wanted = round(desired * len(b) / max(1e-12, 1 - desired))
        for i in shuffled:
            if len(move) >= wanted:
                break
            if group_counts[groups[i]] >= 2 and label_counts[y[i]] >= 2:
                move.append(int(i)); remaining.remove(int(i))
                group_counts[groups[i]] -= 1
                label_counts[y[i]] -= 1
        a, b = np.asarray(sorted(remaining)), np.asarray(sorted([*b, *move]))
    overlap = np.isin(groups[b], groups[a])
    if mode == "low_overlap" and initial and desired < .5:
        # Match the control transfer by the same feature-only policy as search.
        wanted = round(desired * len(control) / max(1e-12, 1 - desired))
        counts = Counter(groups[a].tolist())
        labels = Counter(y[a].tolist())
        move = []
        for i in np.random.default_rng(plan.seed + 3).permutation(a):
            if len(move) >= wanted:
                break
            if counts[groups[i]] >= 2 and labels[y[i]] >= 2:
                move.append(int(i)); counts[groups[i]] -= 1; labels[y[i]] -= 1
        control.extend(pool[i] for i in move)
        a = np.asarray([i for i in a if i not in set(move)], dtype=int)
        overlap = np.isin(groups[b], groups[a])
    full_groups = feature_groups(train[task.feature_columns])
    development_groups = set(full_groups[[pool[i] for i in np.r_[a, b]]])
    control_seen = float(np.mean([group in development_groups for group in full_groups[control]]))
    if not initial and ((mode == "exact_group" and control_seen > 0) or abs(control_seen - desired) > .2):
        raise ValueError("Closed control transfer differs from this protocol; choose it before baseline in a new run")
    observations.append({"type": "exact_X_protocol", "mode": mode, "groups_are_entities": False,
        "desired_feature_overlap": desired, "actual_feature_overlap": float(overlap.mean()),
        "stress_rows": stress, "stress_scope": "unseen_exact_X", "selection_uses_candidate_scores": False,
        "fit_support": {str(label): int(np.sum(y[a] == label)) for label in classes},
        "validation_support": {str(label): int(np.sum(y[b] == label)) for label in classes},
        "control_feature_overlap": control_seen, "all_boundaries_chosen_before_baseline": initial,
        "compromise": "support and intact unseen groups take precedence over matching test overlap"})
    return ValidationPlan.model_validate({**plan.model_dump(), "evaluation_mode": mode, "development_folds": [],
        "fit": [pool[i] for i in a], "validation": [pool[i] for i in b], "control": control, "observations": observations})


def initial_protocol(dataset, plan):
    """One recorded feature/contract decision, before fitting or opening control."""
    task, train = dataset.task, dataset.train
    mode, reason = "holdout", plan.reason
    evidence = {"uses_candidate_scores": False, "uses_control_feedback": False}
    if task.kind == "classification" and plan.strategy == "random" and not task.time_column and not task.group_column:
        groups = feature_groups(train[task.feature_columns])
        duplicate_fraction = 1 - len(np.unique(groups)) / len(groups)
        test_overlap = float(np.isin(feature_groups(dataset.test[task.feature_columns]), groups).mean())
        validation_overlap = float(np.isin(groups[plan.validation], groups[plan.fit]).mean())
        evidence.update(duplicate_fraction=duplicate_fraction, test_feature_overlap=test_overlap,
            initial_validation_feature_overlap=validation_overlap)
        if duplicate_fraction > 0 and len(np.unique(groups)) >= 6 and validation_overlap - test_overlap > .2 and test_overlap < .5:
            mode = "exact_group" if test_overlap <= .05 else "low_overlap"
            reason = "Exact-feature duplicates inflate row holdout overlap relative to test; align validation and closed control before training"
            plan = configure(dataset, plan, mode, initial=True)
    if plan.strategy in {"temporal", "temporal_group"}:
        periods = ordered_periods(train.iloc[plan.fit][task.time_column]).nunique()
        evidence.update(fit_periods=int(periods), horizon=task.horizon if task.kind == "forecasting" else 1,
            nested_history="limited; use declared fixed warmup schedules" if periods < 4 else "multiple transitions available",
            rolling_enabled=False, rolling_reason="A separate, explicitly evaluated hypothesis; no automatic change to chronology")
    decision = {"type": "initial_protocol_decision", "evaluation_mode": mode, "reason": reason, **evidence}
    from .feedback import freeze_partitions
    observations = [row for row in plan.observations if row.get('kind') != 'feedback_partition'
        and row.get('type') != 'class_coverage_gap']
    partitions = freeze_partitions(task, dataset.features(plan.fit), dataset.features(plan.validation),
        ordered_time=ordered_periods(train.iloc[plan.validation][task.time_column]) if plan.strategy in {"temporal", "temporal_group"} else None)
    coverage = {target: {name: {str(label): int((train.iloc[rows][target] == label).sum()) for label in classes}
        for name, rows in (("fit", plan.fit), ("validation", plan.validation), ("control", plan.control))} for target, classes in task.classes.items()}
    for target, parts in coverage.items():
        for name, counts in parts.items():
            missing = [label for label, count in counts.items() if not count]
            if missing:
                observations.append({'type':'class_coverage_gap', 'target':target, 'part':name, 'missing_classes':missing,
                    'boundaries_preserved':plan.strategy != 'random' or mode != 'holdout', 'metric':task.metric.name, 'targets_excluded':False})
    old_partitions = {row["report_id"] for row in plan.observations if "report_id" in row}
    reports = [name for name in plan.feedback_policy.report_ids if name not in old_partitions]
    reports = list(dict.fromkeys([*reports, *[row["report_id"] for row in partitions]]))
    class_order = sorted((f'{target}:{label}' for target in task.classes for label in task.classes[target]),
        key=lambda key: (next(coverage[target]['fit'][str(label)] for target in task.classes for label in task.classes[target]
            if f'{target}:{label}' == key), key))
    feedback = plan.feedback_policy.model_copy(update={"report_ids": reports, 'class_order':class_order})
    return ValidationPlan.model_validate({**plan.model_dump(), "reason": reason, "class_coverage": coverage,
        "feedback_policy": feedback.model_dump(), "observations": [*observations, *partitions, decision]})


def paired_comparison(task, actual, left, right, units, repeats=300, seed=42, prediction_type=None, max_seconds=2.):
    """Paired group bootstrap is descriptive; tiny support never implies certainty."""
    from .evaluation import score_predictions, confusion_details
    units = np.asarray(units)
    unique, inverse = np.unique(units, return_inverse=True)
    target = task.targets[0]
    a = {key: np.asarray(value) for key, value in left.items()} if isinstance(left, dict) else {target: np.asarray(left)}
    b = {key: np.asarray(value) for key, value in right.items()} if isinstance(right, dict) else {target: np.asarray(right)}
    kind = prediction_type or ("labels" if task.classes else "values")
    def delta(indices):
        x = actual.iloc[indices]
        return (score_predictions(task, x, {key: value[indices] for key, value in b.items()}, kind) -
            score_predictions(task, x, {key: value[indices] for key, value in a.items()}, kind)) * (1 if task.metric.maximize else -1)
    report = {"delta": delta(np.arange(len(actual))), "independent_units": len(unique),
        "min_delta_is_significance": False, "interval": None, "status": "uncertain",
        "selection_uses": "measured metric; interval is descriptive, never an automatic significance gate"}
    if task.classes and kind == "labels":
        report.update(left=confusion_details(actual[target], a[target], task.classes[target]),
            right=confusion_details(actual[target], b[target], task.classes[target]))
    support = all((actual[name] == label).sum() >= 10 for name, classes in task.classes.items() for label in classes)
    if len(unique) >= 20 and support and max_seconds > 0:
        import time
        started = time.monotonic()
        rng = np.random.default_rng(seed)
        order = np.argsort(inverse, kind='stable')
        grouped = np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1)
        samples = []
        uncomputable = 0
        for _ in range(repeats):
            if time.monotonic()-started >= max_seconds:
                break
            try:
                draw = rng.integers(0, len(unique), len(unique))
                indices = draw if len(unique) == len(actual) else np.concatenate([grouped[j] for j in draw])
                samples.append(delta(indices))
            except ValueError:
                uncomputable += 1
        report['completed_resamples'] = len(samples)
        report['uncomputable_resamples'] = uncomputable
        if len(samples) >= 30:
            report.update(interval=np.quantile(samples, [.025, .975]).tolist(), status="descriptive_group_bootstrap")
    return report


def saved_pair(c, left, right, part="validation", prepared=None):
    """Compare final outputs on the same already permitted rows and units."""
    from pathlib import Path
    from .contracts import PredictionBundle
    from .evaluation import align_predictions
    from .util import read_json
    if part == "control" and not c.journal.get("control_opened"):
        raise ValueError("Paired control diagnostics require frozen, opened control")
    if c.task.kind == "ranking":
        return {"status": "uncertain", "reason": "query-level intervals require a ranking-specific resampling contract"}
    rows = getattr(c.plan, part)
    predictions, kinds = [], []
    for identifier in (left, right):
        spec, result = c.lookup(identifier)
        if spec.validation_hash != c.plan.hash or not result or result.status != "completed":
            raise ValueError("Paired predictions require completed recipes on the same protocol")
        directory = Path(prepared[identifier]) if prepared is not None else Path(result.artifacts["directory"])
        bundle = PredictionBundle.model_validate(read_json(directory / f"{part}_predictions.json"))
        predictions.append(align_predictions(bundle, [c.dataset.row_keys[i] for i in rows], c.task, c.plan.hash, part))
        kinds.append(bundle.prediction_type)
    if kinds[0] != kinds[1]:
        return {"status": "uncertain", "reason": "final output types differ"}
    X = c.dataset.features(rows)
    units = (X[c.task.time_column].to_numpy() if c.plan.strategy in {"temporal", "temporal_group"}
        else X[c.task.group_column].to_numpy() if c.plan.strategy == "group" else feature_groups(X[c.task.feature_columns]))
    report = paired_comparison(c.task, c.dataset.train.iloc[rows][c.task.targets], *predictions, units, repeats=200,
        seed=c.plan.seed, prediction_type=kinds[0], max_seconds=min(2., max(0., c.budget.remaining-c.budget.final_reserve(c.finalist_costs()))))
    return {**report, "left_id": left, "right_id": right, "scope": part,
        "unit": "period" if c.plan.strategy in {"temporal", "temporal_group"} else "entity" if c.plan.strategy == "group" else "exact_features"}
