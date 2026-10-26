from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .contracts import FeedbackPolicy, SelectionPolicy, TaskSpec, ValidationPlan
from .feedback import freeze_partitions


def selection_for(task: TaskSpec) -> SelectionPolicy:
    # This is only floating-point equivalence. Practical significance and
    # uncertainty are reported separately from the measured metric ranking.
    return SelectionPolicy(maximize=task.metric.maximize, min_delta_abs=1e-12, min_delta_rel=1e-12)


def ordered_periods(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric
    dates = pd.to_datetime(series, errors="coerce", utc=True)
    if dates.notna().all():
        return dates
    raise ValueError("Period order cannot be established from numeric codes or dates")


def stratified_parts(y: pd.DataFrame, seed: int, multilabel: bool = False) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    n = len(y)
    result: list[list[int]] = [[], [], []]
    if multilabel or y.shape[1] > 1:
        # Separate target/class indicators, never the joint label combination.
        matrix = np.column_stack([(y[column] == label).to_numpy(dtype=float)
            for column in y for label in y[column].unique()])
        capacity = np.array([n - 2 * max(1, round(n * 0.15)), max(1, round(n * 0.15)), max(1, round(n * 0.15))])
        desired = np.outer(capacity / n, matrix.sum(axis=0))
        for j, count in enumerate(matrix.sum(axis=0)):
            if count >= 3 and count <= 6:
                desired[:, j] = [count - 2, 1, 1]
            elif count == 2:
                desired[:, j] = [1, 1, 0]
        remaining = capacity.copy()
        singleton_rows = np.flatnonzero(matrix[:, matrix.sum(axis=0) == 1].any(axis=1))
        for i in singleton_rows:
            result[0].append(int(i))
            remaining[0] -= 1
            desired[0] -= matrix[i]
        unassigned = np.ones(n, dtype=bool)
        unassigned[singleton_rows] = False
        while unassigned.any():
            counts = matrix[unassigned].sum(axis=0)
            possible = np.flatnonzero(counts > 0)
            if not len(possible):
                rows = np.flatnonzero(unassigned)
                label = None
            else:
                label = int(possible[np.argmin(counts[possible])])
                rows = np.flatnonzero(unassigned & (matrix[:, label] > 0))
            rng.shuffle(rows)
            for i in rows:
                valid = np.flatnonzero(remaining > 0)
                if not len(valid):
                    valid = np.array([0])
                deficits = desired[valid, label] if label is not None else remaining[valid]
                best = valid[deficits == deficits.max()]
                destination = int(best[np.argmax(remaining[best] / np.maximum(capacity[best], 1) + rng.random(len(best)) * 1e-9)])
                result[destination].append(int(i))
                remaining[destination] -= 1
                desired[destination] -= matrix[i]
                unassigned[i] = False
        # Coverage of fit is required even when competing target indicators
        # cannot all fit the ideal evaluation capacities.
        for j in range(matrix.shape[1]):
            if matrix[result[0], j].sum() == 0:
                for part in (1, 2):
                    candidates = [i for i in result[part] if matrix[i, j]]
                    if candidates:
                        row = candidates[0]
                        result[part].remove(row)
                        result[0].append(row)
                        break
        return tuple(sorted(rows) for rows in result)
    for _, indices in y.iloc[:, 0].groupby(y.iloc[:, 0], sort=False).groups.items():
        rows = np.asarray(list(indices), dtype=int)
        rng.shuffle(rows)
        nv = max(1, round(len(rows) * 0.15)) if len(rows) >= 2 else 0
        nc = max(1, round(len(rows) * 0.15)) if len(rows) >= 3 else 0
        while nv + nc >= len(rows):
            if nc:
                nc -= 1
            else:
                nv -= 1
        result[1].extend(rows[:nv].tolist())
        result[2].extend(rows[nv:nv + nc].tolist())
        result[0].extend(rows[nv + nc:].tolist())
    return tuple(sorted(x) for x in result)


def choose_validation(dataset, seed: int = 42) -> ValidationPlan:
    task, train, test = dataset.task, dataset.train, dataset.test
    n = len(train)
    if n < 8:
        raise ValueError("At least eight training observations are required for independent splits")
    strategy, reason = "random", "Independent-row task; fixed fit/validation/control split"
    observations: list[dict[str, Any]] = []
    if task.explicit_validation:
        requested = task.explicit_validation
        strategy, reason = requested, "Explicit validated task contract"
        if task.kind == "forecasting" and requested == "random":
            raise ValueError("Explicit random validation conflicts with required forecasting chronology")
        if task.kind == "forecasting" and requested == "group":
            strategy = "temporal_group"
            reason = "Explicit group validation strengthened to preserve forecasting chronology"
        elif task.group_column and requested == "random":
            strategy = "group"
            reason = "Explicit random validation split at the declared group level"
        elif task.group_column and requested == "temporal":
            strategy = "temporal_group"
            reason = "Explicit temporal validation strengthened to keep declared groups intact"
        if strategy != requested:
            observations.append({"type": "validation_boundary_strengthened", "requested": requested,
                "effective": strategy})
    elif task.time_column and task.group_column and (task.kind == "forecasting" or
            ordered_periods(test[task.time_column]).min() > ordered_periods(train[task.time_column]).max()):
        strategy, reason = "temporal_group", "Future periods and declared independent groups must both remain separated"
    elif task.kind == "forecasting":
        strategy, reason = "temporal", "Forecasting requires chronological horizons"
    elif task.group_column:
        strategy, reason = "group", "Declared groups must remain intact"
    elif task.time_column:
        tr, te = ordered_periods(train[task.time_column]), ordered_periods(test[task.time_column])
        if te.min() > tr.max():
            strategy, reason = "temporal", "Every test period follows the training history"
            observations.append({"column": task.time_column, "transfer": "next_period", "source": "train_test_features"})
    if strategy in {"temporal", "temporal_group"}:
        if not task.time_column:
            raise ValueError("Temporal validation requires an established time column")
        time = ordered_periods(train[task.time_column])
        periods = sorted(time.unique())
        horizon = task.horizon if task.kind == "forecasting" else 1
        if len(periods) < 2 * horizon + 1:
            raise ValueError("Not enough periods for fit and two independent horizons")
        control_periods = periods[-horizon:]
        validation_periods = periods[-2 * horizon:-horizon]
        fit = np.flatnonzero(time < validation_periods[0]).tolist()
        validation = np.flatnonzero(time.isin(validation_periods)).tolist()
        control = np.flatnonzero(time.isin(control_periods)).tolist()
        if strategy == "temporal_group":
            if not task.group_column or train[task.group_column].isna().any():
                raise ValueError("Joint temporal/group validation requires complete established groups")
            control_groups = set(train.iloc[control][task.group_column])
            validation = [i for i in validation if train.iloc[i][task.group_column] not in control_groups]
            held_groups = control_groups | set(train.iloc[validation][task.group_column])
            fit = [i for i in fit if train.iloc[i][task.group_column] not in held_groups]
            if not fit or not validation or not control:
                raise ValueError("No nonempty joint temporal/group split; never substitute a random split")
            observations.append({"type": "joint_boundary_exclusions", "rows": n - len(fit) - len(validation) - len(control),
                "policy": "later evaluation groups excluded from earlier partitions"})
    elif strategy == "group":
        if not task.group_column:
            raise ValueError("Group validation requires an established group column")
        groups = train[task.group_column]
        if groups.isna().any() or groups.nunique() < 3:
            raise ValueError("At least three complete groups are required")
        first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        remaining, control = next(first.split(train, groups=groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed + 1)
        a, b = next(second.split(train.iloc[remaining], groups=groups.iloc[remaining]))
        fit, validation, control = remaining[a].tolist(), remaining[b].tolist(), control.tolist()
    elif task.kind in {"classification", "multioutput_classification", "multilabel"}:
        fit, validation, control = stratified_parts(train[task.targets], seed, task.kind == "multilabel")
    else:
        order = np.random.default_rng(seed).permutation(n)
        edge = max(1, round(n * 0.15))
        control, validation, fit = order[:edge].tolist(), order[edge:2 * edge].tolist(), order[2 * edge:].tolist()
    coverage = {}
    if task.classes:
        for target, labels in task.classes.items():
            coverage[target] = {name: {str(label): int((train.iloc[rows][target] == label).sum()) for label in labels}
                for name, rows in (("fit", fit), ("validation", validation), ("control", control))}
        for target, parts in coverage.items():
            for part, counts in parts.items():
                missing = [label for label, count in counts.items() if count == 0]
                if missing:
                    observations.append({"type": "class_coverage_gap", "target": target, "part": part,
                        "missing_classes": missing, "boundaries_preserved": strategy != "random",
                        "metric": task.metric.name, "targets_excluded": False})
    # Only masks and classes fixed before any model sees validation feedback are registered.
    reports = ["missingness", "seen_unseen"] + (["classes"] if task.classes else [])
    if task.group_column:
        reports.append("entity_coverage")
    columns = sorted(set(task.feature_columns + ([task.time_column] if task.time_column else [])
        + ([task.group_column] if task.group_column else [])) - set(task.targets))
    partitions = freeze_partitions(task, train.iloc[fit][columns], train.iloc[validation][columns],
        ordered_time=time.iloc[validation] if strategy in {"temporal", "temporal_group"} else None)
    reports.extend(partition["report_id"] for partition in partitions)
    observations.extend(partitions)
    class_order = sorted((f"{target}:{label}" for target in task.classes for label in task.classes[target]),
        key=lambda identifier: (next(coverage[target]['fit'][str(label)] for target in task.classes
            for label in task.classes[target] if f"{target}:{label}" == identifier), identifier))
    if not task.classes and task.kind != "ranking":
        reports.append("regression_errors")
        observations.append({"type": "fit_target_tail_thresholds", "thresholds": {
            target: float(pd.to_numeric(train.iloc[fit][target], errors="raise").quantile(.99)) for target in task.targets}})
    return ValidationPlan(strategy=strategy, reason=reason, fit=fit, validation=validation, control=control,
        seed=seed, data_version=dataset.source_hash, feedback_policy=FeedbackPolicy(report_ids=reports, class_order=class_order,
            version=3, min_slice_rows=1, joint_disclosure="research_v3"),
        selection_policy=selection_for(task), class_coverage=coverage, observations=observations)


def inner_split(X: pd.DataFrame, y: pd.DataFrame, task: TaskSpec, strategy: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    if strategy in {"temporal", "temporal_group"}:
        time = ordered_periods(X[task.time_column])
        periods = sorted(time.unique())
        horizon = task.horizon if task.kind == "forecasting" else 1
        if len(periods) <= horizon:
            raise ValueError("Insufficient inner training history")
        val = np.flatnonzero(time.isin(periods[-horizon:]))
        fit = np.flatnonzero(time < periods[-horizon])
        if strategy == "temporal_group":
            if not task.group_column or X[task.group_column].isna().any():
                raise ValueError("Joint inner split requires complete groups")
            held_groups = set(X.iloc[val][task.group_column])
            fit = np.asarray([i for i in fit if X.iloc[i][task.group_column] not in held_groups], dtype=int)
    elif strategy in {"group", "exact_group"}:
        if strategy == "exact_group":
            from .identity import feature_groups
            groups = feature_groups(X[task.feature_columns])
        else:
            groups = X[task.group_column]
        fit, val = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, groups=groups))
    elif task.kind in {"classification", "multioutput_classification", "multilabel"}:
        a, b, c = stratified_parts(y.reset_index(drop=True), seed, task.kind == "multilabel")
        fit, val = np.asarray(a), np.asarray(b + c)
    else:
        order = np.random.default_rng(seed).permutation(n)
        count = max(1, round(n * 0.2))
        val, fit = order[:count], order[count:]
    if not len(fit) or not len(val):
        raise ValueError("Inner split is empty")
    if task.classes:
        for target in task.targets:
            labels = set(y[target])
            unseen = set(y.iloc[val][target]) - set(y.iloc[fit][target])
            if unseen:
                raise ValueError(f"Inner split lacks training class coverage for {target}; {strategy} boundaries preserved")
            if task.metric.name in {"roc_auc", "average_precision"} and set(y.iloc[val][target]) != labels:
                raise ValueError(f"Primary metric {task.metric.name} is not computable for every class of {target} on the inner split")
    return np.asarray(fit), np.asarray(val)


def pilot_indices(X: pd.DataFrame, y: pd.DataFrame, task: TaskSpec, strategy: str, max_rows: int, seed: int,
                  min_class_rows=3, min_groups=3) -> np.ndarray:
    if len(X) <= max_rows:
        return np.arange(len(X))
    rng = np.random.default_rng(seed)
    if strategy in {"temporal", "temporal_group"}:
        # Preserve all periods. Lower model complexity instead of sampling away a horizon.
        return np.arange(len(X))
    if strategy in {"group", "exact_group"}:
        from .identity import feature_groups
        codes = X[task.group_column].to_numpy() if strategy == "group" else feature_groups(X[task.feature_columns])
        groups = pd.unique(codes)
        rng.shuffle(groups)
        chosen = []
        for index, group in enumerate(groups):
            rows = np.flatnonzero(codes == group).tolist()
            if len(chosen) + len(rows) > max_rows and index >= min_groups:
                break
            chosen.extend(rows)
        return np.asarray(chosen, dtype=int)
    if task.kind in {"classification", "multioutput_classification", "multilabel"}:
        mandatory: set[int] = set()
        for column in y:
            for label in y[column].unique():
                rows = np.flatnonzero((y[column] == label).to_numpy())
                mandatory.update(rng.choice(rows, min(min_class_rows, len(rows)), replace=False).tolist())
        if len(mandatory) > max_rows:
            return np.arange(len(X))
        rest = np.asarray(sorted(set(range(len(X))) - mandatory))
        return np.asarray(sorted(mandatory | set(rng.choice(rest, min(max_rows - len(mandatory), len(rest)), replace=False).tolist())))
    return np.sort(rng.choice(len(X), max_rows, replace=False))


class InsufficientPilotSupport(ValueError):
    pass


def pseudo_label_split(X, y, task, strategy, seed):
    """Declare the teacher, pseudo pool and untouched gate before any fit."""
    try:
        a, gate = inner_split(X, y, task, strategy, seed)
        teacher, pool = inner_split(X.iloc[a], y.iloc[a], task, strategy, seed + 1)
    except ValueError as error:
        raise InsufficientPilotSupport("Insufficient pseudo-label split support: " + str(error)) from error
    teacher, pool = a[teacher], a[pool]
    counts = y.iloc[pool][task.targets[0]].value_counts()
    if len(pool) < 10 or not (counts >= 5).any():
        raise InsufficientPilotSupport("Insufficient pseudo-label pool support for ten confident rows and a class with five correct predictions")
    return teacher, pool, gate


def pilot_population(X, y, task, strategy, max_rows, seed, spec):
    complete_pseudo = spec.shift_strategy == "pseudo_label"
    nested = (spec.engine != "direct" or spec.early_stopping_rounds or spec.checkpoint_counts
        or complete_pseudo or spec.prediction_policy != "default" or spec.log_inverse == "oof_smearing"
        or any(op.name == "target_encode" for op in spec.transformations))
    # Confidence/student admission is a quality test of the complete recipe.
    # Reducing its population or model complexity cannot establish infeasibility.
    positions = np.arange(len(X)) if complete_pseudo else pilot_indices(X, y, task, strategy, max_rows, seed,
        min_class_rows=12 if nested else 3, min_groups=6 if nested else 3)
    def check(rows):
        if not nested:
            return
        frame, actual = X.iloc[rows], y.iloc[rows]
        if complete_pseudo:
            pseudo_label_split(frame, actual, task, strategy, seed)
        if spec.postprocess_fit_policy == "pooled_oof" or spec.log_inverse == "oof_smearing" or any(
                op.name == "target_encode" for op in spec.transformations):
            from .research_evaluation import oof_plan
            parts, _ = oof_plan(spec, task, frame, actual, strategy, seed)
            # An OOF fold may itself select a stopping schedule. Checking only
            # the outer folds can admit a tiny pilot whose inner training split
            # loses a rare class, although the full permitted population works.
            for fit, _, fresh, _ in parts:
                if (fresh.engine == "direct" and (fresh.early_stopping_rounds or fresh.checkpoint_counts)
                        and not fresh.iteration_schedule and not fresh.frozen_iterations):
                    inner_split(frame.iloc[fit], actual.iloc[fit], task, strategy, fresh.seed)
        else:
            inner_split(frame, actual, task, strategy, seed)
    expanded = False
    reason = "Pseudo-label admission requires the complete population and original model complexity" if complete_pseudo and len(X) > max_rows else None
    try:
        check(positions)
    except ValueError as error:
        reason = str(error)
        positions = np.arange(len(X))
        expanded = True
        try:
            check(positions)
        except ValueError as full_error:
            raise InsufficientPilotSupport("insufficient_pilot_support on full permitted fit: " + str(full_error)) from full_error
    return positions, {"requested_rows": max_rows, "actual_rows": len(positions),
        "expanded_for_structural_support": expanded or len(positions) > max_rows,
        "sample_limitation": reason, "failure_on_small_sample_is_recipe_failure": False,
        "quality_gate_requires_complete_recipe": complete_pseudo}
