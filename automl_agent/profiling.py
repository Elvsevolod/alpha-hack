"""Shared, versioned EDA facts. No learning, model selection or network access."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .contracts import DataProfile
from .util import digest

PROFILE_VERSION = "6"
DEFAULT_PROFILE_MODE = "unified"
MAX_CORRELATION_ROWS = 20_000
MAX_CORRELATION_COLUMNS = 30
MIN_PAIR_ROWS = 30


def semantic_type(task, column):
    if column in task.targets:
        return "categorical" if task.classes else "numeric"
    if column in task.categorical_columns:
        return "categorical"
    if column in task.text_columns:
        return "text"
    return "numeric"


def numeric_statistics(values):
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"status": "no_finite_values"}
    clean = clean.astype(float)
    quantiles = clean.quantile([.01, .25, .5, .75, .99])
    result = {"status": "available", "min": float(clean.min()), "max": float(clean.max()),
        "median": float(quantiles.loc[.5]), "q01": float(quantiles.loc[.01]),
        "q25": float(quantiles.loc[.25]), "q75": float(quantiles.loc[.75]),
        "q99": float(quantiles.loc[.99]), "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)), "zero_fraction": float((clean == 0).mean()),
        "negative_fraction": float((clean < 0).mean()), "finite_rows": len(clean)}
    undefined = [name for name, value in result.items() if isinstance(value, float) and not np.isfinite(value)]
    for name in undefined:
        result[name] = None
    if undefined:
        result.update(status="partial", undefined_statistics=undefined)
    return result


def build_profile(dataset, indices, scope="fit", frame=None, deadline=None):
    if scope not in {"fit", "validation_features", "test_features"}:
        raise ValueError("Unsupported profile scope")
    task = dataset.task
    fit = frame if frame is not None else (dataset.test if scope == "test_features" else dataset.train.iloc[indices])
    permitted = task.feature_columns + (task.targets if scope == "fit" else [])
    fit = fit[permitted]
    columns = {}
    for column in task.feature_columns:
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Profile deadline exhausted")
        values, test = fit[column], dataset.test[column]
        item = {"dtype": str(values.dtype), "semantic_type": semantic_type(task, column),
            "missing": int(values.isna().sum()), "missing_fraction": float(values.isna().mean()) if len(values) else 0.,
            "test_missing_fraction": float(test.isna().mean()) if len(test) else 0.,
            "unique": int(values.nunique()), "numeric": bool(pd.api.types.is_numeric_dtype(values)),
            "constant": bool(len(values) and values.nunique() == 1), "empty": bool(values.isna().all())}
        if item["numeric"] and values.notna().any():
            item.update(numeric_statistics(values))
        if item["semantic_type"] != "numeric":
            item["most_common_fit"] = {str(k): int(v) for k, v in values.value_counts().head(8).items()}
            item["new_test_category_rows"] = int((test.notna() & ~test.isin(set(values.dropna()))).sum())
        columns[column] = item
    risks = {"empty_columns": any(v["missing"] == len(fit) for v in columns.values()),
        "unknown_categories": any(v.get("new_test_category_rows", 0) for v in columns.values()),
        "missingness_shift": any(abs(v["missing_fraction"] - v["test_missing_fraction"]) >= .1 for v in columns.values()),
        "high_cardinality": any(columns[c]["unique"] > 100 for c in task.categorical_columns)}
    features = fit[task.feature_columns]
    duplicates = features.duplicated(keep=False)
    risks.update(duplicate_features=bool(duplicates.any()), label_conflicts=False, duplicate_target_variation=False)
    targets = {}
    if scope == "fit":
        if task.classes:
            counts = [int((fit[target] == label).sum()) for target, labels in task.classes.items() for label in labels]
            risks.update(class_imbalance=bool(counts and min(counts) * 10 < max(counts)),
                rare_classes=bool(counts and min(counts) < 100))
            for target, labels in task.classes.items():
                targets[target] = {"semantic_type": "categorical", "counts": [
                    {"label": label, "rows": int((fit[target] == label).sum())} for label in labels],
                    "missing": int(fit[target].isna().sum())}
        else:
            targets = {t: {"semantic_type": "numeric", **numeric_statistics(fit[t])} for t in task.targets}
            risks["heavy_tail"] = any(float(fit[t].quantile(.99) - fit[t].quantile(.5)) >
                4 * max(float(fit[t].quantile(.75) - fit[t].quantile(.25)), 1e-12) for t in task.targets)
        if duplicates.any() and task.feature_columns:
            conflicts = fit.loc[duplicates].groupby(task.feature_columns, dropna=False, observed=True)[task.targets].nunique()
            if task.classes:
                risks["label_conflicts"] = bool((conflicts > 1).any().any())
            else:
                risks["duplicate_target_variation"] = int((conflicts > 1).any(axis=1).sum()) >= 2
    flags = {key: bool(value) for key, value in risks.items()}
    if scope != "fit":
        flags.pop("label_conflicts", None)
        flags.pop("duplicate_target_variation", None)
    findings = []
    questions = [
        ("empty_columns", "Some features contain no observed values.", "Check feature representation; do not drop automatically."),
        ("constant_features", "Some features have one distinct nonmissing value.", "Check utility and missingness; test removal rather than applying it automatically."),
        ("label_conflicts", "Identical features have conflicting labels.", "Check missing context or label noise; do not transfer labels by ID."),
        ("duplicate_target_variation", "Several duplicate-feature groups have different targets.", "Check whether repeated observations reflect missing context."),
        ("duplicate_features", "Repeated feature rows exist.", "Check observation semantics; duplicates alone do not establish leakage."),
        ("rare_classes", "Some declared fit classes have fewer than 100 examples.", "Inspect fit-only inner precision/recall and coverage after the baseline."),
        ("class_imbalance", "Fit class counts differ by more than a factor of ten.", "Compare appropriate hypotheses; class weights are not automatically beneficial."),
        ("missingness_shift", "Fit and test missing fractions differ by at least 0.1.", "Check missing-value handling and transfer assumptions."),
        ("unknown_categories", "Test contains categories absent in this profile's rows.", "Verify unseen-category handling."),
        ("heavy_tail", "The fit target meets the existing heavy-tail heuristic.", "Inspect inner model errors on the original target scale after the baseline."),
        ("high_cardinality", "Some categorical features have more than 100 values.", "Compare supported representations under the measured budget."),
    ]
    for kind, observation, question in questions:
        if flags.get(kind) or kind == "constant_features" and any(item["constant"] for item in columns.values()):
            findings.append({"kind": kind, "scope": scope, "observation": observation,
                "limitation": "Diagnostic signal, not proof of cause or model improvement.", "next_question": question})
    rows = {"profile": len(fit), "train": len(dataset.train), "test": len(dataset.test)}
    if scope == "fit":
        rows["fit"] = len(fit)
    return DataProfile(data_version=dataset.source_hash, scope=scope, source="immutable_csv", version=int(PROFILE_VERSION),
        rows=rows, columns=columns,
        targets=targets, findings=findings, observations=[{"type": "fit_risks", "flags": flags}])


def profile_key(dataset, indices, scope, protocol_hash=None):
    return digest({"data": dataset.source_hash, "task": dataset.task.model_dump(mode="json")
        if hasattr(dataset.task, "model_dump") else vars(dataset.task),
        "indices": list(indices), "scope": scope, "analyzer": PROFILE_VERSION, "protocol_hash": protocol_hash})


def correlations(frame, excluded=None):
    """Called only by the resource-monitored worker on staged, permitted values."""
    if len(frame) > MAX_CORRELATION_ROWS or len(frame.columns) > MAX_CORRELATION_COLUMNS:
        raise ValueError("Correlation input exceeds bounded staging limits")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    masks = frame.notna().astype("int64")
    counts = masks.T @ masks
    matrices = {method: frame.corr(method=method, min_periods=MIN_PAIR_ROWS) for method in ("pearson", "spearman")}
    statuses = {}
    for a in frame:
        statuses[a] = {}
        for b in frame:
            pair = frame[[a, b]].dropna()
            status = "insufficient_support" if int(counts.loc[a, b]) < MIN_PAIR_ROWS else (
                "constant" if any(pair.iloc[:, i].nunique() <= 1 for i in range(2)) else "available")
            if status != "available":
                for matrix in matrices.values():
                    matrix.loc[a, b] = np.nan
            elif any(not np.isfinite(matrix.loc[a, b]) for matrix in matrices.values()):
                status = "undefined"
            statuses[a][b] = status
    return {"correlations": json.loads(matrices["pearson"].to_json()),
        "spearman": json.loads(matrices["spearman"].to_json()), "pair_counts": counts.to_dict(),
        "pair_status": statuses, "excluded_columns": excluded or {}, "min_pair_rows": MIN_PAIR_ROWS}

