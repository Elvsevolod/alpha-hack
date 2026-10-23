from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics

from .contracts import PredictionBundle, TaskSpec
from .feedback import partition_masks, partition_summary, registered_partitions, whole_ranking_groups
from .util import digest


def align_predictions(bundle: PredictionBundle, expected_keys: list[str], task: TaskSpec,
                      protocol_hash: str, split_id: str) -> dict[str, np.ndarray]:
    if bundle.data_version != task.source_hash or bundle.protocol_hash != protocol_hash or bundle.split_id != split_id:
        raise ValueError("Prediction data/protocol/split mismatch")
    if bundle.row_order_hash != digest(bundle.row_keys):
        raise ValueError("Prediction order checksum mismatch")
    if len(bundle.row_keys) != len(set(bundle.row_keys)) or set(bundle.row_keys) != set(expected_keys):
        raise ValueError("Prediction keys are missing, extra, or duplicated")
    if bundle.target_names != task.targets or set(bundle.values) != set(task.targets):
        raise ValueError("Prediction target names mismatch")
    mapping = {key: i for i, key in enumerate(bundle.row_keys)}
    order = np.asarray([mapping[key] for key in expected_keys])
    result = {}
    for target in task.targets:
        arr = np.asarray(bundle.values[target])
        if len(arr) != len(expected_keys):
            raise ValueError("Prediction length mismatch")
        arr = arr[order]
        if bundle.prediction_type == "probabilities":
            labels = bundle.classes.get(target, [])
            expected = task.classes[target]
            if len(labels) != len(set(labels)) or set(labels) != set(expected):
                raise ValueError("Probability classes mismatch")
            if arr.shape != (len(expected_keys), len(labels)):
                raise ValueError("Invalid probability shape")
            arr = arr.astype(float)[:, [labels.index(label) for label in expected]]
            if not np.isfinite(arr).all() or (arr < -1e-10).any() or (arr > 1 + 1e-10).any():
                raise ValueError("Nonfinite or out-of-range probabilities")
            if not np.allclose(arr.sum(axis=1), 1, atol=1e-6):
                raise ValueError("Probability rows do not sum to one")
            arr = np.clip(arr, 0, 1)
        elif bundle.prediction_type == "labels":
            if arr.ndim != 1 or pd.isna(arr).any() or not set(arr.tolist()) <= set(task.classes[target]):
                raise ValueError("Invalid classification labels")
        else:
            if arr.ndim != 1:
                raise ValueError("Numeric predictions must have one value per target and row")
            arr = arr.astype(float)
            if not np.isfinite(arr).all():
                raise ValueError("Nonfinite predictions")
        result[target] = arr
    required_probability = task.metric.name in {"roc_auc", "log_loss", "average_precision"} or task.output.prediction_type == "probabilities"
    if required_probability and bundle.prediction_type != "probabilities":
        raise ValueError("The task requires probabilities, not decision scores")
    if task.kind not in {"classification", "multioutput_classification", "multilabel"} and bundle.prediction_type not in {"values", "scores"}:
        raise ValueError("Task family and prediction type disagree")
    return result


def _ranking(y: np.ndarray, p: np.ndarray, groups: np.ndarray, task: TaskSpec) -> float:
    scores = []
    for group in pd.unique(groups):
        mask = groups == group
        actual, predicted = y[mask].astype(float), p[mask].astype(float)
        k = min(task.metric.k, len(actual))
        if task.metric.name == "ndcg":
            if (actual < 0).any():
                raise ValueError("NDCG relevance must be nonnegative")
            score = metrics.ndcg_score(actual[None, :], predicted[None, :], k=k) if len(actual) > 1 else float(actual[0] > 0)
        else:
            relevant = actual > task.metric.relevance_threshold
            order = np.argsort(-predicted, kind="stable")[:k]
            ordered = relevant[order]
            precision = np.cumsum(ordered) / np.arange(1, k + 1)
            score = float((precision * ordered).sum() / max(1, min(int(relevant.sum()), k)))
        scores.append(float(score))
    return float(np.mean(scores))


def score_predictions(task: TaskSpec, y: pd.DataFrame, predictions: dict[str, np.ndarray],
                      prediction_type: str, groups: np.ndarray | None = None) -> float:
    name = task.metric.name
    if task.kind == "multilabel" and name in {"f1", "accuracy", "average_precision", "roc_auc"}:
        actual = y[task.targets].to_numpy(dtype=int)
        if prediction_type == "probabilities":
            probability = np.column_stack([predictions[t][:, task.classes[t].index(1)] if 1 in task.classes[t] else np.zeros(len(y)) for t in task.targets])
            labels = probability >= 0.5
        else:
            labels = np.column_stack([predictions[t] for t in task.targets]).astype(int)
        if name == "f1":
            value = metrics.f1_score(actual, labels, average=task.metric.average, zero_division=0)
        elif name == "accuracy":
            value = metrics.accuracy_score(actual, labels)
        elif name == "average_precision":
            value = metrics.average_precision_score(actual, probability, average=task.metric.average)
        else:
            value = metrics.roc_auc_score(actual, probability, average=task.metric.average)
        if not np.isfinite(value):
            raise ValueError("Metric is undefined on this split")
        return float(value)
    scores = []
    for target in task.targets:
        actual, predicted = y[target].to_numpy(), predictions[target]
        classes = task.classes.get(target, [])
        if prediction_type == "probabilities":
            labels = np.asarray(classes)[np.argmax(predicted, axis=1)]
        else:
            labels = predicted
        if name == "rmse":
            score = np.sqrt(metrics.mean_squared_error(actual, predicted))
        elif name == "mse":
            score = metrics.mean_squared_error(actual, predicted)
        elif name == "mae":
            score = metrics.mean_absolute_error(actual, predicted)
        elif name == "r2":
            score = metrics.r2_score(actual, predicted)
        elif name == "f1":
            kwargs: dict[str, Any] = {"average": task.metric.average, "zero_division": 0}
            if task.metric.average == "binary":
                if task.metric.positive_label is None:
                    raise ValueError("Binary F1 requires a positive label")
                kwargs["pos_label"] = task.metric.positive_label
            score = metrics.f1_score(actual, labels, **kwargs)
        elif name == "accuracy":
            score = metrics.accuracy_score(actual, labels)
        elif name == "log_loss":
            canonical_classes = sorted(classes)
            canonical_probabilities = predicted[:, [classes.index(label) for label in canonical_classes]]
            score = metrics.log_loss(actual, canonical_probabilities, labels=canonical_classes)
        elif name in {"roc_auc", "average_precision"}:
            if len(classes) == 2:
                positive = task.metric.positive_label
                if positive not in classes:
                    raise ValueError("Positive label is not in the declared class space")
                binary_y, probability = actual == positive, predicted[:, classes.index(positive)]
                scorer = metrics.roc_auc_score if name == "roc_auc" else metrics.average_precision_score
                score = scorer(binary_y, probability)
            else:
                encoded = np.column_stack([actual == label for label in classes])
                if name == "roc_auc":
                    canonical_classes = sorted(classes)
                    canonical_probabilities = predicted[:, [classes.index(label) for label in canonical_classes]]
                    score = metrics.roc_auc_score(actual, canonical_probabilities, labels=canonical_classes, average=task.metric.average, multi_class="ovr")
                else:
                    score = metrics.average_precision_score(encoded, predicted, average=task.metric.average)
        elif name in {"ndcg", "map"}:
            if groups is None:
                raise ValueError("Ranking metric requires groups")
            score = _ranking(actual, predicted, groups, task)
        else:
            raise ValueError(f"Unsupported metric: {name}")
        if not np.isfinite(score):
            raise ValueError("Metric is undefined on this split")
        scores.append(float(score))
    weights = task.metric.target_weights if task.metric.aggregation == "weighted" else None
    return float(np.average(scores, weights=weights))


def feedback_reports(task, plan, X, y, predictions, prediction_type):
    policy = plan.feedback_policy
    base = {"scope": "external_validation", "data_version": task.source_hash,
        "protocol_hash": plan.hash, "policy_hash": policy.hash}
    reports = {}
    families = {}
    missing = X[task.feature_columns].isna().any(axis=1).to_numpy()
    families["missingness"] = {"missing": missing, "observed": ~missing}
    if "classes" in policy.report_ids:
        families["classes"] = {f"{target}:{label}": (y[target] == label).to_numpy()
            for target in task.targets for label in task.classes[target]}
    partitions = registered_partitions(plan)
    for identifier, rule in partitions.items():
        families[identifier] = partition_masks(rule, X)
    for family in policy.report_ids:
        masks = families.get(family, {})
        if family in partitions and len(masks) < 2:
            reports[family] = {**base, "report_id": family, "status": "unsupported", "reason": "single_nonempty_partition"}
            continue
        # Suppress a whole complementary family if a small cell could be reconstructed.
        if not masks or any(int(mask.sum()) < policy.min_slice_rows for mask in masks.values()):
            reports[family] = {**base, "report_id": family, "status": "suppressed", "reason": "minimum_support_or_complementary_disclosure"}
            continue
        slices = {}
        unsupported = False
        for label, mask in masks.items():
            try:
                if task.kind == "ranking" and not whole_ranking_groups(X, task.group_column, mask):
                    raise ValueError("Ranking diagnostics cannot split a query")
                groups = X.loc[mask, task.group_column].to_numpy() if task.kind == "ranking" else None
                value = score_predictions(task, y.loc[mask], {t: p[mask] for t, p in predictions.items()}, prediction_type, groups)
                slices[label] = {"rows": int(mask.sum()), "score": value}
            except ValueError:
                unsupported = True
                break
        reports[family] = {**base, "report_id": family, "status": "unsupported" if unsupported else "available"}
        if not unsupported:
            reports[family]["slices"] = slices
            if family in partitions:
                reports[family]["partition"] = partition_summary(partitions[family])
    return reports


def choose_leader(results, policy):
    eligible = [r for r in results if r.status == "completed" and r.score is not None and r.scope != "pilot"]
    if not eligible:
        return None, None
    if len({r.validation_hash for r in eligible}) > 1:
        raise ValueError("Scores from different validation protocols cannot be compared")
    best = (max if policy.maximize else min)(r.score for r in eligible)
    tied = [r for r in eligible if policy.tied(r.score, best)]
    leader = min(tied, key=lambda r: tuple(getattr(r, field) for field in policy.tie_order))
    return leader, best
