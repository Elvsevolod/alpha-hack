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
    if task.kind in {"classification", "multioutput_classification", "multilabel"} and bundle.prediction_type not in {"labels", "probabilities"}:
        raise ValueError("Classification requires labels or probabilities")
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
    required_probability = task.metric.requires_probabilities or task.output.prediction_type == "probabilities"
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
        elif name == "mape":
            score = metrics.mean_absolute_percentage_error(actual.astype(np.float64), predicted.astype(np.float64))
        elif name == "r2":
            score = metrics.r2_score(actual, predicted)
        elif name == "f1":
            kwargs: dict[str, Any] = {"average": task.metric.average, "zero_division": 0}
            if task.metric.average == "binary":
                if task.metric.positive_label is None:
                    raise ValueError("Binary F1 requires a positive label")
                kwargs["pos_label"] = task.metric.positive_label
            else:
                kwargs["labels"] = classes
            score = metrics.f1_score(actual, labels, **kwargs)
        elif name == "accuracy":
            score = metrics.accuracy_score(actual, labels)
        elif name == "balanced_accuracy":
            if not classes or not set(actual) <= set(classes) or not set(labels) <= set(classes):
                raise ValueError("Balanced accuracy requires declared classes and no unknown labels")
            score = metrics.balanced_accuracy_score(actual, labels, adjusted=False)
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
            raise ValueError(f"unsupported_metric: {name}")
        if not np.isfinite(score):
            raise ValueError("Metric is undefined on this split")
        scores.append(float(score))
    weights = task.metric.target_weights if task.metric.aggregation == "weighted" else None
    return float(np.average(scores, weights=weights))


def feedback_reports(task, plan, X, y, predictions, prediction_type, fit_X=None):
    policy = plan.feedback_policy
    base = {"scope": "external_validation", "data_version": task.source_hash,
        "protocol_hash": plan.hash, "policy_hash": policy.hash}
    reports = {}
    families = {}
    missing = X[task.feature_columns].isna().any(axis=1).to_numpy()
    families["missingness"] = {"missing": missing, "observed": ~missing}
    if policy.version >= 3 and fit_X is not None:
        from .research_evaluation import feature_groups
        known = np.isin(feature_groups(X[task.feature_columns]), feature_groups(fit_X[task.feature_columns]))
        families["seen_unseen"] = {"seen_exact_features": known, "unseen_exact_features": ~known}
        if task.group_column:
            mask = X[task.group_column].isin(fit_X[task.group_column]).to_numpy()
            families["entity_coverage"] = {"known_entity": mask, "cold_entity": ~mask}
    if "classes" in policy.report_ids:
        families["classes"] = {f"{target}:{label}": (y[target] == label).to_numpy()
            for target in task.targets for label in task.classes[target]}
    partitions = registered_partitions(plan)
    for identifier, rule in partitions.items():
        families[identifier] = partition_masks(rule, X)
    for family in policy.report_ids:
        if policy.version >= 3 and family == "regression_errors":
            extra = {name: mask for family_name in ("seen_unseen", "entity_coverage")
                for name, mask in families.get(family_name, {}).items()}
            reports[family] = regression_errors(task, plan, X, y, predictions, base, extra)
            continue
        if family == "classes" and policy.version >= 2:
            reports[family] = class_metrics(task, plan, y, predictions, prediction_type, base)
            continue
        if policy.version == 2 and task.classes and (policy.history_guard!='initial_protocol' or len(task.targets)>1 or any(
                int((y[target]==label).sum())<policy.min_slice_rows for target in task.classes for label in task.classes[target])):
            reports[family] = {**base, "report_id":family, "status":"suppressed", "reason":"joint_class_slice_history_not_established"}
            continue
        masks = families.get(family, {})
        if family in partitions and len(masks) < 2:
            reports[family] = {**base, "report_id": family, "status": "unsupported", "reason": "single_nonempty_partition"}
            continue
        # Suppress a whole complementary family if a small cell could be reconstructed.
        if not masks or policy.version < 3 and any(int(mask.sum()) < policy.min_slice_rows for mask in masks.values()):
            reports[family] = {**base, "report_id": family, "status": "suppressed", "reason": "minimum_support_or_complementary_disclosure"}
            continue
        slices = {}
        unsupported = False
        for label, mask in masks.items():
            if not int(mask.sum()):
                slices[label] = {"rows": 0, "status": "empty", "uncertain": True}
                continue
            try:
                if task.kind == "ranking" and not whole_ranking_groups(X, task.group_column, mask):
                    raise ValueError("Ranking diagnostics cannot split a query")
                groups = X.loc[mask, task.group_column].to_numpy() if task.kind == "ranking" else None
                value = score_predictions(task, y.loc[mask], {t: p[mask] for t, p in predictions.items()}, prediction_type, groups)
                slices[label] = {"rows": int(mask.sum()), "score": value}
                if policy.version >= 3:
                    slices[label].update(status="available", uncertain=int(mask.sum()) < 30)
                    if task.classes and family in {"seen_unseen", "entity_coverage"}:
                        slices[label]["targets"] = {}
                        for target, labels in task.classes.items():
                            values = predictions[target][mask]
                            predicted = np.asarray(labels)[np.argmax(values, axis=1)] if prediction_type == "probabilities" else values
                            slices[label]["targets"][target] = confusion_details(y.loc[mask, target], predicted, labels)
                if family == "classes":
                    slices[label]["score_semantics"] = "metric_conditioned_on_true_class_not_per_class_f1"
            except ValueError:
                unsupported = True
                break
        reports[family] = {**base, "report_id": family, "status": "unsupported" if unsupported else "available"}
        if not unsupported:
            reports[family]["slices"] = slices
            if family in partitions:
                reports[family]["partition"] = partition_summary(partitions[family])
    return reports


def class_metrics(task, plan, y, predictions, prediction_type, base):
    """Trusted full-confusion evaluation; never expose counts or arbitrary masks.

    Sparse classes plus aggregate/slice/history disclosures have no general safe
    composition here. Suppress the entire detail family in that case, including
    a complementary class chosen from the pre-registered fit order. Historical
    reports stay immutable; no claim of retroactive secrecy is made.
    """
    policy = plan.feedback_policy
    if policy.version >= 3:
        targets = {}
        rows = []
        for target in task.targets:
            labels = task.classes[target]
            predicted = np.asarray(labels)[np.argmax(predictions[target], axis=1)] if prediction_type == "probabilities" else predictions[target]
            detail = confusion_details(y[target], predicted, labels)
            targets[target] = detail
            rows.extend({**row, "id": f"{target}:{row['class']}"} for row in detail["classes"])
        return {**base, "report_id": "classes", "status": "available", "policy_version": policy.version,
            "classes": rows, "targets": targets, "composition": "fixed_research_reports; control_remains_closed"}
    rows = []
    sparse = []
    for target in task.targets:
        labels = task.classes[target]
        predicted = np.asarray(labels)[np.argmax(predictions[target], axis=1)] if prediction_type == "probabilities" else predictions[target]
        matrix = metrics.confusion_matrix(y[target], predicted, labels=labels)
        support, predicted_support, tp = matrix.sum(axis=1), matrix.sum(axis=0), matrix.diagonal()
        for index, label in enumerate(labels):
            identifier = f"{target}:{label}"
            if support[index] < policy.min_slice_rows:sparse.append(identifier)
            precision = float(tp[index]/predicted_support[index]) if predicted_support[index] else 0.
            recall = float(tp[index]/support[index]) if support[index] else 0.
            f1 = float(2*tp[index]/(support[index]+predicted_support[index])) if support[index]+predicted_support[index] else 0.
            rows.append({"id":identifier, "support_category":"below_minimum" if support[index]<policy.min_slice_rows else "at_least_minimum",
                "precision":precision, "recall":recall, "f1":f1})
    complementary = next((identifier for identifier in policy.class_order if identifier not in sparse), None) if sparse else None
    restricted = bool(sparse) or len(task.targets)>1 or task.kind == 'multilabel' or policy.history_guard!='initial_protocol'
    for row in rows:
        row['status'] = 'suppressed' if restricted else 'available'
        row['reason'] = ('minimum_support' if row['id'] in sparse else 'complementary_disclosure' if row['id']==complementary
            else 'joint_targets_slices_history_not_established' if restricted else 'full_confusion_matrix')
        if restricted:
            for key in ('precision','recall','f1'):row.pop(key)
    return {**base, "report_id":"classes", "status":"suppressed" if restricted else "available",
        "policy_version":policy.version, "classes":rows,
        "composition":"fixed_all_history_policy; sparse or multi-target details withheld; previous disclosures are not revoked"}


def confusion_details(actual, predicted, labels):
    matrix = metrics.confusion_matrix(actual, predicted, labels=labels)
    support, predicted_support, tp = matrix.sum(axis=1), matrix.sum(axis=0), matrix.diagonal()
    rows = [{"class": label, "status": "available", "support": int(support[i]),
        "predicted_support": int(predicted_support[i]), "true_positive": int(tp[i]),
        "false_positive": int(predicted_support[i] - tp[i]), "false_negative": int(support[i] - tp[i]),
        "precision": float(tp[i] / predicted_support[i]) if predicted_support[i] else 0.,
        "recall": float(tp[i] / support[i]) if support[i] else 0.,
        "f1": float(2 * tp[i] / (support[i] + predicted_support[i])) if support[i] + predicted_support[i] else 0.,
        "uncertain": bool(support[i] < 30 or predicted_support[i] < 30)} for i, label in enumerate(labels)]
    errors = sorted(({"actual": labels[i], "predicted": labels[j], "rows": int(matrix[i, j])}
        for i in range(len(labels)) for j in range(len(labels)) if i != j and matrix[i, j]), key=lambda row: -row["rows"])
    return {"labels": labels, "confusion_matrix": matrix.tolist(), "classes": rows, "main_errors": errors[:10]}


def regression_errors(task, plan, X, y, predictions, base, extra_masks=None):
    thresholds = next((item["thresholds"] for item in plan.observations if item.get("type") == "fit_target_tail_thresholds"), {})
    masks = {"all": np.ones(len(X), dtype=bool), "missing": X[task.feature_columns].isna().any(axis=1).to_numpy()}
    masks["observed"] = ~masks["missing"]
    masks.update(extra_masks or {})
    for identifier, rule in registered_partitions(plan).items():
        masks.update({identifier + ":" + str(key): value for key, value in partition_masks(rule, X).items()})
    targets = {}
    for target in task.targets:
        actual = y[target].to_numpy(dtype=float)
        residual = actual - predictions[target]
        squared = residual ** 2
        total = float(squared.sum())
        slices = {}
        for name, mask in {**masks, "fit_top_1pct_target": actual >= thresholds.get(target, float("inf"))}.items():
            n = int(mask.sum())
            slices[name] = {"rows": n, "status": "available" if n else "empty", "uncertain": n < 30,
                "sse": float(squared[mask].sum()),
                "residual_bias": float(residual[mask].mean()) if n else None,
                "rmse": float(np.sqrt(squared[mask].mean())) if n else None,
                "mae": float(np.abs(residual[mask]).mean()) if n else None,
                "squared_error_fraction": float(squared[mask].sum() / total) if total else 0.}
        targets[target] = slices
    return {**base, "report_id": "regression_errors", "status": "available", "targets": targets,
        "tail_thresholds_from_fit": thresholds, "residual_definition": "actual_minus_prediction"}


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
