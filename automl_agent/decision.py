"""One final label policy after probability calibration and composition."""
from __future__ import annotations

import numpy as np

from .evaluation import confusion_details, score_predictions
from .util import digest


def class_array(classes):
    values = np.asarray(classes)
    if values.dtype.kind in {'U', 'S'} and any(not isinstance(label, str) for label in classes):
        values = np.asarray(classes, dtype=object)
    return values


def decide(probabilities, classes, state=None):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(classes) or not np.isfinite(p).all():
        raise ValueError("Decision probabilities/class order are invalid")
    state = state or {}
    if state.get("classes", classes) != classes:
        raise ValueError("Decision class order differs from the confirmed contract")
    offsets = np.asarray(state.get("offsets", np.zeros(len(classes))), dtype=float)
    if offsets.shape != (len(classes),) or not np.isfinite(offsets).all():
        raise ValueError("Decision offsets have an invalid shape")
    return class_array(classes)[np.argmax(np.log(np.clip(p, 1e-12, 1)) + offsets, axis=1)]


def fit_decision(task, y, probabilities, row_keys, folds_hash=None, method="regularized",
                 scope="fit_only_pooled_oof_decision_training"):
    if task.kind != "classification" or task.output.prediction_type != "labels":
        raise ValueError("Learned decision requires single-output classification labels")
    target = task.targets[0]
    classes = task.classes[target]
    actual = y[target].to_numpy()
    support = np.asarray([(actual == label).sum() for label in classes])
    legacy = method == "legacy"
    penalty = np.full(len(classes), .001) if legacy else .002 * np.maximum(1., 30 / np.maximum(1, support))
    probability = np.asarray(probabilities, dtype=float)
    if not len(y) or probability.shape != (len(y), len(classes)) or not np.isfinite(probability).all():
        raise ValueError('Decision probabilities/class order are invalid')
    raw_logits = np.log(np.clip(probability, 1e-12, 1))
    logits = raw_logits
    order = list(range(len(classes)))
    canonical_keys = []
    if not legacy:
        canonical_keys = [(-int(support[i]), float(logits[:, i].mean()), float((logits[:, i] ** 2).mean()),
            float(logits[actual == label, i].mean()) if support[i] else 0.) for i, label in enumerate(classes)]
        order.sort(key=lambda i: (*canonical_keys[i], i))
        logits = logits[:, order]
        penalty = penalty[order]
    labels = class_array(classes)
    encoded = not legacy
    score_task, score_y = task, y
    actual_indices = None
    if encoded:
        lookup = {label: i for i, label in enumerate(classes)}
        try:
            actual_indices = np.asarray([lookup[label] for label in actual])
        except KeyError:
            raise ValueError('Decision fit targets contain undeclared classes') from None
        metric = task.metric.model_copy(update={'positive_label': lookup.get(task.metric.positive_label, task.metric.positive_label)})
        score_task = task.model_copy(update={'classes': {target: list(range(len(classes)))}, 'metric': metric})
        score_y = y.copy()
        score_y[target] = actual_indices
    offsets = np.zeros(len(classes))

    def metric_for_indices(indices):
        prediction = indices if encoded else labels[indices]
        return score_predictions(score_task, score_y, {target: prediction}, "labels")

    def objective(values):
        indices = np.asarray(order)[np.argmax(logits + values, axis=1)]
        score = metric_for_indices(indices)
        return (score if task.metric.maximize else -score) - float(np.sum(penalty * values ** 2))

    best = objective(offsets)
    grid = (-.5, -.25, 0., .25, .5) if method == "legacy" else (-.75, -.5, -.25, -.125, 0., .125, .25, .5, .75)
    for _ in range(2):
        for column in range(len(classes)):
            if not legacy and support[order[column]] == 0:
                continue
            selected = offsets.copy()
            for value in grid:
                trial = offsets.copy()
                trial[column] = value
                if not legacy:
                    trial -= trial.mean()  # Remove the unidentifiable common shift.
                if np.max(np.abs(trial)) > 1:
                    continue
                score = objective(trial)
                if score > best + 1e-12:
                    best, selected = score, trial
            offsets = selected
    offsets = offsets[np.argsort(order)]
    penalty = penalty[np.argsort(order)]
    adjusted = raw_logits + offsets
    final_indices = np.argmax(adjusted, axis=1)
    training_metric = metric_for_indices(final_indices)
    zero_guard = False
    if not legacy:
        zero_indices = np.argmax(raw_logits, axis=1)
        zero_metric = metric_for_indices(zero_indices)
        measured_objective = (training_metric if task.metric.maximize else -training_metric) - float(np.sum(penalty * offsets ** 2))
        if measured_objective < (zero_metric if task.metric.maximize else -zero_metric) - 1e-12:
            # Canonical-order ties during optimization can differ from the
            # declared argmax contract. Never worsen its actual zero baseline.
            offsets = np.zeros(len(classes))
            adjusted, final_indices, training_metric = raw_logits, zero_indices, zero_metric
            zero_guard = True
    predicted = decide(probabilities, classes, {"offsets": offsets})
    detail = confusion_details(actual_indices, final_indices, list(range(len(classes)))) if encoded else confusion_details(actual, predicted, classes)
    if encoded:
        detail['labels'] = classes
        for row in detail['classes']:
            row['class'] = classes[row['class']]
        for row in detail['main_errors']:
            row['actual'], row['predicted'] = classes[row['actual']], classes[row['predicted']]
    return {"offsets": offsets.tolist(), "classes": classes, "method": method,
        "support": support.tolist(), "penalty": penalty.tolist(),
        "scope": scope, "row_keys_hash": digest(list(map(str, row_keys))),
        "folds_hash": folds_hash, "unbiased_quality_claimed": False,
        "training_metric": training_metric,
        "per_class": detail,
        **({'optimizer': 'canonical_all_coordinates_v1', 'canonical_order': order, 'canonical_key_ties': len(canonical_keys) - len(set(canonical_keys)),
            'zero_objective_guard_applied': zero_guard,
            'raw_argmax_tie_rows': int((np.sum(raw_logits == raw_logits.max(axis=1, keepdims=True), axis=1) > 1).sum()),
            'final_argmax_tie_rows': int((np.sum(adjusted == adjusted.max(axis=1, keepdims=True), axis=1) > 1).sum()),
            'permutation_guarantee': 'distinct_fit_keys_and_no_fit_or_final_argmax_ties'} if not legacy else {})}


def frozen_decision_valid(spec, task):
    """Recognize both serialized formats with the executable class contract."""
    for target in task.targets:
        classes = task.classes.get(target, [])
        state = spec.learned_decision.get(target)
        if state is None and len(task.targets) == 1 and spec.class_offsets is not None:
            state = {"classes": classes, "offsets": spec.class_offsets}
        if not classes or not isinstance(state, dict) or "offsets" not in state:
            return False
        try:
            decide(np.full((1, len(classes)), 1 / len(classes)), classes, state)
        except (ValueError, TypeError):
            return False
    return True
