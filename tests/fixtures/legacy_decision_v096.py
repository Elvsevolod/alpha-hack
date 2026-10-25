"""One final label policy after probability calibration and composition."""
from __future__ import annotations

import numpy as np

from .evaluation import confusion_details, score_predictions
from .util import digest


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
    return np.asarray(classes)[np.argmax(np.log(np.clip(p, 1e-12, 1)) + offsets, axis=1)]


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
    logits = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1))
    offsets = np.zeros(len(classes))

    def objective(values):
        labels = np.asarray(classes)[np.argmax(logits + values, axis=1)]
        score = score_predictions(task, y, {target: labels}, "labels")
        return (score if task.metric.maximize else -score) - float(np.sum(penalty * values ** 2))

    best = objective(offsets)
    grid = (-.5, -.25, 0., .25, .5) if method == "legacy" else (-.75, -.5, -.25, -.125, 0., .125, .25, .5, .75)
    for _ in range(2):
        for column in range(len(classes) if legacy else max(0, len(classes) - 1)):
            if not legacy and support[column] == 0:
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
    predicted = decide(probabilities, classes, {"offsets": offsets})
    return {"offsets": offsets.tolist(), "classes": classes, "method": method,
        "support": support.tolist(), "penalty": penalty.tolist(),
        "scope": scope, "row_keys_hash": digest(list(map(str, row_keys))),
        "folds_hash": folds_hash, "unbiased_quality_claimed": False,
        "training_metric": score_predictions(task, y, {target: predicted}, "labels"),
        "per_class": confusion_details(actual, predicted, classes)}


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
