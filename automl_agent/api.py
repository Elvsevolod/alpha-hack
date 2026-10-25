"""Stable interface available to generated components; no controller access."""
from __future__ import annotations

import numpy as np

from .contracts import PredictionBundle, TrainingSummary
from .util import digest


def prediction_bundle(values, X, context, prediction_type=None, classes=None, **channels):
    keys = [str(key) for key in X.index]
    task = context["task"]
    kind = prediction_type or ("probabilities" if task["classes"] else "values")
    return PredictionBundle(data_version=task["source_hash"], split_id=context["split_id"],
        protocol_hash=context["protocol_hash"], row_keys=keys, row_order_hash=digest(keys),
        target_names=task["targets"], prediction_type=kind,
        values={name: np.asarray(array).tolist() for name, array in values.items()},
        classes=classes or task["classes"], **channels)


__all__ = ["PredictionBundle", "TrainingSummary", "prediction_bundle"]
