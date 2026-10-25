"""Canonical feature identities and task-local learned artifact boundaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .util import digest


def feature_groups(X):
    """Equality of confirmed X, independent of column order and numeric dtype."""
    frame = pd.DataFrame(index=X.index)
    for column in sorted(X.columns):
        values = X[column]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            numeric = pd.to_numeric(values, errors="raise").astype(float)
            frame[column] = numeric.map(lambda x: "missing" if pd.isna(x) else "number:" + float(0 if x == 0 else x).hex())
        else:
            frame[column] = values.map(lambda x: "missing" if pd.isna(x) else "value:" + str(x))
    if not len(frame.columns):
        return np.zeros(len(X), dtype=np.uint64)
    # Include schema: a value in another named feature is a different X.
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64) ^ np.uint64(int(digest(sorted(frame.columns))[:16], 16))


def artifact_scope(context, spec, task, fit_keys, predict_keys=(), purpose="strict_oof"):
    fit, predict = list(map(str, fit_keys)), list(map(str, predict_keys))
    if purpose == "strict_oof" and set(fit) & set(predict):
        raise ValueError("Strict OOF artifact fit/prediction rows overlap")
    # Full key lists are useful for small fixtures but quadratic in artifact
    # size on a million rows; beyond a bound keep only digest and count.
    fit_field = ({"fit_keys": fit} if len(fit) <= 5000
        else {"fit_keys_count": len(fit), "fit_keys_hash": digest(fit)})
    predict_field = ({"prediction_keys": predict} if len(predict) <= 5000
        else {"prediction_keys_count": len(predict), "prediction_keys_hash": digest(predict)})
    return {"task": task.hash, "data": context.get("data_hash"), "protocol": context["protocol_hash"],
        "fold": context.get("split_id"), "recipe": spec.hash, "seed": spec.seed,
        "versions": context.get("library_versions", {}), "phase": context.get("execution_mode", "train"),
        "purpose": purpose, **fit_field, **predict_field}
