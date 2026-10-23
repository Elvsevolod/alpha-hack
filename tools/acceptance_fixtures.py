"""The existing live fixture, independent of the optional pytest test harness."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from automl_agent.contracts import MetricSpec, OutputSpec, TaskSpec


def write_task(root: Path, kind="regression", probability=False, temporal=False, groups=False, multilabel=False):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    n, nt = 180, 25
    a = rng.normal(size=n + nt)
    frame = pd.DataFrame({"row": [f"{i:05d}" for i in range(n + nt)], "value": a,
        "empty": np.full(n + nt, np.nan), "code": np.where(a > 0, "NA", "-")})
    frame.loc[::11, "value"] = np.nan
    frame["answer"] = np.where(a > 0, "positive", "negative") if kind == "classification" else 2 * a
    metric = MetricSpec(name="log_loss" if probability else "f1" if kind == "classification" else "rmse")
    targets = ["answer"]
    if multilabel:
        kind = "multilabel"
        frame["answer"] = (a > 0).astype(int)
        frame["other"] = (a > 0.7).astype(int)
        targets.append("other")
        metric = MetricSpec(name="f1", average="macro")
    if temporal:
        frame["when"] = np.repeat(np.arange(1, 6), [45, 45, 45, 45, nt])
    if groups:
        frame["query"] = np.arange(n + nt) // 5
    train, test = frame.iloc[:n], frame.iloc[n:].drop(columns=targets)
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    if probability:
        columns = ["row", "p_negative", "p_positive"]
        output = OutputSpec(columns=columns, prediction_type="probabilities", probability_columns={"p_negative": ("answer", "negative"), "p_positive": ("answer", "positive")})
    else:
        columns = ["row", *targets]
        output = OutputSpec(columns=columns, prediction_type="labels" if kind in {"classification", "multilabel"} else "values")
    sample = pd.DataFrame({c: test["row"].to_numpy() if c == "row" else np.zeros(nt) for c in columns})
    sample.to_csv(root / "sample_submission.csv", index=False)
    spec = TaskSpec(kind=kind, targets=targets, id_columns=["row"], feature_columns=[c for c in test if c != "row"],
        categorical_columns=["code"], metric=metric, output=output, source_hash="placeholder",
        time_column="when" if temporal else None, group_column="query" if groups else None)
    (root / "task_spec.json").write_text(json.dumps(spec.model_dump(mode="json", exclude={"source_hash"}), ensure_ascii=False))
    (root / "task.md").write_text("Synthetic technical CSV fixture. No external training data.\n")
    return root / "task.md"
