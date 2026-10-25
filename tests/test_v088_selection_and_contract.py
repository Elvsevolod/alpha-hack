import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from automl_agent.contracts import MetricSpec, OutputSpec, PipelineSpec, SelectionPolicy, TaskSpec, Transformation
from automl_agent.finalists import select_finalists
from automl_agent.identity import artifact_scope
from automl_agent.pipelines import refit_iterations
from automl_agent.supervised_features import SupervisedFeatures


def _candidate(identifier, identity, score, windows=None, cost=1.0, created=0):
    return {"id": identifier, "identity": identity, "score": score, "windows": windows or {},
        "cost": cost, "created_index": created}


def test_control_receives_top_candidates_even_when_validation_rejects_them():
    candidates = [
        _candidate("baseline", "b", 44.36, {"w1": 51.0, "w2": 62.0}, created=0),
        _candidate("es", "e", 45.11, {"w2": 55.3}, created=1),
        _candidate("log", "l", 45.20, {"w2": 56.5}, created=2),
        _candidate("cat", "c", 47.32, {"w2": 54.8}, created=3),
    ]
    policy = SelectionPolicy(maximize=False, min_delta_abs=1e-12, min_delta_rel=.005)
    selected, audit = select_finalists(candidates, "baseline", "baseline", policy, ["w1", "w2"])
    assert selected == ["baseline", "es", "log"]
    assert audit["control_top_up"] == ["es", "log"]
    assert audit["limit"] == 3
    assert len(selected) == len(set(audit["selected"]))


def test_control_top_up_uses_maximize_direction_and_distinct_identities():
    candidates = [
        _candidate("baseline", "b", .600, created=0),
        _candidate("best", "x", .650, created=1),
        _candidate("dupe", "x", .640, created=2),
        _candidate("third", "y", .630, created=3),
        _candidate("fourth", "z", .620, created=4),
    ]
    policy = SelectionPolicy(maximize=True, min_delta_abs=.005)
    selected, _ = select_finalists(candidates, "baseline", "best", policy, [])
    assert selected == ["baseline", "best", "dupe"] or selected == ["baseline", "best", "third"]
    # Identity x must appear once; the next slot goes to the best remaining identity.
    assert len({c["identity"] for c in candidates if c["id"] in selected}) == len(selected)


def test_refit_iterations_scales_only_for_larger_refit_population():
    spec = PipelineSpec(model="lightgbm", iteration_scale_rows=22000, refit_iteration_policy="sqrt_rows")
    assert refit_iterations(spec, 130, 53000, "refit") == 202
    assert refit_iterations(spec, 130, 53000, "train") == 130
    assert refit_iterations(spec, 130, 11000, "refit") == 130
    unscaled = PipelineSpec(model="lightgbm")
    assert refit_iterations(unscaled, 130, 53000, "refit") == 130


def test_local_contract_binds_canonical_roles_from_subdirectories(tmp_path):
    root = tmp_path / "task"
    data = root / "regression" / "participants"
    data.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3, 4], "x": [1.0, 2.0, 3.0, 4.0], "site_eui": [10.0, 20.0, 30.0, 40.0]}).to_csv(
        data / "train.csv", index=False)
    pd.DataFrame({"id": [5, 6], "x": [5.0, 6.0]}).to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": [5, 6], "site_eui": [0, 0]}).to_csv(data / "sample_submission.csv", index=False)
    (root / "task.md").write_text(
        "Задача регрессии.\
Основная метрика — RMSE.\
"
        "Данные: regression/participants/train.csv и regression/participants/test.csv.\
"
        "Целевая переменная: site_eui.\
", encoding="utf-8")
    from automl_agent.data import Dataset
    dataset = Dataset(root / "task.md", deferred=True)
    task = dataset.local_contract()
    assert task is not None
    assert task.kind == "regression" and task.targets == ["site_eui"]
    assert task.metric.name == "rmse" and task.id_columns == ["id"]
    assert dataset.paths["train.csv"].name == "train.csv"


def test_local_contract_refuses_ambiguous_basenames(tmp_path):
    root = tmp_path / "task"
    for relative in ("a/train.csv", "b/train.csv", "a/test.csv"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": [1], "x": [1.0], "y": [1.0]}).to_csv(path, index=False)
    (root / "task.md").write_text("Основная метрика — RMSE.\
Целевая переменная: y.\
train.csv test.csv\
", encoding="utf-8")
    from automl_agent.data import Dataset
    dataset = Dataset(root / "task.md", deferred=True)
    assert dataset._fallback_roles() is None


def test_artifact_scope_keeps_only_digest_for_large_key_lists(tabular):
    X, y, task = tabular
    spec = PipelineSpec(model="ridge")
    context = {"protocol_hash": "protocol", "data_hash": "data", "split_id": "validation"}
    large = artifact_scope(context, spec, task, [f"train:{i}" for i in range(6000)], [f"test:{i}" for i in range(6000)])
    assert "fit_keys" not in large and large["fit_keys_count"] == 6000
    assert "prediction_keys" not in large and large["prediction_keys_count"] == 6000
    assert len(large["fit_keys_hash"]) == 64
    small = artifact_scope(context, spec, task, ["train:0", "train:1"], ["test:0"])
    assert small["fit_keys"] == ["train:0", "train:1"] and small["prediction_keys"] == ["test:0"]


def test_target_encode_early_rows_use_leak_free_prior():
    frame = pd.DataFrame({
        "entity": ["A", "B", "A", "B", "A", "B"],
        "when": [1, 1, 2, 2, 3, 3],
        "answer": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    y = frame[["answer"]].copy()
    task = TaskSpec(kind="regression", targets=["answer"], id_columns=["identifier"],
        feature_columns=["entity", "when"], categorical_columns=["entity"], time_column="when",
        metric=MetricSpec(name="rmse"), output=OutputSpec(columns=["identifier", "answer"], prediction_type="values"),
        source_hash="fixture")
    spec = PipelineSpec(model="lightgbm",
        transformations=[Transformation(name="target_encode", parameters={"columns": ["entity"], "smoothing": 20})])
    generated = "__te_entity_answer"
    features = SupervisedFeatures(spec, task, "temporal").fit_transform(frame, y, frame)
    early = features.loc[features["when"] == 1, generated]
    # The earliest period has no strictly earlier labels, so it falls back to a
    # label-free neutral value instead of a prior computed from its own target.
    assert np.allclose(early.to_numpy(), 0.0)

    changed_future = y.copy()
    changed_future.loc[frame["when"] == 3, "answer"] += 1000
    second = SupervisedFeatures(spec, task, "temporal").fit_transform(frame, changed_future, frame)
    assert np.allclose(second.loc[frame["when"] == 1, generated].to_numpy(), early.to_numpy())

    changed_own = y.copy()
    changed_own.loc[0, "answer"] += 999
    third = SupervisedFeatures(spec, task, "temporal").fit_transform(frame, changed_own, frame)
    assert np.allclose(third.loc[frame["when"] == 1, generated].to_numpy(), early.to_numpy())

    report = SupervisedFeatures(spec, task, "temporal")
    report.fit_transform(frame, y, frame)
    assert report.report[0]["uncovered_prior"] == "strictly_past_period_mean"
    assert report.report[0]["uncovered_fallback"] == 0.0
    assert report.report[0]["uncovered_rows"] == 2

    # A wider history proves that warmup rows use only strictly earlier periods.
    wide = pd.DataFrame({"entity": ["A"] * 8, "when": list(range(1, 9))})
    wide_y = pd.DataFrame({"answer": np.arange(1, 9, dtype=float)})
    encoded = SupervisedFeatures(spec, task, "temporal").fit_transform(wide, wide_y, wide)
    assert np.allclose(encoded[generated].to_numpy()[:5], [0.0, 1.0, 1.5, 2.0, 2.5])


def test_bootstrap_submission_copies_template_before_contract(tmp_path):
    from automl_agent.controller import Controller
    root = tmp_path / "task"
    (root / "regression" / "participants").mkdir(parents=True)
    sample = root / "regression" / "participants" / "sample_submission.csv"
    sample.write_text("id,site_eui\
1,0\
", encoding="utf-8")
    fake = SimpleNamespace(dataset=SimpleNamespace(
            task_path=root / "task.md",
            catalog={"regression/participants/sample_submission.csv": {"columns": ["id", "site_eui"]}}),
        journal=SimpleNamespace(event=lambda *args, **kwargs: None))
    Controller._bootstrap_submission(fake)
    output = root / "submission.csv"
    assert output.read_text(encoding="utf-8") == sample.read_text(encoding="utf-8")
