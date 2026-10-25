"""Focused regressions for the v088 independent review findings."""
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import MetricSpec, OutputSpec, PipelineSpec, TaskSpec, Transformation
from automl_agent.controller import Controller
from automl_agent.data import Dataset, locate_csv
from automl_agent.pipelines import DirectPipeline
from automl_agent.supervised_features import SupervisedFeatures
from automl_agent.util import atomic_json
from conftest import write_task


def test_clustering_negation_targets_the_task_not_its_labels():
    from automl_agent.clustering_route import _affirmative_mention

    def affirmative(text):
        return any(_affirmative_mention(text, match)
            for match in re.finditer(r'\bclustering\b|\bкластеризац\w*', text, re.I))

    assert affirmative("Perform clustering without labels. Use two clusters.")
    assert affirmative("This is not a clustering task. Perform clustering without labels.")
    assert affirmative("Выполнить кластеризацию без меток.")
    assert affirmative("Выполните кластеризацию, не используя метки.")
    assert not affirmative("This is not a clustering task.")
    assert not affirmative("Do not perform clustering.")
    assert not affirmative("Кластеризация не требуется.")


def test_locate_csv_extracts_explicit_paths_without_leading_prose(tmp_path):
    root = tmp_path / "task"
    for folder in ("primary", "auxiliary"):
        (root / folder).mkdir(parents=True)
        pd.DataFrame({"id": [1, 2], "x": [1.0, 2.0], "y": [1.0, 2.0]}).to_csv(root / folder / "train.csv", index=False)
        pd.DataFrame({"id": [3], "x": [3.0]}).to_csv(root / folder / "test.csv", index=False)
    # Two ambiguous train basenames: only the cited path resolves them.
    (root / "task.md").write_text(
        "Primary metric: RMSE.\nTarget: y.\nUse primary/train.csv and primary/test.csv.\n", encoding="utf-8")
    dataset = Dataset(root / "task.md", deferred=True)
    assert dataset._fallback_roles() == {"train.csv": "primary/train.csv", "test.csv": "primary/test.csv"}
    located = locate_csv(root / "task.md", (root / "task.md").read_text(), "train.csv")
    assert located.resolve() == (root / "primary/train.csv").resolve()


def test_template_is_published_before_inventory_and_bad_periphery_is_recorded(tmp_path):
    path = write_task(tmp_path)
    (tmp_path / "unused.csv").write_text("aux,aux\n1,2\n")
    sample = (tmp_path / "sample_submission.csv").read_bytes()
    (tmp_path / "submission.csv").unlink(missing_ok=True)
    controller = Controller(path, Limits(seconds=30, cpu=1), run_id="periphery")
    try:
        assert (tmp_path / "submission.csv").read_bytes() == sample
        assert "unused.csv" in controller.dataset.inventory_errors
        kinds = {row[0] for row in controller.journal.db.execute("SELECT kind FROM events")}
        assert "bootstrap_publication" in kinds and "csv_inventory_warnings" in kinds
    finally:
        controller.close()


def test_existing_submission_template_is_never_overwritten(tmp_path):
    path = write_task(tmp_path)
    marker = b"existing,output\n1,2\n"
    (tmp_path / "submission.csv").write_bytes(marker)
    controller = Controller(path, Limits(seconds=30, cpu=1), run_id="existing")
    try:
        assert (tmp_path / "submission.csv").read_bytes() == marker
        assert not controller.bootstrap_source
    finally:
        controller.close()


def test_frozen_spec_scales_iterations_only_for_a_frozen_schedule(tabular):
    X, y, task = tabular
    context = {"strategy": "random", "cpu": 1, "budget_seconds": 30, "memory_mb": 1024,
        "protocol_hash": "fixture", "data_hash": "fixture", "execution_mode": "train",
        "task": task.model_dump(mode="json"), "split_id": "fixture"}
    plain = DirectPipeline(PipelineSpec(model="ridge"), task)
    plain.fit(X, y, context)
    assert plain.frozen_spec().iteration_scale_rows is None

    scheduled = DirectPipeline(PipelineSpec(model="lightgbm", iteration_schedule={task.targets[0]: 3}), task)
    scheduled.fit(X, y, context)
    frozen = scheduled.frozen_spec()
    assert frozen.iteration_schedule == {task.targets[0]: 3}
    assert frozen.iteration_scale_rows == len(X)


def test_regularized_inner_holdout_uses_the_selected_decision_algorithm():
    rng = np.random.default_rng(17)
    X = pd.DataFrame({"x": rng.normal(size=240), "z": rng.normal(size=240)},
        index=[f"train:{i}" for i in range(240)])
    y = pd.DataFrame({"y": np.asarray(["a", "b", "c"])[np.arange(240) % 3]}, index=X.index)
    task = TaskSpec(kind="classification", targets=["y"], id_columns=["id"], feature_columns=list(X),
        classes={"y": ["a", "b", "c"]}, metric=MetricSpec(name="f1", average="macro"),
        output=OutputSpec(columns=["id", "y"], prediction_type="labels"), source_hash="fixture")
    context = {"strategy": "random", "cpu": 1, "budget_seconds": 30, "memory_mb": 2048,
        "protocol_hash": "fixture", "data_hash": "fixture", "execution_mode": "train",
        "task": task.model_dump(mode="json"), "split_id": "fixture"}
    states = {}
    for method in ("legacy", "regularized"):
        spec = PipelineSpec(model="logistic", prediction_policy="class_offsets",
            postprocess_fit_policy="inner_holdout", decision_method=method)
        pipeline = DirectPipeline(spec, task)
        pipeline.fit(X, y, context)
        states[method] = pipeline.postprocessors["y"]
    assert states["legacy"]["method"] == "legacy"
    assert states["regularized"]["method"] == "regularized"
    assert states["regularized"]["scope"] == "fit_only_inner_holdout_decision_training"
    for state in states.values():
        assert {"offsets", "support", "penalty", "method"} <= set(state)
        assert np.isfinite(state["offsets"]).all()
    assert np.isclose(sum(states['regularized']['offsets']),0.)
    # Legacy preserves v076's uncentered coordinates and constant penalty.
    assert np.allclose(states['legacy']['penalty'],.001)


def test_resource_stop_is_not_masked_by_a_log_error(tmp_path, monkeypatch):
    from automl_agent import executor as module
    from automl_agent.journal import Journal

    root = tmp_path / "attempt"
    root.mkdir()
    (root / "verified.bin").write_bytes(b"verified")
    journal = Journal(root / "journal")
    executor = module.Executor(1, 2048, __import__("time").time() + 60, journal)

    def popen(*args, **kwargs):
        return SimpleNamespace(pid=87654321, stdout=None, returncode=0, poll=lambda: None, wait=lambda **k: 0)

    class Process:
        def __init__(self, *args):
            pass
        def create_time(self):
            return 1.
        def children(self, **kwargs):
            return []
        def memory_info(self):
            return SimpleNamespace(rss=1)
        def is_running(self):
            return True

    class Output:
        def __init__(self, *args):
            self.error = "log_limit"
            self.total = 1
        def finish(self):
            pass

    def write_result(name, *args, **kwargs):
        atomic_json(root / "worker_result.json",
            {"status": "completed", "hashes": {"verified.bin": hashlib.sha256(b"verified").hexdigest()}})

    try:
        with monkeypatch.context() as patch:
            patch.setattr(module.subprocess, "Popen", popen)
            patch.setattr(module.psutil, "Process", Process)
            patch.setattr(module, "RotatingOutput", Output)
            patch.setattr(module, "terminate_tree", lambda *a: None)
            patch.setattr(module.os, "killpg", lambda *a: None)
            patch.setattr(module.shutil, "disk_usage", lambda *a: SimpleNamespace(free=0))
            patch.setattr(module, "snapshot", lambda *a: None)
            patch.setattr(module, "emit", write_result)
            result = executor.run(root, {"mode": "train"}, 10)
        diagnostic = json.loads((root / "diagnostic_process_result.json").read_text())
        assert result["status"] == "disk_limit"
        assert diagnostic["reason"] == "disk_limit" and diagnostic["log_error"] == "log_limit"
    finally:
        journal.close()


def test_autogluon_selection_refit_trains_on_every_permitted_row(tmp_path, monkeypatch):
    pytest.importorskip("autogluon.tabular")
    from autogluon.tabular.models.lgb.lgb_model import LGBModel
    from automl_agent.search import SearchPipeline

    history = []
    original = LGBModel._fit

    def watch(self, X, y, *args, **kwargs):
        history.append(len(X))
        return original(self, X, y, *args, **kwargs)

    monkeypatch.setattr(LGBModel, "_fit", watch)
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"x": np.arange(80, dtype=float)}, index=[f"train:{i}" for i in range(80)])
    y = pd.DataFrame({"answer": rng.normal(size=80)}, index=X.index)
    task = TaskSpec(kind="regression", targets=["answer"], id_columns=["id"], feature_columns=["x"],
        metric=MetricSpec(name="rmse"), output=OutputSpec(columns=["id", "answer"], prediction_type="values"),
        source_hash="fixture")
    spec = PipelineSpec(engine="autogluon", model="library_search", learners=["GBM"], n_trials=None,
        autogluon_configurations={"GBM": [{"n_estimators": 2, "min_data_in_leaf": 1}]})
    pipeline = SearchPipeline(spec, task)
    context = {"strategy": "random", "cpu": 1, "budget_seconds": 30, "memory_mb": 2048,
        "budget_mode": "full_comparison", "workdir": str(tmp_path)}
    pipeline.fit(X.iloc[:60], y.iloc[:60], context)
    pipeline.refit(X, y, {**context, "execution_mode": "refit"})
    assert pipeline.selected["answer"]["full_refit_rows"] == len(X)
    assert pipeline.selected["answer"]["full_refit_added_rows"] == len(X) - len(pipeline.autogluon_train_keys["answer"])
    assert history[-1] == len(X)


def test_target_encoding_never_reads_its_own_or_future_labels():
    frame = pd.DataFrame({"entity": ["A", "B"] * 3, "when": [1, 1, 2, 2, 3, 3]})
    y = pd.DataFrame({"answer": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
    task = TaskSpec(kind="regression", targets=["answer"], id_columns=["id"], feature_columns=list(frame),
        categorical_columns=["entity"], time_column="when", metric=MetricSpec(name="rmse"),
        output=OutputSpec(columns=["id", "answer"], prediction_type="values"), source_hash="fixture")
    spec = PipelineSpec(model="lightgbm",
        transformations=[Transformation(name="target_encode", parameters={"columns": ["entity"]})])
    column = "__te_entity_answer"
    first = SupervisedFeatures(spec, task, "temporal").fit_transform(frame, y, frame)
    changed = y.copy()
    changed.loc[frame["when"] == 3, "answer"] += 1000
    second = SupervisedFeatures(spec, task, "temporal").fit_transform(frame, changed, frame)
    assert np.allclose(first.loc[frame["when"] == 1, column].to_numpy(), 0.0)
    assert np.array_equal(first[column].to_numpy(), second[column].to_numpy())
