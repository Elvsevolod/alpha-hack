import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.compatibility import ranking_bootstrap_issue, ranking_label_issue
from automl_agent.contracts import ExecutionPolicy, ExperimentResult, MetricSpec, OutputSpec, PipelineSpec, TaskSpec
from automl_agent.controller import Controller
from automl_agent.pipelines import build_pipeline
from conftest import write_task
from test_pipelines import context


def rank_task(metric="ndcg", feature_group=False, text=False):
    features = ["x", "query"] if feature_group else ["x"]
    if text:
        features.append("words")
    return TaskSpec(kind="ranking", targets=["relevance"], id_columns=["row"],
        feature_columns=features, text_columns=["words"] if text else [], group_column="query",
        metric=MetricSpec(name=metric, k=3),
        output=OutputSpec(columns=["row", "relevance"], prediction_type="scores"),
        source_hash="rank-fixture")


@pytest.mark.parametrize("values, label_gain, expected", [
    ([0, 1, 3], None, None),
    ([0.0, 1.0, 3.0], None, None),
    ([0, 1.5], None, "integer-valued"),
    ([-1, 0], None, "nonnegative"),
    ([0, 31], None, "label_gain"),
    ([0, 4], [0, 1, 3], "label_gain"),
])
def test_ranking_label_compatibility(values, label_gain, expected):
    issue = ranking_label_issue(values, label_gain)
    assert (issue is None) if expected is None else expected in issue


def test_bootstrap_gate_matches_screened_task_contract():
    task = rank_task()
    train = pd.DataFrame({"query": [0, 1, 2, 3], "relevance": [0, 1, 2, 3]})
    group_plan = SimpleNamespace(strategy="group", fit=[0], validation=[1], control=[2, 3])
    random_plan = SimpleNamespace(strategy="random", fit=[0], validation=[1], control=[2, 3])
    assert ranking_bootstrap_issue(task, train, group_plan) is None
    assert "group-preserving" in ranking_bootstrap_issue(task, train, random_plan)
    overlap_plan = SimpleNamespace(strategy="group", fit=[0], validation=[1, 2], control=[3])
    overlap_train = pd.DataFrame({"query": [0, 0, 1, 2], "relevance": [0, 1, 2, 3]})
    assert "query groups must remain disjoint" in ranking_bootstrap_issue(task, overlap_train, overlap_plan)
    assert "NDCG" in ranking_bootstrap_issue(rank_task(metric="map"), train, group_plan)
    multi_target = task.model_copy(update={"targets": ["relevance", "other"]})
    assert "one relevance target" in ranking_bootstrap_issue(
        multi_target, pd.DataFrame({"query": [0, 0, 1, 2], "relevance": [0, 1, 2, 3],
            "other": [1, 2, 3, 4]}), group_plan)
    assert "text" in ranking_bootstrap_issue(rank_task(text=True), train, group_plan)
    assert "group identifiers" in ranking_bootstrap_issue(rank_task(feature_group=True), train, group_plan)
    assert "integer-valued" in ranking_bootstrap_issue(
        task, pd.DataFrame({"query": [0, 1, 2, 3], "relevance": [0, 1.5, 2, 3]}), group_plan)


def write_ranking_task(root, explicit_validation=None):
    root.mkdir(parents=True, exist_ok=True)
    groups = np.repeat(np.arange(24), 10)
    rows = np.arange(len(groups))
    frame = pd.DataFrame({"row": [f"r{i:04d}" for i in rows], "query": groups,
        "x": np.sin(rows / 3), "relevance": np.tile([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], 24)})
    train = frame.iloc[:200]
    test = frame.iloc[200:].drop(columns="relevance")
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    pd.DataFrame({"row": test["row"], "relevance": 0}).to_csv(root / "sample_submission.csv", index=False)
    spec = rank_task().model_copy(update={"explicit_validation": explicit_validation})
    (root / "task_spec.json").write_text(json.dumps(spec.model_dump(mode="json", exclude={"source_hash"})))
    protocol = " Use a random validation split." if explicit_validation == "random" else ""
    (root / "task.md").write_text("Synthetic ranking fixture." + protocol + "\n")
    return root / "task.md"


@pytest.mark.parametrize("ridge_status,ranker_status,expected_anchor", [
    ("completed", "completed", "ridge"),
    ("error", "completed", "lgbm_ranker"),
])
def test_prepare_runs_optional_ranker_and_keeps_first_successful_baseline(
        tmp_path, monkeypatch, ridge_status, ranker_status, expected_anchor):
    controller = Controller(write_ranking_task(tmp_path), Limits(seconds=900, cpu=1), run_id="ranker_bootstrap")
    calls = []

    def execute(spec, automatic_pilot=False):
        calls.append(spec)
        status = ridge_status if spec.pipeline.model == "ridge" else ranker_status
        return ExperimentResult(experiment_id=spec.experiment_id, status=status,
            validation_hash=spec.validation_hash, score=0.8 if status == "completed" else None,
            scope=spec.status, created_index=len(calls), error=None if status == "completed" else "fixture failure")

    monkeypatch.setattr(controller.executor, "probe", lambda *args: ExecutionPolicy(
        writable_directories=[], tests={}, limitations=[]))
    monkeypatch.setattr(controller, "execute_experiment", execute)
    monkeypatch.setattr(controller, "accept_result", lambda *args: None)
    try:
        controller.prepare()
        assert [spec.pipeline.model for spec in calls] == ["ridge", "lgbm_ranker"]
        assert calls[0].status == calls[1].status == "baseline"
        anchor, result = controller.lookup(controller.journal.get("baseline_id"))
        assert anchor.pipeline.model == expected_anchor and result.status == (ridge_status if expected_anchor == "ridge" else ranker_status)
        assert controller.journal.get("bootstrap_ranker_id") == calls[1].experiment_id
    finally:
        controller.close()


def test_prepare_uses_whole_query_splits_for_explicit_random_validation(tmp_path, monkeypatch):
    controller = Controller(write_ranking_task(tmp_path, "random"), Limits(seconds=900, cpu=1),
        run_id="ranker_random_split")
    calls = []

    def execute(spec, automatic_pilot=False):
        calls.append(spec)
        return ExperimentResult(experiment_id=spec.experiment_id, status="completed",
            validation_hash=spec.validation_hash, score=0.8, scope=spec.status, created_index=len(calls))

    monkeypatch.setattr(controller.executor, "probe", lambda *args: ExecutionPolicy(
        writable_directories=[], tests={}, limitations=[]))
    monkeypatch.setattr(controller, "execute_experiment", execute)
    monkeypatch.setattr(controller, "accept_result", lambda *args: None)
    try:
        controller.prepare()
        assert controller.plan.strategy == "group"
        parts = [set(controller.dataset.train.iloc[rows]["query"])
            for rows in (controller.plan.fit, controller.plan.validation, controller.plan.control)]
        assert not parts[0] & parts[1]
        assert not parts[0] & parts[2]
        assert not parts[1] & parts[2]
        assert [spec.pipeline.model for spec in calls] == ["ridge", "lgbm_ranker"]
        assert controller.journal.get("bootstrap_ranker_id") == calls[1].experiment_id
    finally:
        controller.close()


def test_prepare_retries_saved_ranker_spec_without_result(tmp_path, monkeypatch):
    task_path = write_ranking_task(tmp_path)
    first = Controller(task_path, Limits(seconds=900, cpu=1), run_id="ranker_resume")

    class SimulatedInterruption(BaseException):
        pass

    def interrupt_after_ranker_launch(spec, automatic_pilot=False):
        if spec.pipeline.model == "lgbm_ranker":
            raise SimulatedInterruption
        return ExperimentResult(experiment_id=spec.experiment_id, status="completed",
            validation_hash=spec.validation_hash, score=0.8, scope=spec.status, created_index=1)

    monkeypatch.setattr(first.executor, "probe", lambda *args: ExecutionPolicy(
        writable_directories=[], tests={}, limitations=[]))
    monkeypatch.setattr(first, "execute_experiment", interrupt_after_ranker_launch)
    monkeypatch.setattr(first, "accept_result", lambda *args: None)
    saved_id = None
    try:
        with pytest.raises(SimulatedInterruption):
            first.prepare()
        saved_id = first.journal.get("bootstrap_ranker_id")
        assert saved_id
        assert first.lookup(saved_id)[1] is None
    finally:
        first.close()

    resumed = Controller(task_path, Limits(seconds=900, cpu=1), run_id="ranker_resume")
    calls = []

    def complete_saved(spec, automatic_pilot=False):
        calls.append(spec)
        return ExperimentResult(experiment_id=spec.experiment_id, status="completed",
            validation_hash=spec.validation_hash, score=0.82, scope=spec.status, created_index=2)

    monkeypatch.setattr(resumed.executor, "probe", lambda *args: ExecutionPolicy(
        writable_directories=[], tests={}, limitations=[]))
    monkeypatch.setattr(resumed, "execute_experiment", complete_saved)
    monkeypatch.setattr(resumed, "accept_result", lambda *args: None)
    try:
        resumed.prepare()
        assert len(calls) == 1
        assert calls[0].experiment_id == saved_id
        assert calls[0].pipeline.model == "lgbm_ranker"
        assert resumed.lookup(saved_id)[1].status == "completed"
    finally:
        resumed.close()


def test_ranker_fit_casts_integral_float_relevance_and_rejects_continuous(tmp_path):
    groups = np.repeat(np.arange(8), 10)
    X = pd.DataFrame({"query": groups, "x": np.linspace(0, 1, len(groups))})
    task = rank_task()
    pipeline = build_pipeline(
        PipelineSpec(model="lgbm_ranker", representation="native", hyperparameters={"n_estimators": 8}),
        task, {"cpu": 1, "memory_mb": 2048})
    y = pd.DataFrame({"relevance": np.tile([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0], 8)})
    fit_context = context(task, tmp_path)
    fit_context["strategy"] = "group"
    assert pipeline.fit(X, y, fit_context).rows == len(X)
    with pytest.raises(ValueError, match="integer-valued"):
        pipeline.fit(X, y.assign(relevance=np.linspace(0.0, 1.0, len(y))), fit_context)
