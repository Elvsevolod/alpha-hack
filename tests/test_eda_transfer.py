import json
from pathlib import Path
from types import SimpleNamespace
import time

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import DatasetReportArgs, InspectArgs, MetricSpec, OutputSpec, PipelineSpec
from automl_agent.controller import Controller, SYSTEM
from automl_agent.data import Dataset, _explicit_validation_supported
from automl_agent.executor import stage_frame
from automl_agent.pipelines import DirectPipeline
from automl_agent.profiling import build_profile, correlations, profile_key
from automl_agent.util import canonical, digest, read_json
from conftest import write_task
from test_pipelines import context


@pytest.fixture
def dataset(tmp_path):
    d = Dataset(write_task(tmp_path))
    d.activate(d.explicit())
    d.profile_mode = "unified"
    return d


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    c = Controller(write_task(tmp_path_factory.mktemp("eda_transfer")), Limits(seconds=600, cpu=1), run_id="eda_transfer")
    c.no_llm = True
    c.prepare()
    yield c
    assert c.budget.counter("llm_calls") == 0
    c.close()


@pytest.mark.parametrize("text,expected", [
    ("Validation must use disjoint groups", True), ("Validation must use disjoint groups.", True),
    ("Random split is allowed", False), ("The data contain disjoint groups", False),
    ("Validation must not use disjoint groups", False), ("For example: Validation must use disjoint groups", False),
])
def test_disjoint_groups_requires_a_direct_prescription(text, expected):
    assert _explicit_validation_supported("group", text) is expected


@pytest.mark.parametrize("kind", ["regression", "classification", "ranking", "schedule"])
def test_punctuation_aliases_preserve_predictions_roles_and_serialization(tabular, tmp_path, kind):
    X, y, task = tabular
    renamed = {"number": "a[b]", "second": "c,d", "category": 'q"x'}
    bad = X.rename(columns=renamed)
    updates = {"feature_columns": list(bad), "categorical_columns": ['q"x']}
    model = "lightgbm"
    if kind == "classification":
        y = y.copy(); y["answer"] = np.where(y.answer > 0, "yes", "no")
        updates.update(kind="classification", classes={"answer": ["no", "yes"]}, metric=MetricSpec(name="f1"),
            output=OutputSpec(columns=["answer"], prediction_type="labels"))
    elif kind == "ranking":
        X = X.copy(); X["query"] = np.arange(len(X)) % 12
        bad["query"] = X["query"]
        y = y.copy(); y["answer"] = np.arange(len(X)) % 3
        updates.update(kind="ranking", group_column="query", metric=MetricSpec(name="ndcg"),
            output=OutputSpec(columns=["answer"], prediction_type="scores"))
        model = "lgbm_ranker"
        task = task.model_copy(update={"kind": "ranking", "group_column": "query", "metric": updates["metric"], "output": updates["output"]})
    else:
        updates["kind"] = "regression"
    if kind == "classification":
        task = task.model_copy(update={k: v for k, v in updates.items() if k not in {"feature_columns", "categorical_columns"}})
    bad_task = task.model_copy(update=updates)
    spec = PipelineSpec(model=model, representation="native", hyperparameters={"n_estimators": 12, "min_child_samples": 3},
        early_stopping_rounds=2 if kind == "schedule" else None)
    actual = DirectPipeline(spec, bad_task)
    summary = actual.fit(bad, y, context(bad_task, tmp_path))
    reference = DirectPipeline(spec, task)
    reference.fit(X, y, context(task, tmp_path))
    mapping = summary.preprocessing["feature_name_mapping"]
    assert all(name.startswith("feature_") for name in mapping) and 'q"x' in mapping.values()
    assert summary.preprocessing["feature_name_mapping_hash"] == digest(mapping)
    assert list(bad.columns) == [renamed.get(name, name) for name in X.columns]
    prediction = actual.predict(bad.iloc[:15], context(bad_task, tmp_path))
    expected = reference.predict(X.iloc[:15], context(task, tmp_path))
    assert prediction.prediction_type == expected.prediction_type
    if prediction.prediction_type == "labels":
        assert prediction.values == expected.values
    else:
        np.testing.assert_allclose(prediction.values["answer"], expected.values["answer"])
    restored = cloudpickle.loads(cloudpickle.dumps(actual))
    assert restored.predict(bad.iloc[:15], context(bad_task, tmp_path)).model_dump() == prediction.model_dump()


def test_ranker_receives_positional_categories_aliases_and_sorted_weights(tabular, tmp_path, monkeypatch):
    import lightgbm
    X, y, task = tabular
    X = X.rename(columns={"category": "cat,code"}); X["query"] = np.arange(len(X)) % 12
    task = task.model_copy(update={"kind": "ranking", "feature_columns": ["number", "second", "cat,code", "empty"],
        "categorical_columns": ["cat,code"], "group_column": "query", "metric": MetricSpec(name="ndcg"),
        "output": OutputSpec(columns=["answer"], prediction_type="scores")})
    y = y.copy(); y["answer"] = np.arange(len(y)) % 3
    original = lightgbm.LGBMRanker.fit; received = []
    def wrapped(self, matrix, values, **kwargs):
        received.append(kwargs)
        return original(self, matrix, values, **kwargs)
    monkeypatch.setattr(lightgbm.LGBMRanker, "fit", wrapped)
    c = context(task, tmp_path); c["sample_weights"] = pd.Series(np.arange(len(X)) + 1, index=X.index)
    pipeline = DirectPipeline(PipelineSpec(model="lgbm_ranker", representation="native", hyperparameters={"n_estimators": 4}), task)
    pipeline.fit(X, y, c)
    assert received[0]["feature_name"] and all(type(i) is int for i in received[0]["categorical_feature"])
    order = np.argsort(X["query"].astype(str).to_numpy(), kind="stable")
    np.testing.assert_array_equal(received[0]["sample_weight"], c["sample_weights"].to_numpy()[order])


def test_profile_matches_old_facts_and_only_fit_targets(dataset):
    indices = list(range(100))
    previous = dataset._legacy_profile(indices)
    current = dataset.profile(indices)
    for name, facts in previous.columns.items():
        assert all(current.columns[name][key] == value for key, value in facts.items())
    assert current.targets and current.findings
    dataset.train.loc[100:, "answer"] = 1e100
    assert build_profile(dataset, indices).model_dump() == current.model_dump()
    for scope in ("validation_features", "test_features"):
        document = build_profile(dataset, list(range(100, 150)), scope)
        assert not document.targets and "label_conflicts" not in document.observations[0]["flags"]


def test_profile_cache_is_bounded_tracks_protocol_and_does_not_share_mutations(dataset, monkeypatch):
    import automl_agent.profiling as module
    original = module.build_profile; calls = []
    def build(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "build_profile", build)
    a = dataset.profile(list(range(100)), protocol_hash="a")
    a.columns.clear()
    assert dataset.profile(list(range(100)), protocol_hash="a").columns and len(calls) == 1
    assert profile_key(dataset, [1], "fit", "a") != profile_key(dataset, [1], "fit", "b")
    dataset.profile(list(range(99)), protocol_hash="a")
    dataset.profile(list(range(100)), protocol_hash="b")
    dataset.source_hash += "changed"
    dataset.profile(list(range(100)), protocol_hash="a")
    assert len(calls) == 4
    for i in range(10):
        dataset.profile(list(range(50)), protocol_hash=str(i))
    assert len(dataset._profile_cache) == 8


def test_profile_failure_falls_back_once_to_old_facts(dataset, monkeypatch):
    import automl_agent.profiling as module
    monkeypatch.setattr(module, "build_profile", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("fixture")))
    first = dataset.profile([0, 1, 2])
    assert first.status == "partial" and first.error and first.columns
    assert dataset.profile([0, 1, 2]) == first and dataset.profile_cache_stats["fallbacks"] == 1


def test_tool_and_context_use_the_same_profile_with_three_findings(agent, monkeypatch):
    c = agent; monkeypatch.setattr(c.dataset, "profile_mode", "unified")
    full = c.dataset.profile(c.plan.fit, protocol_hash=c.plan.hash)
    report = c.dataset.inspect(InspectArgs(report="profile", include_target=True), c.plan)
    assert report["columns"] == full.columns and report["targets"] == full.targets
    assert not c.dataset.inspect(InspectArgs(report="profile"), c.plan)["targets"]
    for scope in ("validation_features", "test_features"):
        with pytest.raises(ValueError):
            c.dataset.inspect(InspectArgs(report="profile", scope=scope, include_target=True), c.plan)
    snapshot = c.context_memory.snapshot(SYSTEM)
    summary = json.loads(snapshot.profile_json)["summary"]
    assert len(summary["findings"]) <= 3 and summary["status"] == "available"
    assert not hasattr(c, "eda")


def test_correlations_have_support_and_unknown_instead_of_zero():
    x = np.arange(60, dtype=float)
    frame = pd.DataFrame({"x": x, "linear": 3*x+1, "rank": x**3, "constant": 1., "rare": [1.] * 10 + [np.nan] * 50})
    report = correlations(frame)
    assert report["correlations"]["x"]["linear"] == pytest.approx(1)
    assert report["spearman"]["x"]["rank"] == pytest.approx(1)
    assert report["pair_status"]["x"]["constant"] == "constant"
    assert report["pair_status"]["x"]["rare"] == "insufficient_support"
    assert report["correlations"]["x"]["rare"] is None and report["pair_counts"]["x"]["rare"] == 10
    with pytest.raises(ValueError):
        correlations(pd.DataFrame(np.zeros((20_001, 2))))
    with pytest.raises(ValueError):
        correlations(pd.DataFrame(np.zeros((2, 31))))


def test_supported_correlations_run_in_worker_once_and_respect_scope(agent, monkeypatch):
    c = agent; original = c.executor.run; calls = []
    def run(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(c.executor, "run", run)
    args = DatasetReportArgs(report="correlations", columns=["value", "empty"], correlation_method="supported", max_chars=12000)
    first = c.run_tools.dataset(args)
    assert first["status"] == "success" and first["report"]["pair_status"]["value"]["empty"] == "insufficient_support"
    assert first == c.run_tools.dataset(args) and len(calls) == 1
    assert calls[0]["mode"] == "correlations" and "task" not in calls[0]
    with pytest.raises(ValueError):
        c.run_tools.dataset(args.model_copy(update={"scope": "validation_features", "columns": ["answer"]}))


def test_supported_correlations_failure_is_unknown_and_budget_protects_reserve(agent, monkeypatch):
    c = agent
    monkeypatch.setattr(c.executor, "run", lambda *a, **k: {"status": "timeout"})
    args = DatasetReportArgs(report="correlations", columns=["value"], correlation_method="supported")
    failed = c.run_tools.dataset(args)
    assert failed["status"] == "technical_failure" and failed["relationship_status"] == "unknown" and "correlations" not in failed
    monkeypatch.setattr(c.budget, "deadline", time.time() + .1)
    small = c.run_tools.dataset(args.model_copy(update={"columns": ["empty"]}))
    assert small["status"] == "budget_too_small" and small["relationship_status"] == "unknown"


@pytest.mark.parametrize("rows", [0, 1, 300])
def test_streaming_roundtrip_and_bulk_equivalence(tmp_path, rows):
    from automl_agent.worker import read_frame
    frame = pd.DataFrame({"x": [np.nan if i % 7 == 0 else float(i) for i in range(rows)],
        "s": [None if i % 9 == 0 else 'a,"б' for i in range(rows)]}, index=[f"row:{i}" for i in range(rows)])
    stage_frame(tmp_path / "bulk.json", frame, streaming=False)
    stage_frame(tmp_path / "stream.json", frame, streaming=True, deadline=time.time()+10, chunk_rows=7)
    pd.testing.assert_frame_equal(read_frame(tmp_path / "bulk.json"), read_frame(tmp_path / "stream.json"))
    assert not list(tmp_path.glob(".pending-*"))


def test_interrupted_streaming_never_replaces_previous_file_and_cleans_up(tmp_path):
    path = tmp_path / "frame.json"; path.write_bytes(b"previous verified artifact")
    frame = pd.DataFrame({"x": list(range(129)) + [object()]})
    with pytest.raises(TypeError):
        stage_frame(path, frame, streaming=True)
    assert path.read_bytes() == b"previous verified artifact" and not list(tmp_path.glob(".pending-*"))
    with pytest.raises(TimeoutError):
        stage_frame(path, frame, streaming=True, deadline=time.time()-1)
    with pytest.raises(MemoryError):
        stage_frame(path, frame, streaming=True, memory_mb=.001)
    assert path.read_bytes() == b"previous verified artifact"


def test_readable_report_keeps_dimensions_separate_and_is_optional(agent, monkeypatch):
    import automl_agent.reporting as reporting
    c = agent
    previous = c.publisher.output.read_bytes()
    c.journal.put("failure", "private-target-value-not-for-readable-report")
    report = c.report()
    text = (c.workdir / "REPORT.md").read_text()
    assert set(report["verification_dimensions"]) == {"constraints", "decision_quality", "predictive_quality"}
    assert "unassessed" in report["verification_dimensions"]["decision_quality"]
    assert "private-target-value" not in text and report["readable_report_status"] == "available"
    monkeypatch.setattr(reporting, "write_full_summary", lambda *a: (_ for _ in ()).throw(PermissionError("fixture")))
    failed = c.report()
    assert failed["readable_report_status"] == "unavailable"
    assert read_json(c.workdir / "run_report.json")["readable_report_status"] == "unavailable"
    assert c.publisher.output.read_bytes() == previous


def test_packaging_guard_requires_review_before_nested_sources(tmp_path):
    from tools.build_flat_submission import production_agent_sources
    folder = tmp_path / "automl_agent"; folder.mkdir()
    (folder / "__init__.py").touch(); (folder / "production.py").touch()
    assert len(production_agent_sources(tmp_path)) == 2
    (folder / "eda").mkdir(); (folder / "eda/agent.py").touch()
    with pytest.raises(ValueError, match="explicit production allowlist"):
        production_agent_sources(tmp_path)
    root = Path(__file__).resolve().parents[1]
    assert set(production_agent_sources(root)) == set((root / "automl_agent").rglob("*.py"))


def test_filtered_profile_does_not_poison_full_profile(dataset):
    from automl_agent.validation import choose_validation
    plan = choose_validation(dataset)
    args = InspectArgs(report="profile", include_target=True)
    full = dataset.inspect(args, plan)
    filtered = dataset.inspect(args.model_copy(update={"predicate": {"column": "value", "op": "gt", "value": 0}}), plan)
    assert filtered["rows"] < full["rows"] and filtered["targets"] != full["targets"]
    assert dataset.inspect(args, plan) == full


def test_old_journal_keeps_legacy_profile_and_unknown_versions_require_a_new_run(tmp_path):
    path = write_task(tmp_path)
    c = Controller(path, Limits(seconds=180, cpu=1), run_id="old_profile")
    try:
        c.prepare()
        output = c.publisher.output.read_bytes()
        assert c.dataset.profile_mode == "unified"
        c.journal.db.execute("DELETE FROM metadata WHERE key='profile_analyzer'"); c.journal.db.commit()
    finally:
        c.close()
    c = Controller(path, Limits(seconds=180, cpu=1), run_id="old_profile")
    try:
        c.prepare()
        assert c.dataset.profile_mode == "legacy" and c.publisher.output.read_bytes() == output
        c.journal.put("profile_analyzer", {"mode": "unified", "version": "unknown"})
        with pytest.raises(ValueError, match="frozen runtime or use a new run_id"):
            c.prepare()
        assert c.publisher.output.read_bytes() == output
    finally:
        c.close()


def test_correlation_sample_is_bounded_repeatable_and_independent_of_targets(agent, monkeypatch):
    c = agent
    frame = pd.concat([c.dataset.train] * 140, ignore_index=True)
    monkeypatch.setattr(c.dataset, "train", frame)
    monkeypatch.setattr(c, "plan", c.plan.model_copy(update={"fit": list(range(len(frame)))}))
    monkeypatch.setattr(c, "finalist_costs", lambda: [])
    monkeypatch.setattr(c.budget, "final_reserve", lambda *a, **k: 0)
    args = DatasetReportArgs(report="correlations", columns=["value", "empty"], correlation_method="supported", max_chars=12000)
    first = c.run_tools.dataset(args)["report"]
    assert first["rows"] == 20_000 < first["original_rows"]
    frame["answer"] = 1e100
    second = c.run_tools.dataset(args)["report"]
    assert first == second and first["sampled_row_hash"] == second["sampled_row_hash"]


def test_streaming_checks_memory_between_chunks(tmp_path, monkeypatch):
    import automl_agent.executor as module
    path = tmp_path / "frame.json"; path.write_bytes(b"old")
    calls = []
    class Process:
        def memory_info(self):
            calls.append(1)
            return SimpleNamespace(rss=100 if len(calls) < 4 else 10**10)
    monkeypatch.setattr(module.psutil, "Process", Process)
    with pytest.raises(MemoryError):
        stage_frame(path, pd.DataFrame({"x": range(400)}), streaming=True, memory_mb=1, chunk_rows=128)
    assert len(calls) == 4 and path.read_bytes() == b"old" and not list(tmp_path.glob(".pending-*"))
