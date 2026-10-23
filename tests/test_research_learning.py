"""Cross-task checks for source-led research and conditional experiment reviews."""
import json

import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import ExperimentResult, ExperimentSpec, Hypothesis, MetricSpec, PipelineSpec, SelectionPolicy
from automl_agent.controller import Controller, SYSTEM
from automl_agent.research import RESEARCH_POLICY, experiment_review
from conftest import write_task


def trial(identifier, score, **changes):
    spec = ExperimentSpec(experiment_id=identifier,
        hypothesis=Hypothesis(observation="Fit errors remain", expected_effect="Lower held-out error",
            changed_component="regularization", acceptance="Beat a fixed comparator without exceeding cost"),
        pipeline=PipelineSpec(model="ridge"), task_hash="task", data_version="data",
        validation_hash="split", library_versions={"scikit-learn": "1.5.2"}, budget_seconds=10)
    result = ExperimentResult(experiment_id=identifier, status="completed", validation_hash="split", score=score)
    return spec.model_copy(update=changes), result


@pytest.mark.parametrize("metric,reference,score,expected", [
    ("rmse", 10, 9, "improved_locally"), ("rmse", 10, 11, "no_material_gain"),
    ("mae", 10, 9.9995, "no_material_gain"), ("log_loss", .8, .7, "improved_locally"),
    ("f1", .7, .8, "improved_locally"), ("accuracy", .8, .7, "no_material_gain"),
    ("r2", -.3, -.2, "improved_locally"), ("ndcg", .7, .8, "improved_locally")])
def test_review_respects_primary_metric_direction_and_material_gain(metric, reference, score, expected):
    from types import SimpleNamespace
    task = SimpleNamespace(metric=MetricSpec(name=metric))
    s, r = trial("new", score)
    review = experiment_review(s, r, [trial("old", reference)], task, SelectionPolicy(maximize=task.metric.maximize))
    assert review["outcome"] == expected
    assert review["comparison"]["acceptance"] == "requires_hypothesis_review"
    assert review["transfer_status"] == "task_local_requires_revalidation"
    assert not review["comparison"]["causality_established"]


@pytest.mark.parametrize("change", [{"task_hash": "other"}, {"data_version": "other"},
    {"validation_hash": "other"}, {"library_versions": {"scikit-learn": "other"}}])
def test_review_does_not_compare_different_protocol_data_or_environment(change):
    from types import SimpleNamespace
    task = SimpleNamespace(metric=MetricSpec(name="rmse"))
    s, r = trial("new", 1)
    old = trial("old", 100, **change)
    review = experiment_review(s, r, [old], task, SelectionPolicy(maximize=False))
    assert review["outcome"] == "not_comparable" and review["comparison"]["gain"] is None


@pytest.mark.parametrize("status,scope,outcome", [("timeout", "full", "technical_failure"),
    ("memory_limit", "full", "technical_failure"), ("interrupted", "full", "unresolved"),
    ("outcome_unknown", "full", "unresolved"), ("completed", "pilot", "pilot_only")])
def test_failure_and_pilot_never_become_negative_quality_evidence(status, scope, outcome):
    from types import SimpleNamespace
    task = SimpleNamespace(metric=MetricSpec(name="rmse"))
    s, r = trial("new", 100)
    review = experiment_review(s, r.model_copy(update={"status": status, "scope": scope}),
        [trial("old", 1)], task, SelectionPolicy(maximize=False))
    assert review["outcome"] == outcome and review["comparison"]["gain"] is None


def test_parent_comparison_is_explicit_and_pilots_are_not_benchmarks():
    from types import SimpleNamespace
    task = SimpleNamespace(metric=MetricSpec(name="rmse"))
    s, r = trial("new", 5, parent_id="parent")
    p, pr = trial("pilot", .01)
    review = experiment_review(s, r, [trial("parent", 6), trial("incumbent", 4),
        (p, pr.model_copy(update={"scope": "pilot"}))], task, SelectionPolicy(maximize=False))
    assert review["comparison"]["experiment_id"] == "parent"
    assert review["comparison"]["basis"] == "parent" and review["comparison"]["gain"] == 1
    assert review["outcome"] == "improved_parent_only"
    assert review["comparison"]["incumbent_id"] == "incumbent"


def test_review_never_copies_private_feedback_predictions_or_inner_labels():
    from types import SimpleNamespace
    task = SimpleNamespace(metric=MetricSpec(name="rmse"))
    s, r = trial("new", 5)
    r = r.model_copy(update={"feedback": {"hidden": "PRIVATE_SENTINEL"},
        "artifacts": {"predictions": "PRIVATE_SENTINEL"},
        "training_summary": {"inner_labels": "PRIVATE_SENTINEL", "rows": 123, "auxiliary_fits": 2}})
    review = experiment_review(s, r, [trial("old", 6)], task, SelectionPolicy(maximize=False))
    assert "PRIVATE_SENTINEL" not in json.dumps(review)
    assert review["measurement"]["fit_rows"] == 123 and review["measurement"]["auxiliary_fits"] == 2


def test_recovered_outcome_creates_a_new_review_and_preserves_old_evidence(tmp_path):
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1), run_id="review_recovery")
    try:
        c.prepare()
        baseline, measured = c.lookup(c.journal.get("baseline_id"))
        spec = baseline.model_copy(update={"experiment_id": "recovery", "status": "full"})
        unknown = measured.model_copy(update={"experiment_id": "recovery", "status": "outcome_unknown", "score": None, "scope": "full"})
        c.journal.save_experiment(spec, unknown)
        c.context_memory.index_experiments()
        first = c.context_memory.latest_research_review
        done = unknown.model_copy(update={"status": "completed", "score": measured.score + 100})
        c.journal.save_experiment(spec, done)
        c.context_memory.index_experiments()
        second = c.context_memory.latest_research_review
        assert first != second and c.context_memory.store.record(first)
        from automl_agent.contracts import RunReadArgs
        for identifier, outcome in [(first, "unresolved"), (second, "no_material_gain")]:
            detail = c.run_tools.read(RunReadArgs(ids=[identifier], max_chars=12000))
            assert json.loads(detail["records"][0]["content"])["outcome"] == outcome
    finally:
        c.close()


@pytest.mark.parametrize("kind,temporal,groups", [("regression", False, False),
    ("classification", False, False), ("regression", True, True)])
@pytest.mark.parametrize("mode", ["native", "json"])
def test_review_is_durable_discoverable_and_survives_wire_compression(tmp_path, monkeypatch, kind, temporal, groups, mode):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fixture.invalid/v1")
    path = write_task(tmp_path, kind=kind, temporal=temporal, groups=groups)
    c = Controller(path, Limits(seconds=180, cpu=1), run_id="research_learning")
    try:
        c.prepare()
        snapshot = c.context_memory.snapshot(SYSTEM)
        identifier = c.context_memory.latest_research_review
        card = c.context_memory.store.record(identifier)
        assert card["component"] == "research_review"
        files = list((c.workdir / "research_reviews").glob("*.json"))
        assert files
        review = json.loads(files[-1].read_text())
        assert review["hypothesis"]["source"] == "agent_assumption"
        assert review["reproduction"]["validation_hash"] == c.plan.hash
        before = {p.name: p.read_bytes() for p in files}
        c.context_memory.snapshot(SYSTEM)
        assert {p.name: p.read_bytes() for p in (c.workdir / "research_reviews").glob("*.json")} == before
        for stage in range(4):
            prepared = c.context_memory.compressor.prepare(snapshot, mode, budget=262144, minimum_stage=stage)
            wire = json.loads(prepared.wire_json)
            actual = json.loads(wire["messages"][1]["content"])
            assert identifier in actual["core"]["current_work"]["related_ids"]
            assert RESEARCH_POLICY in wire["messages"][0]["content"]
            assert c.knowledge.audit(prepared, wire)["send_allowed"]
        # Read the review with the same public tools available to Qwen.
        from automl_agent.contracts import RunReadArgs
        detail = c.run_tools.read(RunReadArgs(ids=[identifier], max_chars=12000))
        assert "task_local_requires_revalidation" in json.dumps(detail)
    finally:
        c.close()
    resumed = Controller(path, Limits(seconds=180, cpu=1), run_id="research_learning")
    try:
        resumed.prepare()
        resumed.context_memory.snapshot(SYSTEM)
        assert resumed.context_memory.latest_research_review == identifier
        assert {p.name: p.read_bytes() for p in (resumed.workdir / "research_reviews").glob("*.json")} == before
    finally:
        resumed.close()
