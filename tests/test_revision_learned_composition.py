import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis, PipelineSpec, ToolCommand
from automl_agent.controller import Controller
from automl_agent.tool_results import payload
from automl_agent.util import atomic_json, file_hash, read_json
from conftest import write_task


H = Hypothesis(observation="Offline split change fixture", expected_effect="Relearn fit-only composition",
    changed_component="composition", acceptance="No old fit labels in revised heldout predictions")


def setup_composition(tmp_path, monkeypatch, strategy, decision="argmax"):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    task = write_task(tmp_path, temporal=True, kind="classification" if decision == "regularized" else "regression")
    for name in ("train", "test"):
        frame = pd.read_csv(tmp_path / f"{name}.csv", dtype={"row": str})
        frame["when"] = np.arange(len(frame)) // 10 + (18 if name == "test" else 0)
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    c = Controller(task, Limits(seconds=360, cpu=1), run_id="revision_composition")
    # This boundary test targets research composition, not the independent
    # rolling stability models (which deliberately use smaller fit windows).
    monkeypatch.setattr(c, "temporal_stability", lambda *a: {"status": "fixture_skipped"})
    c.prepare()
    members = []
    for i, alpha in enumerate((1, 5)):
        trained = payload(c.dispatch(ToolCommand(operation_id=f"member_{i}", tool="train_model", arguments={
            "pipeline": PipelineSpec(model="logistic" if decision == "regularized" else "ridge",
                hyperparameters={"C": alpha} if decision == "regularized" else {"alpha": alpha}).model_dump(),
            "hypothesis": H.model_dump(), "budget_seconds": 25})))
        assert trained["result"]["status"] == "completed", trained
        members.append(trained["experiment"]["experiment_id"])
    blend = payload(c.dispatch(ToolCommand(operation_id="blend", tool="build_ensemble", arguments={
        "members": members, "strategy": strategy, "final_decision": decision,
        "hypothesis": H.model_dump(), "budget_seconds": 70})))
    assert blend["result"]["status"] == "completed", blend
    return c, task, blend["experiment"]["experiment_id"]


@pytest.mark.parametrize("strategy,decision", [("weighted_oof", "argmax"), ("stacked_oof", "argmax"),
    ("uniform", "regularized")])
def test_revision_relearns_composition_on_new_fit(tmp_path, monkeypatch, strategy, decision):
    import automl_agent.learned_blend as lb
    c, _, blend_id = setup_composition(tmp_path, monkeypatch, strategy, decision)
    old_fit, old_control = set(c.plan.fit), c.plan.control[:]
    prior, _ = c.lookup(blend_id)
    member_spec, _ = c.lookup(prior.pipeline.members[0])
    old_oof = lb.oof_directory(c, member_spec, c.plan.fit, "research")
    old_meta_keys = set(read_json(old_oof / "oof_predictions.json")["row_keys"])
    calls = []
    function = "fit_final_layer" if decision == "regularized" else "fit_blend"
    original = getattr(lb, function)
    def capture(task, reports, y, *args):
        expected = [c.dataset.row_keys[i] for i in c.plan.fit]
        assert y.index.tolist() == expected
        assert all(set(report["row_keys"]) <= set(expected) for report in reports)
        calls.append(expected)
        return original(task, reports, y, *args)
    monkeypatch.setattr(lb, function, capture)
    try:
        revision = payload(c.dispatch(ToolCommand(operation_id="revise", tool="revise_validation", arguments={
            "proposed_strategy": "random", "rationale": "Offline fit-only composition test"})))
        assert revision["revision_status"] == "completed", revision
        assert old_fit.intersection(c.plan.validation), "Fixture must expose the old-fit/new-validation overlap"
        assert old_meta_keys.intersection(c.dataset.row_keys[i] for i in c.plan.validation)
        assert c.plan.control == old_control and not c.journal.get("control_opened")
        assert len(calls) == 1, "Composition must be learned again inside revised fit"
        spec, result = c.lookup(c.journal.get("protocol_revision")["replacements"][blend_id])
        assert spec.pipeline.ensemble_strategy == strategy
        assert bool(spec.pipeline.learned_decision) == (decision == "regularized")
        assert bool(spec.pipeline.ensemble_meta) == (strategy == "stacked_oof")
        assert (spec.pipeline.ensemble_weights is not None) == (strategy == "weighted_oof")
        assert result.training_summary["research_oof_seconds"] > 0
        raw = json.loads((c.workdir / "attempts" / spec.experiment_id / "worker_result.json").read_text())
        assert raw["frozen_pipeline"] == spec.pipeline.model_dump(mode="json")
    finally:
        c.close()


@pytest.mark.parametrize("point", ["known_oof", "unknown_oof", "ready", "after_compose"])
def test_revision_composition_recovery(tmp_path, monkeypatch, point):
    import automl_agent.learned_blend as lb
    c, task, blend_id = setup_composition(tmp_path, monkeypatch, "stacked_oof")
    old_hash, old_csv = c.plan.hash, c.publisher.output.read_bytes()
    original_oof, original_run, original_put = lb.member_oof, c.executor.run, c.journal.put
    original_execute = c.execute_experiment
    interrupted = []
    def member(controller, identifier, seconds, **kwargs):
        report = original_oof(controller, identifier, seconds, **kwargs)
        if controller.plan.version > 1 and not interrupted:
            interrupted.append(identifier)
            raise KeyboardInterrupt()
        return report
    def run(directory, request, seconds):
        if request.get("mode") == "oof" and c.plan.version > 1 and not interrupted:
            interrupted.append(str(directory))
            atomic_json(directory / "request.json", request)
            raise KeyboardInterrupt()
        return original_run(directory, request, seconds)
    def put(key, value):
        original_put(key, value)
        if key == "protocol_revision" and not interrupted:
            if any(item["status"] == "ready" and item.get("composition_refit") for item in value["candidates"]):
                interrupted.append("ready")
                raise KeyboardInterrupt()
    def execute(spec, **kwargs):
        result = original_execute(spec, **kwargs)
        if spec.pipeline.engine == "ensemble" and not interrupted:
            interrupted.append(spec.experiment_id)
            raise KeyboardInterrupt()
        return result
    if point == "known_oof": monkeypatch.setattr(lb, "member_oof", member)
    elif point == "unknown_oof": monkeypatch.setattr(c.executor, "run", run)
    elif point == "ready": monkeypatch.setattr(c.journal, "put", put)
    else: monkeypatch.setattr(c, "execute_experiment", execute)
    with pytest.raises(KeyboardInterrupt):
        c.dispatch(ToolCommand(operation_id="revise", tool="revise_validation", arguments={
            "proposed_strategy": "random", "rationale": "Offline composition recovery"}))
    before_count = c.budget.counter("candidate_count")
    deadline = c.budget.deadline
    c.close()
    monkeypatch.undo()
    resumed = Controller(task, Limits(seconds=999, cpu=1), run_id="revision_composition")
    monkeypatch.setattr(resumed, "temporal_stability", lambda *a: {"status": "fixture_skipped"})
    executed = []
    resumed_run = resumed.executor.run
    def track(directory, request, seconds):
        executed.append(str(directory))
        return resumed_run(directory, request, seconds)
    monkeypatch.setattr(resumed.executor, "run", track)
    try:
        resumed.prepare()
        operation = resumed.journal.get("protocol_revision")
        assert resumed.budget.counter("candidate_count") == before_count
        assert resumed.budget.deadline == deadline
        if point == "unknown_oof":
            assert operation["status"] == "aborted" and "unknown" in operation["reason"]
            assert operation["candidates"][-1]["status"] == "outcome_unknown"
            assert resumed.journal.operation("revise").result["data"]["revision_status"] == "outcome_unknown"
            assert resumed.plan.hash == old_hash and resumed.publisher.output.read_bytes() == old_csv
            assert not executed
        else:
            assert operation["status"] == "completed", operation.get("reason")
            assert resumed.plan.hash != old_hash and not resumed.journal.get("control_opened")
            if point == "known_oof":
                assert len(executed) == 1, "Only the second uncached OOF member should run"
            else:
                assert not executed, "Persisted fitted composition must not run workers again"
            spec, result = resumed.lookup(operation["replacements"][blend_id])
            assert spec.pipeline.ensemble_meta
            assert result.artifacts["worker_manifest_sha256"] == file_hash(Path(result.artifacts["worker_manifest"]))
            raw = read_json(Path(result.artifacts["worker_manifest"]))
            assert result.training_summary["research_oof_seconds"] == raw["training_summary"]["research_oof_seconds"] > 0
            assert result.seconds >= result.training_summary["research_oof_seconds"]
    finally:
        resumed.close()
