from automl_agent.tool_results import payload as tool_payload
import json
from pathlib import Path
import time

import httpx
import numpy as np
import pytest

from automl_agent import progress
from automl_agent.budget import Budget, Limits
from automl_agent.cleanup import protected_ids, recover_cleanup, staging
from automl_agent.completion import allocation, estimate
from automl_agent.contracts import ExperimentResult, Hypothesis, InspectArgs, PipelineSpec, ToolCommand, TrainArgs
from automl_agent.controller import Controller, SYSTEM
from automl_agent.qwen import QwenContextError
from automl_agent.search import SearchPipeline
from automl_agent.util import atomic_bytes, digest, file_hash
from conftest import write_task

H = Hypothesis(observation="Controlled regression evidence", expected_effect="Verify recovery contract",
    changed_component="fixture", acceptance="Preserve IDs, hashes, deadline and verified CSV")


@pytest.mark.parametrize("point", ["before_commit", "after_commit", "after_replace"])
def test_protocol_crash_recovery_does_not_retrain_confirmed_work(tmp_path, monkeypatch, point):
    task = write_task(tmp_path, temporal=True)
    c = Controller(task, Limits(seconds=180, cpu=1), run_id="revision")
    c.prepare()
    old = c.plan.hash
    deadline = c.budget.deadline
    candidate_count = c.budget.counter("candidate_count")
    old_many = c.journal.put_many
    old_recover = c.publisher.recover
    def put_many(values, event=None):
        if "protocol_revision" in values:
            if point == "before_commit":
                raise KeyboardInterrupt()
            old_many(values, event)
            if point == "after_commit":
                raise KeyboardInterrupt()
        else:
            old_many(values, event)
    monkeypatch.setattr(c.journal, "put_many", put_many)
    if point == "after_replace":
        import automl_agent.publication as publication
        original = publication.atomic_bytes
        def replace(path, content, mode=None):
            original(path, content, mode)
            if path.name == "submission.csv":
                raise KeyboardInterrupt()
        monkeypatch.setattr(publication, "atomic_bytes", replace)
    with pytest.raises(KeyboardInterrupt):
        c.dispatch(ToolCommand(operation_id="revision_v2", tool="revise_validation", arguments={"proposed_strategy": "random", "rationale": "fixed control"}))
    assert c.budget.counter("candidate_count") == candidate_count + 1
    c.close()
    monkeypatch.undo()
    resumed = Controller(task, Limits(seconds=999, cpu=1), run_id="revision")
    try:
        monkeypatch.setattr(resumed, "execute_experiment", lambda *a, **k: pytest.fail("Confirmed computation retrained"))
        resumed.prepare()
        assert resumed.plan.hash != old and resumed.budget.deadline == deadline
        assert resumed.journal.get("protocol_revision")["status"] == "completed"
        assert resumed.journal.operation("revision_v2").result["data"]["revision_status"] == "completed"
        assert resumed.journal.get("publication")["validation_hash"] == resumed.plan.hash
        assert file_hash(resumed.publisher.output) == resumed.journal.get("publication")["csv_sha256"]
        assert resumed.budget.counter("candidate_count") == candidate_count + 1
    finally:
        resumed.close()


@pytest.mark.parametrize("point", ["after_worker", "between_candidates", "unknown_outcome"])
def test_protocol_mid_operation_recovery_preserves_confirmed_work_or_aborts_unknown(tmp_path, monkeypatch, point):
    path = write_task(tmp_path, temporal=True)
    c = Controller(path, Limits(seconds=180, cpu=1), run_id="mid_revision")
    c.prepare()
    trained = tool_payload(c.dispatch(ToolCommand(operation_id="additional", tool="train_model", arguments={
        "pipeline": PipelineSpec(model="ridge", hyperparameters={"alpha": 2}).model_dump(),
        "hypothesis": H.model_dump(), "budget_seconds": 10})))
    assert trained["result"]["status"] == "completed"
    original_plan, deadline = c.plan.hash, c.budget.deadline
    baseline, leader = c.journal.get("baseline_id"), c.journal.get("leader_id")
    prior_csv, count = c.publisher.output.read_bytes(), c.budget.counter("candidate_count")
    execute, put = c.execute_experiment, c.journal.put
    def interrupted(spec, **kwargs):
        if point == "unknown_outcome":
            raise KeyboardInterrupt()
        result = execute(spec, **kwargs)
        if point == "after_worker":
            raise KeyboardInterrupt()
        return result
    def between(key, value):
        put(key, value)
        if key == "protocol_revision" and point == "between_candidates" and value["candidates"][0]["status"] == "completed" and value["candidates"][1]["status"] == "pending":
            raise KeyboardInterrupt()
    monkeypatch.setattr(c, "execute_experiment", interrupted)
    monkeypatch.setattr(c.journal, "put", between)
    with pytest.raises(KeyboardInterrupt):
        c.dispatch(ToolCommand(operation_id="revision_v2", tool="revise_validation", arguments={"proposed_strategy": "random", "rationale": "Fixed-control recovery fixture"}))
    c.close()
    monkeypatch.undo()
    resumed = Controller(path, Limits(seconds=999, cpu=1), run_id="mid_revision")
    calls = []
    execute_remaining = resumed.execute_experiment
    def remaining(spec, **kwargs):
        calls.append(spec.experiment_id)
        return execute_remaining(spec, **kwargs)
    monkeypatch.setattr(resumed, "execute_experiment", remaining)
    try:
        resumed.prepare()
        revision = resumed.journal.get("protocol_revision")
        assert resumed.budget.deadline == deadline
        if point == "unknown_outcome":
            assert revision["status"] == "aborted" and revision["candidates"][0]["status"] == "outcome_unknown"
            assert resumed.plan.hash == original_plan and not calls
            assert resumed.journal.get("baseline_id") == baseline and resumed.journal.get("leader_id") == leader
            assert resumed.publisher.output.read_bytes() == prior_csv
            assert resumed.budget.counter("candidate_count") == count + 1
        else:
            assert revision["status"] == "completed" and resumed.plan.hash != original_plan
            assert calls == [revision["candidates"][1]["spec"]["experiment_id"]]
            assert resumed.budget.counter("candidate_count") == count + 2
            assert resumed.journal.get("publication")["validation_hash"] == resumed.plan.hash
    finally:
        resumed.close()


def test_300_record_context_keeps_negative_memory_in_both_transports_after_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fixture/v1")
    path = write_task(tmp_path)
    c = Controller(path, Limits(seconds=180, cpu=1), run_id="memory")
    try:
        c.prepare()
        spec, result = c.lookup("baseline")
        for i in range(300):
            current = spec.model_copy(update={"experiment_id": f"history_{i}"})
            outcome = result.model_copy(update={"experiment_id": current.experiment_id, "status": "unsupported_mode",
                "error": "native_transformations_unsupported: select_kbest", "score": None})
            c.journal.save_experiment(current, outcome)
            c.capabilities.records[str(i)] = {"model": "lightgbm", "engine": "direct", "task": "regression",
                "strategy": "random", "status": "unsupported_mode", "versions": c.capabilities.versions, "probe_id": str(i)}
        c.journal.put("context_compaction_level", 2)
        for transport in ("native", "json"):
            c.qwen.transport = transport
            context = c.context()
            assert c.qwen.command_size(SYSTEM, context) < c.qwen.context_tokens
            assert context["mandatory_memory"]["baseline_id"] == "baseline"
            assert context["mandatory_memory"]["negative_results"][0]["reason"].startswith("native_transformations_unsupported")
            assert len(context["capabilities"]["verified_modes"]) < 5
            requests = []
            def handler(request):
                body = json.loads(request.content)
                assert "native_transformations_unsupported" in body["messages"][1]["content"]
                requests.append(body)
                args = {"pipeline": {"model": "extra_trees", "hyperparameters": {"n_estimators": 5}},
                    "hypothesis": H.model_dump(), "budget_seconds": 10}
                command = {"operation_id": "new_hypothesis", "tool": "train_model", "arguments": args}
                message = {"content": json.dumps(command)}
                if "tools" in body:
                    message = {"tool_calls": [{"id": "new_hypothesis", "function": {"name": "train_model", "arguments": json.dumps(args)}}]}
                return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": message}]})
            c.qwen.client.close()
            c.qwen.client = httpx.Client(transport=httpx.MockTransport(handler))
            command, _ = c.qwen.next_command(SYSTEM, context)
            assert command.arguments["pipeline"]["model"] == "extra_trees" and len(requests) == 1
        result = tool_payload(c.dispatch(ToolCommand(operation_id="history_page", tool="analyze_experiments", arguments={"cursor": 290, "limit": 10})))
        assert len(result["experiments"]) == 10 and result["next_cursor"] == 300
    finally:
        c.close()
    resumed = Controller(path, Limits(seconds=180, cpu=1), run_id="memory")
    try:
        resumed.prepare()
        memory = resumed.context()["mandatory_memory"]
        assert memory["negative_results"][0]["reason"].startswith("native_transformations_unsupported")
        assert resumed.journal.get("context_compaction_level") == 2
    finally:
        resumed.close()


def test_full_9000_budget_is_controller_owned_and_survives_serialization(tmp_path, monkeypatch):
    args = TrainArgs(pipeline=PipelineSpec(engine="flaml", model="library_search", n_trials=None), hypothesis=H, budget_seconds=9000)
    assert TrainArgs.model_validate_json(args.model_dump_json()).pipeline.n_trials is None
    c = Controller(write_task(tmp_path), Limits(seconds=10500, cpu=1), budget_mode="full_comparison")
    try:
        c.prepare()
        observed = []
        def execute(spec, **kw):
            observed.append(spec)
            return c.failed_result(spec, {"status": "timeout", "error": "controlled no-training budget probe"})
        monkeypatch.setattr(c, "execute_experiment", execute)
        operation, _ = c.journal.begin_operation(c.run_id, "large_budget", "run_automl", args.model_dump())
        c.train(args, operation)
        assert observed[0].budget_seconds == 9000 and observed[0].budget_mode == "full_comparison"
        assert observed[0].pipeline.n_trials is None
        assert c.budget.grant_training(9000, full_comparison=False) == 9000
    finally:
        c.close()


def test_flaml_actual_time_search_passes_41_trials_without_max_iter(tabular, tmp_path, monkeypatch):
    from flaml import AutoML
    X, y, task = tabular
    X = X.drop(columns=["category", "empty"])
    task = task.model_copy(update={"feature_columns": list(X), "categorical_columns": []})
    calls = []
    original = AutoML.fit
    def recorded(self, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(self, *args, **kwargs)
    monkeypatch.setattr(AutoML, "fit", recorded)
    spec = PipelineSpec(engine="flaml", model="library_search", learners=["lgbm"], n_trials=None)
    pipeline = SearchPipeline(spec, task)
    summary = pipeline.fit(X, y, {"strategy": "random", "cpu": 1, "budget_seconds": 7,
        "budget_mode": "full_comparison", "workdir": str(tmp_path), "phase_estimates": {
            "selected_fit_seconds": .2, "prediction_seconds": .1, "publication_seconds": .1}})
    assert "max_iter" not in calls[0] and calls[0]["early_stop"] is False
    evidence = summary.budget["targets"]["answer"]
    assert evidence["attempted_trials"] > 41
    assert evidence["completed_trials"] > 40
    assert evidence["stop_reason"] == "calculated_time_budget_exhausted"


def test_real_executor_and_engine_arguments_preserve_large_budget(tmp_path, tabular, monkeypatch):
    c = Controller(write_task(tmp_path / "task"), Limits(seconds=10500, cpu=1), budget_mode="full_comparison")
    try:
        c.prepare()
        request = {"mode": "probe", "probe_paths": {}, "write_probe_paths": {},
            "context": {"budget_seconds": 9000, "budget_mode": "full_comparison"}}
        result = c.executor.run(c.workdir / "large_budget_probe", request, 9000)
        assert result["status"] == "completed"
        saved = json.loads((c.workdir / "large_budget_probe/request.json").read_text())
        assert saved["context"]["budget_seconds"] == 8990
        assert saved["context"]["worker_overhead_reserve"] == 10
    finally:
        c.close()
    import flaml
    recorded = []
    class Probe:
        def fit(self, **kwargs):
            recorded.append(kwargs)
            raise ValueError("controlled engine allocation probe")
    monkeypatch.setattr(flaml, "AutoML", Probe)
    X, y, task = tabular
    pipeline = SearchPipeline(PipelineSpec(engine="flaml", model="library_search", n_trials=None), task)
    with pytest.raises(ValueError, match="controlled engine"):
        pipeline.fit(X, y, {"strategy": "random", "cpu": 1, "budget_seconds": 8990, "workdir": str(tmp_path),
            "phase_estimates": {"selected_fit_seconds": 20, "prediction_seconds": 2, "publication_seconds": 1}})
    assert recorded[0]["time_budget"] > 8900
    assert "max_iter" not in recorded[0] and recorded[0]["early_stop"] is False


def test_progress_requires_new_evidence_and_persists_dead_end(tmp_path, monkeypatch):
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1))
    try:
        c.prepare()
        assert progress.record(c, "observation", {"source_hash": "fixed"})
        state = c.journal.get("progress_state")
        state["last_useful_at"] = time.time() - 601
        c.journal.put("progress_state", state)
        for i in range(3):
            progress.record(c, "observation", {"source_hash": "fixed"})
        assert progress.exhausted(c)
        saved = c.journal.get("progress_state")["last_useful_at"]
        assert not progress.record(c, "observation", {"source_hash": "fixed"})
        assert c.journal.get("progress_state")["last_useful_at"] == saved
        assert progress.record(c, "experiment", {"computation_hash": "new", "status": "completed"})
        assert not progress.exhausted(c)
    finally:
        c.close()


def test_dependency_allocation_60_5_5_and_shared_nodes(tmp_path):
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1))
    try:
        c.prepare()
        base, result = c.lookup("baseline")
        for i, cost in enumerate((60, 5, 5)):
            spec = base.model_copy(update={"experiment_id": f"node{i}"})
            out = result.model_copy(update={"experiment_id": spec.experiment_id, "seconds": cost})
            c.journal.save_experiment(spec, out)
        ids = ["node0", "node1", "node2"]
        granted = allocation(c, ids, "node0", c.plan.fit, "control", 100)
        assert 80 < granted < 90
        graph = estimate(c, ["node0", "node0", "node1", "node2"])
        assert len(graph["nodes"]) == 3
    finally:
        c.close()


def test_60_5_5_control_preparation_finishes_within_100_and_reuses_nodes(tmp_path, monkeypatch):
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1))
    try:
        c.prepare()
        ids = ["baseline"]
        for number in (2, 3):
            outcome = tool_payload(c.dispatch(ToolCommand(operation_id=f"model_{number}", tool="train_model", arguments={
                "pipeline": PipelineSpec(model="ridge", hyperparameters={"alpha": number}).model_dump(),
                "hypothesis": H.model_dump(), "budget_seconds": 10})))
            ids.append(outcome["experiment"]["experiment_id"])
        rows = sorted(c.plan.fit + c.plan.validation)
        for identifier, seconds in zip(ids, (60, 5, 5)):
            spec, result = c.lookup(identifier)
            result.seconds = seconds
            result.training_summary["rows"] = len(rows)
            c.journal.save_experiment(spec, result)
        c.journal.put("leader_id", ids[-1])
        c.freeze("Controlled preparation timing fixture")
        plan = c.journal.get("finalization_plan")
        # The fixture prepares all three measured nodes in one phase.
        for identifier in ids:
            spec, _ = c.lookup(identifier)
            plan["policies"][identifier] = {"experiment_id": identifier, "pipeline_hash": spec.pipeline.hash,
                "frozen_pipeline": spec.pipeline.model_dump(mode="json"), "dependencies": []}
        c.journal.put("finalization_plan", plan)
        spent, calls = [0], []
        monkeypatch.setattr(Budget, "remaining", property(lambda self: 100 - spent[0]))
        original = c.executor.run
        costs = dict(zip(ids, (60, 5, 5)))
        def measured(directory, request, seconds):
            assert seconds >= costs[directory.name]
            calls.append(directory.name)
            result = original(directory, request, seconds)
            spent[0] += costs[directory.name]
            return result
        monkeypatch.setattr(c.executor, "run", measured)
        for index, identifier in enumerate(ids):
            granted = allocation(c, ids[index:], identifier, rows, "control", c.budget.remaining)
            c._prepare_finalist(identifier, rows, "control", granted)
        c._prepare_finalist(ids[0], rows, "control", c.budget.remaining)
        assert spent[0] == 70 and calls == ids
    finally:
        c.close()


def test_actual_preparation_overrun_keeps_published_csv_and_control_closed(tmp_path, monkeypatch):
    c = Controller(write_task(tmp_path), Limits(seconds=90, cpu=1))
    try:
        c.prepare()
        previous = c.publisher.output.read_bytes()
        c.freeze("Controlled underestimated worker duration")
        def overrun(directory, request, seconds):
            c.budget.deadline = time.time()
            return {"status": "timeout", "error": "Actual preparation exceeded its estimate"}
        monkeypatch.setattr(c.executor, "run", overrun)
        c.finalize()
        assert c.journal.get("control_status") == "control_preparation_failed"
        assert not c.journal.get("control_opened") and c.publisher.output.read_bytes() == previous
        assert c.state == "completed"
    finally:
        c.close()


def test_selected_model_cost_uses_actual_inner_training_rows(tmp_path):
    c = Controller(write_task(tmp_path), Limits(seconds=90, cpu=1))
    try:
        c.prepare()
        spec, result = c.lookup("baseline")
        result.seconds = 60
        # This fixture isolates row-scaling of a selected fit. Real worker phase
        # timings from prepare() belong to a different cost scenario and vary by host.
        result.training_summary = {"rows":120, "seconds":60,
            "budget":{"selected_fit_actual_seconds":2., "selected_fit_rows":60}}
        c.journal.save_experiment(spec, result)
        forecast = estimate(c, [spec.experiment_id], list(range(120)))
        assert forecast["seconds"] == pytest.approx(5.7)
    finally:
        c.close()
