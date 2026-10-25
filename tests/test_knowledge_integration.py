from automl_agent.tool_results import payload as tool_payload
import json

import httpx
import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import ReferenceArgs, ToolCommand
from automl_agent.controller import Controller, SYSTEM
from ml_knowledge.context import audit_request
from ml_knowledge.core import canonical
from conftest import write_task


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fixture.invalid/v1")
    monkeypatch.delenv("OPENAI_EXTRA_BODY", raising=False)
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1), run_id="knowledge")
    c.prepare()
    yield c
    c.close()


@pytest.mark.parametrize("mode", ["native", "json"])
def test_whole_knowledge_packet_survives_every_stage_and_exact_wire_audit(controller, mode):
    c = controller
    snapshot = c.context_memory.snapshot(SYSTEM)
    packet = c.knowledge.packet
    assert len(packet["materials"]) <= 3 and len(canonical(packet)) <= 6000
    assert packet["materials"]
    for stage in range(4):
        prepared = c.context_memory.compressor.prepare(snapshot, mode, budget=262144, minimum_stage=stage)
        payload = json.loads(prepared.wire_json)
        actual = json.loads(payload["messages"][1]["content"])
        assert actual["reference_evidence"] == packet
        assert c.knowledge.audit(prepared, payload)["send_allowed"]
        assert "external_data" in payload["messages"][0]["content"]
    assert not any(e["kind"] == "provided" for e in c.knowledge.memory.history())


def test_loss_of_evidence_blocks_the_actual_client_before_network_and_budget_claim(controller):
    c = controller
    snapshot = c.context_memory.snapshot(SYSTEM)
    compressor = c.context_memory.compressor
    original = compressor.transport.render
    def lose(context, mode, **kwargs):
        payload = json.loads(original(context, mode, **kwargs))
        content = json.loads(payload["messages"][1]["content"])
        content.pop("reference_evidence")
        payload["messages"][1]["content"] = canonical(content)
        return canonical(payload)
    compressor.transport.render = lose
    prepared = compressor.prepare(snapshot, "json", budget=262144)
    calls = c.budget.counter("llm_calls")
    def unexpected(request):
        raise AssertionError("Unverified request reached the network")
    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(unexpected))
    with pytest.raises(ValueError, match="lost after context compression"):
        c.qwen._request(None, prepared=prepared)
    assert c.budget.counter("llm_calls") == calls


def test_reference_cursor_and_budget_contract_matches_new_runtime(controller):
    c = controller
    args = ReferenceArgs(mode="api", query="sklearn.linear_model.Ridge", max_chars=2000)
    first = c.references.query(args, c.task, c.plan)
    assert first["status"] == "partial" and first["entry"]["version"] == "1.5.2"
    assert isinstance(first["next_cursor"], str)
    second = c.references.query(args.model_copy(update={"cursor": first["next_cursor"]}), c.task, c.plan)
    assert second["entry"]["fragment"]["start"] == first["entry"]["fragment"]["end"]
    assert len(canonical(first)) <= 2000 and len(canonical(second)) <= 2000
    changed = args.model_copy(update={"query": "sklearn.linear_model.LogisticRegression", "cursor": first["next_cursor"]})
    assert c.references.query(changed, c.task, c.plan)["status"] == "invalid_cursor"
    with pytest.raises(ValueError):
        ReferenceArgs(cursor=10)
    assert c.references.query(ReferenceArgs(mode="api", query="Ridge", version="9.9"), c.task, c.plan)["status"] == "missing_version"


def test_selection_uses_only_fit_aggregates_and_confirmed_contract(controller):
    c = controller
    c.context_memory.snapshot(SYSTEM)
    context = c.knowledge.selection_context()
    assert context.fit_profile["rows"] == len(c.plan.fit)
    assert {"rows", "columns", "missing", "categorical", "text", "groups", "time"} <= set(context.fit_profile)
    assert context.metric == c.task.metric.name
    assert all(type(value) is bool for key, value in context.fit_profile.items() if key not in {"rows", "columns"})
    original = context.binding()
    c.dataset.train.loc[c.plan.control, c.task.targets] = 999999
    c.dataset.test[c.task.feature_columns[0]] = 123456
    assert c.knowledge.selection_context().binding() == original
    assert "999999" not in canonical(c.knowledge.packet)
    c.journal.put("task_confirmation", None)
    with pytest.raises(ValueError, match="confirmed"):
        c.knowledge.selection_context()


def test_training_accepts_canonical_new_ids_and_records_verified_outcomes(controller):
    c = controller
    c.context_memory.snapshot(SYSTEM)
    result = tool_payload(c.dispatch(ToolCommand(operation_id="knowledge_train", tool="train_model", arguments={
        "pipeline": {"model": "ridge", "hyperparameters": {"alpha": 3}}, "budget_seconds": 30,
        "method_refs": ["method:linear"], "recipe_ref": "recipe:ridge",
        "reference_ids": ["api_sklearn_linear_model_Ridge"],
        "hypothesis": {"observation": "Fit profile has numeric features", "expected_effect": "Measure regularization",
            "changed_component": "ridge regularization", "acceptance": "Fixed RMSE and complete training"}})))
    assert result["result"]["status"] == "completed", result
    context = c.knowledge.selection_context().binding()
    records = c.knowledge.memory.current(c.references, context)
    measured = [r for r in records if r["type"] == "experiment_result"]
    assert len(measured) == 2  # baseline plus the dispatched candidate
    assert all(r["verification"] == "source_verified" for r in measured)
    history = c.knowledge.memory.history()
    assert any(e["kind"] == "referenced" and e["record_id"] == "method:linear" for e in history)
    assert not any(e["kind"] == "applied" for e in history)
    c.context_memory.snapshot(SYSTEM)
    assert len([r for r in c.knowledge.memory.current(c.references, context) if r["type"] == "experiment_result"]) == 2


@pytest.mark.parametrize("mode", ["native", "json"])
def test_mock_model_receives_routed_knowledge_and_applies_registered_card(controller, mode):
    c = controller
    c.qwen.transport = mode
    snapshot = c.context_memory.snapshot(SYSTEM)
    packet = c.knowledge.packet
    identifier = packet["materials"][0]["id"]
    assert identifier.startswith("method:hard_case_ml")
    seen = []
    arguments = {"pipeline": {"model": "ridge", "hyperparameters": {"alpha": 2}}, "budget_seconds": 30,
        "method_refs": [identifier], "hypothesis": {"observation": "Read a complete routed missing-data card",
            "expected_effect": "Compare a bounded regularized candidate", "changed_component": "regularization",
            "acceptance": "Completed candidate on unchanged RMSE protocol"}}
    def respond(request):
        payload = json.loads(request.content)
        actual = json.loads(payload["messages"][1]["content"])
        assert actual["reference_evidence"] == packet
        assert "No imputer is mandatory" in payload["messages"][0]["content"]
        seen.append(identifier)
        message = {"role": "assistant"}
        if mode == "native":
            message["tool_calls"] = [{"id": "routed", "type": "function", "function": {"name": "train_model",
                "arguments": json.dumps({"operation_id": "routed_train", **arguments})}}]
        else:
            message["content"] = json.dumps({"operation_id": "routed_train", "tool": "train_model", "arguments": arguments})
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "tool_calls" if mode == "native" else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(respond))
    command, _ = c.qwen.next_command(SYSTEM, snapshot)
    result = tool_payload(c.dispatch(command))
    assert seen == [identifier]
    assert result["result"]["status"] == "completed", result
    assert result["experiment"]["method_refs"] == [identifier]


def test_model_cannot_supply_frozen_postprocessing_values(controller):
    from automl_agent.contracts import TrainArgs
    args = TrainArgs(pipeline={"model": "extra_trees", "prediction_policy": "class_offsets",
        "postprocess_fit_policy": "inner_holdout", "class_offsets": [0., .25, -.25]},
        hypothesis={"observation": "Synthetic rejected override", "expected_effect": "No worker execution",
            "changed_component": "forged frozen values", "acceptance": "Controller rejects before fit"})
    with pytest.raises(ValueError, match="controller-owned frozen"):
        controller.make_spec(args, "forged_offsets")


def test_memory_projection_recovers_without_repeating_the_completed_tool(tmp_path, monkeypatch):
    task = write_task(tmp_path)
    c = Controller(task, Limits(seconds=180, cpu=1), run_id="resume_knowledge")
    try:
        c.prepare()
        c.context_memory.snapshot(SYSTEM)
        original = c.knowledge.completed
        def interrupted(operation):
            raise RuntimeError("crash after durable tool completion")
        monkeypatch.setattr(c.knowledge, "completed", interrupted)
        with pytest.raises(RuntimeError, match="crash after"):
            tool_payload(c.dispatch(ToolCommand(operation_id="read_before_crash", tool="get_reference",
                arguments={"mode": "methods", "reference_id": "linear", "max_chars": 12000})))
        assert c.journal.operation("read_before_crash").status == "succeeded"
        assert c.context_memory.store.db.execute("SELECT 1 FROM cc_operations WHERE operation_id='read_before_crash'").fetchone()
        monkeypatch.setattr(c.knowledge, "completed", original)
        deadline = c.budget.deadline
    finally:
        c.close()
    resumed = Controller(task, Limits(seconds=1000, cpu=1), run_id="resume_knowledge")
    try:
        resumed.prepare()
        assert resumed.budget.deadline == deadline
        resumed.context_memory.snapshot(SYSTEM)
        rows = resumed.knowledge.memory.current(resumed.references, resumed.knowledge.selection_context().binding())
        assert any(r["type"] == "project_recommendation" and r["source"]["reference_id"] == "method:linear" for r in rows)
        assert resumed.journal.db.execute("SELECT count(*) FROM operations WHERE id='read_before_crash'").fetchone()[0] == 1
        progress = resumed.knowledge.memory.progress_count
        resumed.context_memory.snapshot(SYSTEM)
        assert resumed.knowledge.memory.progress_count == progress
    finally:
        resumed.close()

