from automl_agent.tool_results import payload as tool_payload
import json

import httpx
import pytest

from context_compressor import BudgetExceeded, Command, Source
from automl_agent.budget import Limits
from automl_agent.contracts import ToolCommand
from automl_agent.controller import Controller, SYSTEM
from automl_agent.qwen import QwenOutcomeUnknown, QwenRetryableError
from automl_agent.util import canonical
from conftest import write_task


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key-not-a-real-credential")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fixture.invalid/v1")
    monkeypatch.delenv("OPENAI_EXTRA_BODY", raising=False)
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1), run_id="integration")
    c.prepare()
    yield c
    c.close()


def response(payload, operation="inspect_next", note=None, arguments=None, tool="get_dataset_report"):
    arguments = {"report": "missing"} if arguments is None else arguments
    if "tools" in payload:
        content = {"operation_id": operation, **arguments}
        if note is not None:
            content["research_note"] = note
        message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_test", "type": "function",
            "function": {"name": tool, "arguments": canonical(content)}}]}
    else:
        content = {"operation_id": operation, "tool": tool, "arguments": arguments}
        if note is not None:
            content["research_note"] = note
        message = {"role": "assistant", "content": canonical(content)}
    return {"choices": [{"message": message, "finish_reason": "tool_calls" if "tools" in payload else "stop"}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 17, "total_tokens": 140}}


def mock_client(c, handler):
    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(handler))


def pending(c):
    tool_payload(c.dispatch(ToolCommand(operation_id="read_first", tool="get_reference", arguments={"query": "ridge", "max_chars": 500})))
    return c.context_memory.store.pending_fragments()


def test_compact_request_keeps_decision_fields_and_caches_summary_migration(controller):
    c = controller
    memory = c.context_memory
    snapshot = memory.snapshot(SYSTEM)
    core = json.loads(snapshot.core_json)
    assert core["task"]["targets"] == c.task.targets
    assert core["task"]["id_columns"] == c.task.id_columns
    assert core["task"]["output"]["columns"] == c.task.output.columns
    assert core["policies_resources"]["selection"] == c.plan.selection_policy.model_dump(mode="json")
    details = [json.loads(value) for value in snapshot.details]
    assert details and all("content" not in item and "model" in item["summary"] for item in details)
    assert all(len(canonical(item["summary"]).encode()) <= 1536 for item in details)
    records = memory.store.db.execute("SELECT count(*) FROM cc_records WHERE kind='experiment'").fetchone()[0]
    memory.store.db.execute("DELETE FROM cc_sources WHERE section='summary'")  # pre-v015 journal
    memory.snapshot(SYSTEM)
    assert memory.store.db.execute("SELECT count(*) FROM cc_sources WHERE section='summary'").fetchone()[0] == records
    assert memory.store.db.execute("SELECT count(*) FROM cc_records WHERE kind='experiment'").fetchone()[0] == records


def test_current_hypothesis_and_distinct_rejected_ids_remain_inline(controller):
    c = controller
    changed = "generic transformation " * 28
    c.journal.put("active_hypothesis", {"operation_id": "active_probe", "changed_component": changed,
        "observation": "Observed missing values", "expected_effect": "Compare imputation", "acceptance": "Fixed validation"})
    for identifier in ("old_invalid_read", "new_invalid_read"):
        tool_payload(c.dispatch(ToolCommand(operation_id=identifier, tool="inspect_dataset", arguments={"columns": ["does_not_exist"]})))
    snapshot = c.context_memory.snapshot(SYSTEM)
    for stage in range(4):
        prepared = c.context_memory.compressor.fit(snapshot, "native", budget=262144, minimum_stage=stage)
        work = json.loads(json.loads(prepared.wire_json)["messages"][1]["content"])["core"]["current_work"]
        assert work["hypothesis"]["changed_component"] == changed
        assert work["hypothesis"]["source"] == "agent_assumption"
        assert {item["operation_id"] for item in work["rejected_operations"]} == {"old_invalid_read", "new_invalid_read"}


def test_compact_feedback_keeps_worst_registered_slices_and_suppression(controller):
    c = controller
    spec, result = c.records()[0]
    identifier = "compact_feedback_probe"
    feedback = {"allowed": {"status": "available", "scope": "external_validation", "slices": {
        "low_error": {"rows": 50, "score": 2.0}, "high_error": {"rows": 60, "score": 7.0}}},
        "private": {"status": "suppressed", "reason": "minimum_support_or_complementary_disclosure"}}
    c.journal.save_experiment(spec.model_copy(update={"experiment_id": identifier}),
        result.model_copy(update={"experiment_id": identifier, "feedback": feedback}))
    snapshot = c.context_memory.snapshot(SYSTEM)
    detail = next(json.loads(value) for value in snapshot.details if json.loads(value)["record_id"] == identifier)
    assert detail["summary"]["weakest_slices"] == [{"report_id": "allowed", "slice": "high_error", "rows": 60,
        "score": 7.0, "scope": "external_validation"}]
    for stage in range(4):
        prepared = c.context_memory.compressor.fit(snapshot, "native", budget=262144, minimum_stage=stage)
        work = json.loads(json.loads(prepared.wire_json)["messages"][1]["content"])["core"]["current_work"]
        assert work["latest_id"] == identifier
        assert work["latest_result"]["record_id"] == identifier
        assert work["latest_id"] != work["leader_id"]


@pytest.mark.parametrize("mode", ["native", "json"])
def test_real_client_sends_exact_compressed_body_and_acknowledges_ranges(controller, mode):
    c = controller
    before = pending(c)
    c.qwen.transport = mode
    wire = []
    def handle(request):
        wire.append(request.content)
        return httpx.Response(200, json=response(json.loads(request.content)))
    mock_client(c, handle)
    command, _ = c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
    store = c.context_memory.store
    saved = store.db.execute("SELECT request_id,wire_json FROM cc_requests ORDER BY rowid DESC LIMIT 1").fetchone()
    assert wire == [saved["wire_json"].encode()]
    body = json.loads(json.loads(wire[0])["messages"][1]["content"])
    assert body["observations"] == before
    for item in before:
        assert store.observation_status(item["observation_id"])["delivered_until"] == item["range"]["end"]
    assert store.pending_commands()[0]["operation_id"] == command.operation_id
    assert store.db.execute("SELECT COUNT(*) FROM cc_usage").fetchone()[0] == 1
    tool_payload(c.dispatch(command))
    assert not store.pending_commands()


def test_failed_terminal_write_rolls_back_result_and_outbox(controller, monkeypatch):
    c = controller
    original = c.context_memory.store.complete_operation
    def fail(*args, **kwargs):
        commit = kwargs["commit"]
        def crash(db):
            commit(db)
            raise RuntimeError("crash before commit")
        return original(*args, **{**kwargs, "commit": crash})
    monkeypatch.setattr(c.context_memory.store, "complete_operation", fail)
    with pytest.raises(RuntimeError, match="crash before commit"):
        tool_payload(c.dispatch(ToolCommand(operation_id="atomic_result", tool="inspect_dataset", arguments={"report": "missing"})))
    assert c.journal.operation("atomic_result").status == "running"
    assert not c.context_memory.store.db.execute("SELECT 1 FROM cc_operations WHERE operation_id='atomic_result'").fetchone()
    assert not c.context_memory.store.db.execute("SELECT 1 FROM cc_observations WHERE operation_id='atomic_result'").fetchone()
    monkeypatch.setattr(c.context_memory.store, "complete_operation", original)
    count = len(c.records())
    c.recover_operations()
    assert c.journal.operation("atomic_result").outcome == "confirmed"
    assert c.journal.operation("atomic_result").status == "succeeded"
    assert len(c.records()) == count
    assert c.context_memory.store.db.execute("SELECT 1 FROM cc_observations WHERE operation_id='atomic_result'").fetchone()


@pytest.mark.parametrize("failure", ["timeout", "invalid", "truncated"])
def test_unaccepted_response_does_not_consume_observations(controller, failure):
    c = controller
    before = pending(c)
    c.qwen.transport = "json"
    def handle(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("fixture", request=request)
        body = response(json.loads(request.content))
        if failure == "invalid":
            body["choices"][0]["message"]["content"] = '{"tool":"finish"}'
        else:
            body["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=body)
    mock_client(c, handle)
    with pytest.raises((ValueError, QwenRetryableError, QwenOutcomeUnknown)):
        c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
    if failure == "timeout":
        assert c.journal.get('qwen_unknown_outcome')
        with pytest.raises(QwenOutcomeUnknown):
            c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
    assert not c.context_memory.store.pending_commands()
    for item in before:
        assert c.context_memory.store.observation_status(item["observation_id"])["delivered_until"] == item["range"]["start"]


def test_native_fallback_uses_same_snapshot_with_distinct_request_ids(controller):
    c = controller
    pending(c)
    bodies = []
    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "tools" in body:
            return httpx.Response(400, json={"error": {"code": "unsupported_tools"}})
        return httpx.Response(200, json=response(body))
    mock_client(c, handle)
    c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
    assert c.qwen.transport == "json" and len(bodies) == 2
    requests = c.context_memory.store.db.execute("SELECT request_id,snapshot_id FROM cc_requests").fetchall()
    assert requests[0]["request_id"] != requests[1]["request_id"]
    assert requests[0]["snapshot_id"] == requests[1]["snapshot_id"]
    assert c.context_memory.store.db.execute("SELECT COUNT(*) FROM cc_receipts").fetchone()[0] == 1


def test_failed_json_repair_keeps_selected_transport_and_pending_result(controller):
    c = controller
    fragments = pending(c)
    attempts = []
    def malformed(request):
        body = json.loads(request.content)
        attempts.append("native" if "tools" in body else "json")
        if "tools" in body:
            return httpx.Response(400, json={"error": {"code": "unsupported_tools"}})
        return httpx.Response(200, json=response(body, arguments={"research_note": {}, "report": "missing"}))
    mock_client(c, malformed)
    with pytest.raises(ValueError):
        c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
    assert attempts == ["native", "json"] and c.qwen.transport == c.journal.get("transport") == "json"
    assert not c.context_memory.store.pending_commands()
    assert all(c.context_memory.store.observation_status(item["observation_id"])["delivered_until"] == 0 for item in fragments)
    def repaired(request):
        body = json.loads(request.content)
        assert "tools" not in body
        return httpx.Response(200, json=response(body))
    mock_client(c, repaired)
    command, _ = c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
    assert command.operation_id == "inspect_next"


@pytest.mark.parametrize("mode", ["native", "json"])
def test_provider_pressure_retries_without_losing_new_results(controller, mode):
    c = controller
    c.qwen.transport = mode
    pending(c)
    stages = []
    def handle(request):
        payload = json.loads(request.content)
        context = json.loads(payload["messages"][1]["content"])
        stages.append(context["compaction"]["stage"])
        assert context["observations"]
        if len(stages) <= 3:
            return httpx.Response(400, json={"error": {"code": "context_length_exceeded"}})
        return httpx.Response(200, json=response(payload))
    mock_client(c, handle)
    c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
    assert stages == [0, 1, 2, 3]
    rows = c.context_memory.store.db.execute("SELECT DISTINCT snapshot_id FROM cc_requests").fetchall()
    assert len(rows) == 1 and c.budget.counter("candidate_count") == 0


def test_minimum_context_failure_retains_queue_and_trained_csv(controller):
    c = controller
    pending(c)
    csv = c.publisher.output.read_bytes()
    mock_client(c, lambda request: httpx.Response(413, json={"error": {"code": "context_length_exceeded"}}))
    with pytest.raises(BudgetExceeded):
        c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
    assert c.publisher.output.read_bytes() == csv
    assert c.context_memory.store.db.execute("SELECT COUNT(*) FROM cc_receipts").fetchone()[0] == 0


def test_accepted_command_and_note_survive_restart_without_new_model_call(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fixture.invalid/v1")
    monkeypatch.delenv("OPENAI_EXTRA_BODY", raising=False)
    path = write_task(tmp_path)
    c = Controller(path, Limits(seconds=180, cpu=1), run_id="recovery")
    note = {"question": "Does missingness explain the error?", "reading_goal": "Read fit-only counts",
        "expected_evidence": "Observed missing values", "related_ids": ["baseline"]}
    try:
        c.prepare()
        pending(c)
        mock_client(c, lambda request: httpx.Response(200, json=response(json.loads(request.content), note=note)))
        c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
        calls, deadline = c.budget.counter("llm_calls"), c.budget.deadline
    finally:
        c.close()
    c = Controller(path, Limits(seconds=9999), run_id="recovery")
    try:
        c.prepare()
        command = c.context_memory.pending()
        assert command.operation_id == "inspect_next"
        state = c.journal.get("research_state")
        assert state["questions"] == [note["question"]] and "agent_assumption" in state["assumptions"][0]
        tool_payload(c.dispatch(command))
        assert c.context_memory.pending() is None
        assert c.budget.counter("llm_calls") == calls and c.budget.deadline == deadline
        first = tool_payload(c.journal.operation(command.operation_id).result)
        assert tool_payload(c.dispatch(command)) == first
        assert c.context_memory.store.db.execute("SELECT COUNT(*) FROM cc_operations WHERE operation_id=?", (command.operation_id,)).fetchone()[0] == 1
    finally:
        c.close()


def test_context_routes_retrieve_unicode_and_reject_control_labels(controller):
    c = controller
    source = Source("old-code", "1", "граница🙂" * 400, "code")
    c.context_memory.store.put_source(source)
    first = tool_payload(c.dispatch(ToolCommand(operation_id="read_unicode", tool="analyze_experiments", arguments={
        "context_read": {"source_id": source.source_id, "version": "1", "scope": "code"}})))
    assert first["content"].encode() == source.content.encode()[:first["range"]["end"]]
    assert first["continuation"] and not first["document_complete"]
    fragment = c.context_memory.store.pending_fragments()[0]
    assert fragment["content"] == first["content"] and fragment["source"] == first["source"]
    with pytest.raises(ValueError):
        tool_payload(c.dispatch(ToolCommand(operation_id="read_control", tool="analyze_experiments", arguments={
            "context_read": {"source_id": "labels", "version": "1", "scope": "control"}})))


def test_fit_profile_is_calculated_once_per_immutable_split(controller, monkeypatch):
    c = controller
    original = c.dataset.profile
    calls = []
    def profile(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(c.dataset, "profile", profile)
    a = c.context_memory.snapshot(SYSTEM)
    b = c.context_memory.snapshot(SYSTEM)
    assert len(calls) == 1
    assert a.profile_json == b.profile_json
    assert json.loads(a.core_json)["policies_resources"]["deadline_epoch"] == c.budget.deadline


def test_invalid_engine_cannot_acknowledge_a_result(controller):
    c = controller
    fragments = pending(c)
    snapshot = c.context_memory.snapshot(SYSTEM)
    prepared = c.context_memory.compressor.prepare(snapshot, "json", budget=c.qwen.context_tokens)
    command = Command.parse({"operation_id": "bad_engine", "tool": "train_model", "arguments": {
        "pipeline": {"engine": "flaml", "model": "flaml"},
        "hypothesis": {"observation": "x", "expected_effect": "x", "changed_component": "x", "acceptance": "x"}}})
    with pytest.raises(ValueError, match="engine"):
        c.context_memory.store.accept_command(prepared.request_id, command, c.context_memory.validate)
    assert all(c.context_memory.store.observation_status(item["observation_id"])["delivered_until"] == 0 for item in fragments)


def test_forbidden_external_predicate_does_not_acknowledge_observations(controller):
    c = controller
    fragments = pending(c)
    snapshot = c.context_memory.snapshot(SYSTEM)
    prepared = c.context_memory.compressor.prepare(snapshot, "json", budget=c.qwen.context_tokens)
    command = Command.parse({"operation_id": "bad_slice", "tool": "get_dataset_report", "arguments": {
        "scope": "validation_features", "predicate": {"column": "value", "op": "gt", "value": 0}}})
    with pytest.raises(ValueError, match="predicates"):
        c.context_memory.store.accept_command(prepared.request_id, command, c.context_memory.validate)
    assert all(c.context_memory.store.observation_status(item["observation_id"])["delivered_until"] == 0 for item in fragments)


@pytest.mark.parametrize("scope", ["fit", "validation_features", "test_features"])
def test_inspection_delivery_preserves_original_scope(controller, scope):
    c = controller
    tool_payload(c.dispatch(ToolCommand(operation_id="inspect_scope", tool="inspect_dataset", arguments={"scope": scope, "report": "missing"})))
    snapshot = c.context_memory.snapshot(SYSTEM)
    observations = json.loads(snapshot.observations_json)
    assert observations[0]["source"]["scope"] == scope
    assert observations[0]["source"]["trust"] == "external_data"


def test_custom_tool_preserves_accepted_direct_pipeline_input(controller):
    c = controller
    c.context_memory.snapshot(SYSTEM)
    # The existing custom tool converts the engine after validating the supplied
    # pipeline; an omitted engine defaults to direct and was already supported.
    c.context_memory.validate(Command.parse({"operation_id": "custom_direct", "tool": "run_custom_experiment", "arguments": {
        "pipeline": {"model": "ridge"}, "code": "def build_pipeline(context):\n    return None\n",
        "hypothesis": {"observation": "x", "expected_effect": "x", "changed_component": "x", "acceptance": "x"}}}))


@pytest.mark.parametrize("mode,encoded_objects", [("native", False), ("json", False), ("native", True)])
def test_controller_compressed_research_trains_and_finishes(controller, mode, encoded_objects):
    c = controller
    c.qwen.transport = mode
    steps = []
    def handle(request):
        payload = json.loads(request.content)
        context = json.loads(payload["messages"][1]["content"])
        steps.append(context)
        if len(steps) == 1:
            args = {"pipeline": {"model": "ridge", "hyperparameters": {"alpha": 3}}, "budget_seconds": 30,
                "hypothesis": {"observation": "Compare the trained baseline on the fixed split",
                    "expected_effect": "Measure ridge error", "changed_component": "model", "acceptance": "Measured RMSE"}}
            if encoded_objects:
                args.update({name: json.dumps(args[name]) for name in ("pipeline", "hypothesis")})
            return httpx.Response(200, json=response(payload, "compressed_train", tool="train_model", arguments=args))
        assert any(item["kind"] == "training_card" for item in context["observations"])
        card = json.loads(next(item["content"] for item in context["observations"] if item["kind"] == "training_card"))
        assert card["status"] == "completed" and isinstance(card["score"], float)
        return httpx.Response(200, json=response(payload, "compressed_finish", tool="finish", arguments={
            "reason": "Integrated compression verified", "result_ids":["baseline"], "imputation_skip_reason": "Bounded integration check",
            "skipped_directions": {"all": "This fixture verifies transport and two commands; model quality is measured by the separate development suite."}}))
    mock_client(c, handle)
    report = c.run()
    assert report["state"] == "completed" and report["failure"] is None, report
    assert report["control_status"] == report["refit_status"] == "completed"
    assert len(steps) == 2 and tool_payload(c.journal.operation("compressed_train").result)["result"]["status"] == "completed"
    assert c.context_memory.store.db.execute("SELECT COUNT(*) FROM cc_receipts").fetchone()[0] == 2
