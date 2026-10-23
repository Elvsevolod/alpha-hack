import json

import httpx
import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import (KnowledgeListArgs, KnowledgeSearchArgs, RunListArgs,
    RunReadArgs, RunSearchArgs, TOOL_SCHEMAS, ToolCommand, OperationRecord)
from automl_agent.controller import Controller, SYSTEM
from automl_agent.knowledge_tools import CursorError
from automl_agent.tool_results import RESULT_SCHEMAS
from automl_agent.util import digest
from context_compressor import Record, Source
from conftest import write_task


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-api2-fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://fixture.invalid/v1")
    monkeypatch.delenv("OPENAI_EXTRA_BODY", raising=False)
    c = Controller(write_task(tmp_path), Limits(seconds=600, cpu=1), run_id="api2_test")
    c.prepare()
    yield c
    c.close()


def call(c, name, args, op):
    result = c.dispatch(ToolCommand(operation_id=op, tool=name, arguments=args))
    assert RESULT_SCHEMAS[name].model_validate(result)
    assert c.context_memory.store.observation_status(result["observation_id"]) is not None
    return result


def test_all_fifteen_actions_share_one_public_registry_and_bounded_transports(agent):
    assert len(TOOL_SCHEMAS) == 15 and set(TOOL_SCHEMAS) == set(RESULT_SCHEMAS)
    assert set(ToolCommand.model_json_schema()["properties"]["tool"]["enum"]) == set(TOOL_SCHEMAS)
    snapshot = agent.context_memory.snapshot(SYSTEM)
    for mode in ("native", "json"):
        compressor = agent.context_memory.compressor
        assert compressor.transport.overhead_bytes(mode) <= 49152
        prepared = compressor.prepare(snapshot, mode, budget=131072)
        assert prepared.conservative_size + 8192 <= 131072
        body = json.loads(prepared.wire_json)
        context = json.loads(body["messages"][1]["content"])
        schemas = {x["function"]["name"] for x in body["tools"]} if mode == "native" else set(context["tool_argument_schemas"])
        assert schemas == set(TOOL_SCHEMAS)
        assert "analyze_experiments" not in json.dumps(context["core"])
        route = context["core"]["reading_routes"]["contract"]
        assert route["tool"] == "read_run_records"
        TOOL_SCHEMAS[route["tool"]].model_validate(route["arguments"])


def test_recipe_ref_is_reference_only_in_public_schema_and_agent_policy():
    description = TOOL_SCHEMAS["train_model"].model_json_schema()["properties"]["recipe_ref"]["description"]
    assert "does not choose or apply a pipeline" in description
    assert "Encode recipe settings in pipeline" in description
    assert "`recipe_ref` only records/checks" in SYSTEM
    assert "Copy chosen settings into `pipeline`" in SYSTEM


def test_exact_symbol_version_missing_and_no_substitution(agent):
    result = call(agent, "search_knowledge", {"query": "sklearn.linear_model.Ridge", "strategy": "exact_symbol"}, "exact")
    assert result["status"] == "success"
    found = result["data"]["entries"][0]
    assert found["version"] == "1.5.2" and found["answer_confirmed"] is False
    read = call(agent, "read_knowledge", found["read"]["arguments"], "read")
    assert "Ridge" in read["data"]["entry"]["content"]
    missing = call(agent, "search_knowledge", {"query": "sklearn.linear_model.Ridge", "filters": {"library": "scikit-learn", "version": "99.0"}}, "version")
    assert missing["status"] == "missing_version" and not missing["data"]["entries"]
    absent = call(agent, "search_knowledge", {"query": "qxzv_missing_api_999", "strategy": "exact_symbol"}, "absent")
    assert absent["status"] == "no_matches"


def test_catalog_and_lexical_zero_matches_are_distinct(agent):
    missing = agent.knowledge_tools.discover(KnowledgeSearchArgs(collection="methods", query="qxzv_nonexistent_audit_42"))
    assert missing["status"] == "no_matches" and missing["entries"] == []
    catalog = agent.knowledge_tools.discover(KnowledgeListArgs(collection="methods"), catalog=True)
    assert catalog["catalog_browse"] and catalog["total"] >= 26 and catalog["entries"]
    bad = agent.knowledge_tools.discover(KnowledgeSearchArgs(collection="methods", query="linear", strategy="fts"))
    assert bad["status"] == "incompatible"


def test_search_pagination_is_stable_and_bound_to_filters(agent):
    args = KnowledgeSearchArgs(query="imputation", limit=1)
    first = agent.knowledge_tools.discover(args)
    assert first["next_cursor"]
    second = agent.knowledge_tools.discover(args.model_copy(update={"cursor": first["next_cursor"]}))
    assert first["entries"][0]["id"] != second["entries"][0]["id"]
    with pytest.raises(CursorError):
        agent.knowledge_tools.discover(args.model_copy(update={"query": "ensemble", "cursor": first["next_cursor"]}))
    agent.knowledge_tools.cache.clear()
    assert agent.knowledge_tools.discover(args.model_copy(update={"cursor": first["next_cursor"]}))["entries"] == second["entries"]


def test_fts_failure_uses_explicit_lexical_fallback(agent):
    agent.references.use_fts = False
    value = agent.knowledge_tools.discover(KnowledgeSearchArgs(query="missing imputation", strategy="auto"))
    assert value["method"] == "lexical" and value["fallback_reason"] == "fts_unavailable" and value["entries"]
    explicit = agent.knowledge_tools.discover(KnowledgeSearchArgs(query="imputation", strategy="fts"))
    assert explicit["status"] == "technical_failure"


def test_history_finds_old_unknown_id_in_three_calls_and_notes(agent):
    store = agent.context_memory.store
    with store.transaction():
        for i in range(1000):
            content = "Unique ancient diagnostic: incompatible sparse matrix" if i == 7 else "unrelated observation " + str(i)
            source = Source("history_" + str(i), "1", content, "controller", origin="fixture")
            store._put_source(source)
            store._put_record(Record("old_" + str(i), source.source_id, "1", "controller", component="encoding",
                engine="ridge", status="failed", error_code="sparse_input", protocol=agent.plan.hash))
    result = call(agent, "search_run_records", {"query": "ANCIENT diagnostic", "filters": {"engine": "ridge", "status": "failed"}}, "search_old")
    assert result["data"]["total"] == 1
    record = result["data"]["records"][0]
    result = call(agent, "read_run_records", {"ids": [record["record_id"]]}, "read_old")
    assert "incompatible sparse matrix" in result["data"]["records"][0]["content"]


def test_record_pages_keep_snapshot_after_new_records(agent):
    first = agent.run_tools.search(RunListArgs(limit=1), catalog=True)
    token = first["next_cursor"]
    assert token
    second = agent.run_tools.search(RunListArgs(limit=1, cursor=token), catalog=True)
    assert first["records"][0]["record_id"] != second["records"][0]["record_id"]
    with pytest.raises(CursorError):
        agent.run_tools.search(RunListArgs(limit=1, cursor=token, filters={"kind": "operation"}), catalog=True)


def test_validation_rejection_is_idempotent_and_keeps_csv(agent):
    old = agent.plan.hash
    csv = agent.publisher.output.read_bytes()
    args = {"proposed_strategy": agent.plan.strategy, "rationale": "Do not modify a locked task"}
    first = call(agent, "revise_validation", args, "revise")
    assert first["data"]["decision"] == "rejected" and first["data"]["revision_status"] == "rejected"
    assert first["data"]["active_protocol_hash"] == old == agent.plan.hash
    assert agent.publisher.output.read_bytes() == csv
    assert call(agent, "revise_validation", args, "revise") == first


@pytest.mark.parametrize("mode", ["native", "json"])
def test_2500_character_read_survives_every_compaction_stage(agent, mode):
    text = "UNIQUE_DOCUMENT_SECTION\n" + "context delivery evidence " * 105
    operation, _ = agent.journal.begin_operation(agent.run_id, "large", "read_knowledge", {"fixture": True})
    operation.status, operation.outcome, operation.result = "succeeded", "confirmed", {"status": "success", "content": text, "complete": True}
    agent.context_memory.complete(operation)
    snapshot = agent.context_memory.snapshot(SYSTEM)
    for stage in range(4):
        prepared = agent.context_memory.compressor.prepare(snapshot, mode, budget=262144, minimum_stage=stage)
        context = json.loads(json.loads(prepared.wire_json)["messages"][1]["content"])
        contents = [json.loads(item["content"])["data"]["content"] for item in context["observations"] if item["operation_id"] == "large"]
        assert contents == [text]
    assert agent.context_memory.store.observation_status(operation.result["observation_id"])["state"] == "included"


def test_unknown_code_scope_and_evidence_refs_are_rejected(agent):
    with pytest.raises(ValueError):
        RunReadArgs(ids=["a", "b"], include_code=True)
    with pytest.raises(ValueError):
        RunReadArgs(source={"source_id": "secret", "version": "1", "scope": "control"})
    assert agent.run_tools.read(RunReadArgs(ids=["unknown"]))["status"] == "unknown_id"
    with pytest.raises(ValueError):
        agent.validate_evidence([{"kind": "knowledge", "id": "missing", "collection": "documents", "version": agent.references.version}])


def test_rephrasing_reordering_and_new_operation_ids_are_not_evidence(agent):
    result = agent.knowledge_tools.discover(KnowledgeListArgs(collection="methods"), catalog=True)
    first = agent.evidence_fragments("list_knowledge", result)
    second = agent.evidence_fragments("search_knowledge", {**result, "entries": list(reversed(result["entries"]))})
    from automl_agent import progress
    assert progress.record_many(agent, "observation", first)
    assert not progress.record_many(agent, "observation", second)
    assert not progress.record_many(agent, "observation", [])


def test_old_run_is_rejected_before_training(tmp_path):
    from automl_agent.journal import Journal
    task = write_task(tmp_path)
    journal = Journal(tmp_path / ".agent_work" / "old")
    journal.put("source_hash", "old")
    journal.close()
    with pytest.raises(ValueError, match="Incompatible tool API"):
        Controller(task, run_id="old")


def test_accepted_api2_command_survives_restart_without_second_execution(agent):
    result = call(agent, "get_dataset_report", {"report": "missing"}, "report")
    assert result["status"] == "success" and result["data"]["scope"] == "fit"
    assert call(agent, "get_dataset_report", {"report": "missing"}, "report") == result
    assert agent.journal.db.execute("SELECT count(*) FROM operations WHERE id='report'").fetchone()[0] == 1


def test_new_receipt_ids_do_not_turn_repeated_reads_into_progress(agent):
    for i in range(3):
        call(agent, "get_dataset_report", {"report": "missing", "knowledge_gap": "Verify repeat identity"}, f"same_{i}")
    result = agent.run_tools.search(RunSearchArgs(filters={"kind": "observation", "tool": "get_dataset_report"}))
    assert result["records"]
    assert agent.evidence_fragments("search_run_records", result) == []
    read = agent.run_tools.read(RunReadArgs(ids=[result["records"][0]["record_id"]]))
    assert agent.evidence_fragments("read_run_records", read) == []


def test_legacy_symbol_alias_fit_signature_is_not_lost(agent):
    result = agent.knowledge_tools.discover(KnowledgeSearchArgs(query="MLPClassifier sample_weight early_stopping", max_chars=12000))
    expected = agent.references.document("api_sklearn_neural_network_MLPClassifier_fit")["id"]
    assert expected in [entry["id"] for entry in result["entries"]]


def test_versioned_utf8_source_read_has_public_continuation_and_evidence(agent):
    content = "строка о проверке доступа\n" * 60
    source = Source("utf8_fixture", "version_1", content, "controller")
    route = agent.context_memory.store.put_source(source)
    fragments = []
    for index in range(20):
        result = call(agent, route["tool"], {**route["arguments"], "knowledge_gap": "Read the next immutable UTF-8 fragment"}, "utf8_" + str(index))
        fragments.append(result["data"]["content"])
        agent.validate_evidence([result["evidence_ref"]])
        if result["complete"]:
            break
        route = result["continuation"]
        assert route["tool"] == "read_run_records"
    assert ''.join(fragments) == content


def test_delivery_envelope_stays_within_requested_read_budget(agent):
    from automl_agent.util import canonical
    for maximum in (2000, 6000, 12000):
        result = call(agent, "search_knowledge", {"query": "imputation", "max_chars": maximum,
            "knowledge_gap": "Check the bounded card envelope"}, "budget_" + str(maximum))
        assert len(canonical(result)) <= maximum


def test_indexed_memory_keeps_standalone_conflict_and_idempotence_semantics(tmp_path):
    from automl_agent.knowledge import IndexedKnowledgeMemory
    from ml_knowledge.memory import KnowledgeMemory, DEPENDENCIES
    context = {key: "same-context" for key in DEPENDENCIES}
    histories = []
    for cls, name in ((KnowledgeMemory, "original"), (IndexedKnowledgeMemory, "indexed")):
        memory = cls(tmp_path / (name + '.sqlite'))
        try:
            for claim, topic in (("claim one", "topic"), ("claim two", "topic"), ("another claim", "other"), ("claim one", "topic")):
                memory.hypothesis(claim, observation="observed", expected_effect="expected", test="measured", context=context, topic=topic)
            histories.append((memory.history(), memory.progress_count))
        finally:
            memory.close()
    assert histories[0] == histories[1]


@pytest.mark.parametrize("stop", ["call_limit", "context_budget"])
def test_local_budget_stop_is_not_an_api_error(tmp_path, monkeypatch, stop):
    from context_compressor import BudgetExceeded
    monkeypatch.setenv("OPENAI_API_KEY", "offline_fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://fixture.invalid/v1")
    c = Controller(write_task(tmp_path), Limits(seconds=120, cpu=1, max_calls=0 if stop == "call_limit" else 3))
    def next_command(*args):
        if stop == "call_limit":
            pytest.fail("Exhausted call budget must not prepare another model call")
        raise BudgetExceeded("Protected context does not fit")
    monkeypatch.setattr(c.qwen, "next_command", next_command)
    try:
        result = c.run()
        assert result["state"] == "completed"
        assert result["control_status"] == result["refit_status"] == "completed"
        assert result["model_error_count"] == 0 and result["model_calls"] == 0
        assert c.journal.db.execute("SELECT count(*) FROM events WHERE kind='qwen_failure'").fetchone()[0] == 0
    finally:
        c.close()
