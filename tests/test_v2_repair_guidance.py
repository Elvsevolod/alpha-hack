"""Repair messages must lead to executable public routes with unchanged trust checks."""
import json

import pytest

from automl_agent.contracts import ToolCommand
from automl_agent.controller import SYSTEM
from automl_agent.tool_results import payload
from test_tool_api_v2 import agent


@pytest.mark.parametrize("mode", ["native", "json"])
def test_reference_repair_through_public_transport(agent, monkeypatch, mode):
    c = agent
    c.context_memory.snapshot(SYSTEM)
    transport = c.context_memory.compressor.transport
    count = 0

    def invoke(tool, arguments):
        nonlocal count
        count += 1
        operation_id = "repair_" + str(count)
        if mode == "native":
            message = {"tool_calls": [{"type": "function", "function": {"name": tool,
                "arguments": json.dumps({"operation_id": operation_id, **arguments})}}]}
        else:
            message = {"content": json.dumps({"operation_id": operation_id,
                "tool": tool, "arguments": arguments})}
        command = transport.parse_response({"choices": [{"finish_reason":
            "tool_calls" if mode == "native" else "stop", "message": message}]}, mode)
        c.context_memory.validate(command)
        typed = ToolCommand.model_validate({k: v for k, v in command.as_dict().items() if k != "research_note"})
        return payload(c.dispatch(typed))

    before = c.publisher.output.read_bytes()
    executions = []

    def execute(spec, **_):
        executions.append(spec)
        return c.failed_result(spec, {"status": "error", "error": "offline executor sentinel"})

    monkeypatch.setattr(c, "execute_experiment", execute)
    train = {"pipeline": {"model": "ridge", "hyperparameters": {"alpha": 17}},
        "budget_seconds": 30, "reference_ids": ["invented_api_id"],
        "hypothesis": {"observation": "Invalid reference returned a repair instruction",
            "changed_component": "reference repair", "expected_effect": "Reach executor with verified IDs",
            "acceptance": "Public routes return acceptable immutable references"}}
    rejected = invoke("train_model", train)
    assert rejected["status"] == "rejected" and not executions
    assert "collection=documents" in rejected["error"] and "get_reference" not in rejected["error"]
    prepared = c.context_memory.compressor.prepare(c.context_memory.snapshot(SYSTEM), mode, budget=262144)
    assert rejected["error"] in json.loads(prepared.wire_json)["messages"][1]["content"]

    references = []
    for collection, query, field in [("documents", "sklearn.linear_model.Ridge", "reference_ids"),
            ("methods", "linear", "method_refs"), ("recipes", "ridge", "recipe_ref"),
            ("search_spaces", "ridge", "search_space_ref")]:
        if collection != "documents":
            with pytest.raises(ValueError) as error:
                c.references.catalog_entry(collection, "invented_id")
            assert "collection=" + collection in str(error.value)
            assert "get_reference" not in str(error.value)
        found = invoke("search_knowledge", {"collection": collection, "query": query})
        entry = next(e for e in found["entries"] if (e["id"] == "api_sklearn_linear_model_Ridge"
            if collection == "documents" else e["id"].endswith(":" + query)))
        read = invoke(entry["read"]["tool"], entry["read"]["arguments"])
        assert read["status"] == "success"
        references.append(read["reference"])
        c.validate_evidence([read["reference"]])
        if collection == "documents":
            assert c.references.known_document(entry["id"])
        else:
            c.references.catalog_entry(collection, entry["id"])
        if collection != "search_spaces":
            train[field] = [entry["id"]] if field.endswith("_ids") or field == "method_refs" else entry["id"]

    # Wrong namespace and version still fail; returned evidence and IDs reach execution.
    assert not c.references.known_document("recipe:ridge")
    with pytest.raises(ValueError) as error:
        c.validate_evidence([{**references[0], "version": "invented_version"}])
    assert "entry.read" in str(error.value) and "get_reference" not in str(error.value)
    with pytest.raises(ValueError):
        c.validate_evidence([{**references[0], "collection": "recipes"}])
    train["hypothesis"]["evidence_refs"] = references
    accepted = invoke("train_model", train)
    assert len(executions) == 1 and accepted["result"]["error"]
    duplicate = invoke("train_model", train)
    assert duplicate["status"] == "duplicate_computation" and len(executions) == 1
    assert "read_run_records" in duplicate["next_action"]
    read = invoke("read_run_records", {"ids": [duplicate["existing_experiment_id"]]})
    assert read["status"] == "success"

    with pytest.raises(ValueError) as error:
        c.validate_evidence([{"kind": "record", "id": "invented_record", "version": "1"}])
    assert "search_run_records" in str(error.value) and "analyze_experiments" not in str(error.value)
    found = invoke("list_run_records", {})
    record = found["records"][0]
    assert invoke("read_run_records", {"ids": [record["record_id"]]})["status"] == "success"
    c.validate_evidence([{"kind": "record", "id": record["record_id"], "version": record["source_version"]}])
    assert c.publisher.output.read_bytes() == before
