import json
from pathlib import Path
import time

from context_compressor import Emission, Record, REQUIRED_FLAGS, Source, Transport, configure
from context_compressor.contracts import canonical


def task(wide=False):
    n = 10000 if wide else 8
    return {"kind": "classification" if wide else "regression", "targets": ["answer"],
            "id_columns": ["id"], "feature_columns": [f"feature_{i}" for i in range(n)],
            "categorical_columns": [], "text_columns": [], "series_columns": [],
            "time_column": None, "group_column": None, "horizon": None,
            "metric": {"name": "log_loss" if wide else "rmse", "average": "macro", "positive_label": None,
                       "aggregation": "uniform", "target_weights": None, "relevance_threshold": 0},
            "output": {"prediction_type": "probabilities" if wide else "values",
                       "columns": [f"class_{i}" for i in range(1000)] if wide else ["id", "answer"],
                       "probability_columns": {f"class_{i}": ["answer", i] for i in range(1000)} if wide else {}},
            "classes": {"answer": list(range(1000))} if wide else {}, "source_hash": "data-v1",
            "feature_constraints": {f"feature_{i}": {"role": "numeric"} for i in range(n)} if wide else {},
            "evidence": {"metric": ["confirmed fixture"]}, "explicit_validation": "random"}


def setup(store, **overrides):
    old = store.get_meta("controller_state")
    values = {"run_id": "run-1", "task": task(), "policies": {"selection": {"maximize": False},
              "permissions": dict(REQUIRED_FLAGS), "constraints": ["Full policies are enforced by the controller."]},
              "policy_flags": dict(REQUIRED_FLAGS), "data_hash": "data-v1", "protocol_hash": "protocol-v1",
              "environment_hash": "environment-v1", "deadline_epoch": old["policies_resources"]["deadline_epoch"] if old else time.time() + 3600,
              "reserve_seconds": 60, "cpu": 6, "memory_mib": 5120,
              "counters": {"calls": 0, "candidates": 0, "stagnation": 0}, "work": {}}
    values.update(overrides)
    return configure(store, **values)


def transport(real=False, **overrides):
    if real:
        values = json.loads((Path(__file__).parent / "fixtures/controller_schemas.json").read_text())
    else:
        values = {"system": "Research under the confirmed task, resources and fixed policies.", "tool_schemas": {
            "analyze_experiments": {"type": "object", "properties": {"experiment_ids": {"type": "array", "items": {"type": "string"}}}, "additionalProperties": False},
            "get_reference": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False},
            "finish": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"], "additionalProperties": False}}}
    values.update(overrides)
    return Transport(**values)


def emit(store, operation_id="read-1", content="Справка: use a fit-only split.", kind="reference", scope="reference"):
    source = Source("doc-" + operation_id, "v1", content, scope)
    observation_id = store.complete_operation(operation_id, "result-v1", {"status": "succeeded", "reference": source.source_id}, [Emission(source, kind=kind)])[0]
    return observation_id


def response(mode="json", operation_id="step-1", note=None, tool="analyze_experiments", arguments=None, finish_reason=None):
    arguments = {} if arguments is None else arguments
    command = {"operation_id": operation_id, "tool": tool, "arguments": arguments}
    if note is not None:
        command["research_note"] = note
    if mode == "json":
        message = {"content": canonical(command)}
    else:
        payload = {**arguments, "operation_id": operation_id}
        if note is not None:
            payload["research_note"] = note
        message = {"tool_calls": [{"id": "provider-1", "type": "function", "function": {"name": tool, "arguments": canonical(payload)}}]}
    return {"choices": [{"finish_reason": finish_reason or ("stop" if mode == "json" else "tool_calls"), "message": message}]}


def note(status="awaiting_evidence"):
    return {"question": "Стоит ли изменить заполнение пропусков?", "reading_goal": "Уточнить условия сравнения двух вариантов.",
            "expected_evidence": "Сравнимые метрики одного протокола.", "related_ids": ["experiment-0"], "status": status}


def allowed(command):
    """Fixture controller's semantic validator (not used in production)."""
    if command.tool == "finish" and not json.loads(command.arguments_json).get("reason"):
        raise ValueError("finish requires a reason")


def seed_history(store, count, *, first_index=0, detail_bytes=1024, record_options=None):
    with store.transaction():
        for i in range(first_index, first_index + count):
            source = Source(f"report-{i}", "v1", f"Report {i}:\n" + ("данные " * detail_bytes)[:detail_bytes], "registered_feedback")
            store._put_source(source)
            options = {"component": f"component-{i % 17}", "engine": f"engine-{i % 3}",
                       "status": "completed" if i % 3 else "failed", "error_code": None if i % 3 else f"ERR_{i % 11}",
                       "protocol": "protocol-v1", "environment": "environment-v1", "metric": "rmse", "maximize": False,
                       "score": 1.0 + i / 1000 if i % 3 else None, "seconds": 0.1, "peak_memory_mib": 32,
                       "checked_change": "Checked imputation change"}
            options.update(record_options or {})
            store._put_record(Record(f"experiment-{i}", source.source_id, source.version, **options))


def user_context(prepared):
    return json.loads(json.loads(prepared.wire_json)["messages"][1]["content"])
