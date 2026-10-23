from automl_agent.tool_results import payload as tool_payload
import json
import time

import httpx
import pytest

from automl_agent.budget import Budget, Limits
from automl_agent.contracts import ToolCommand, PipelineSpec
from automl_agent.controller import Controller
from automl_agent.journal import Journal
from automl_agent.qwen import QwenClient, QwenError, QwenUnavailable, QwenRetryableError
from automl_agent.util import atomic_bytes
from automl_agent.pipelines import make_estimator
from conftest import write_task


def test_default_research_continues_beyond_old_time_and_count_quotas(tmp_path):
    journal = Journal(tmp_path)
    budget = Budget(journal, Limits(), time.time() - 2400)
    journal.put("llm_seconds", 1800)
    for _ in range(50):
        budget.claim_call()
        budget.claim_candidate()
    assert budget.snapshot()["qwen_calls"] is None
    assert budget.snapshot()["candidates"] is None
    assert budget.counter("llm_calls") == 50
    assert budget.final_reserve() == 300
    journal.close()


def test_explicit_development_limits_and_finalization_reserve_still_apply(tmp_path):
    journal = Journal(tmp_path)
    budget = Budget(journal, Limits(seconds=100, max_calls=1, max_candidates=1), time.time())
    budget.claim_call()
    budget.claim_candidate()
    with pytest.raises(TimeoutError, match="call limit"):
        budget.claim_call()
    with pytest.raises(TimeoutError, match="candidate limit"):
        budget.claim_candidate()
    journal.close()


def client(tmp_path, monkeypatch, handler):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model-proxy:8000/v1")
    monkeypatch.setenv("OPENAI_MODEL", "fixture-model")
    monkeypatch.delenv("OPENAI_EXTRA_BODY", raising=False)
    monkeypatch.delenv("AGENT_OUTPUT_TOKENS", raising=False)
    journal = Journal(tmp_path)
    qwen = QwenClient(journal, Budget(journal, Limits(seconds=3600), time.time()))
    qwen.client.close()
    qwen.client = httpx.Client(transport=httpx.MockTransport(handler))
    return qwen, journal


def test_internal_http_endpoint_and_truncation_recovery(tmp_path, monkeypatch):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        assert str(request.url) == "http://model-proxy:8000/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"finish_reason": "length" if len(requests) == 1 else "stop",
            "message": {"content": "{}"}}], "usage": {"total_tokens": 10}})
    qwen, journal = client(tmp_path, monkeypatch, handler)
    try:
        with pytest.raises(QwenError, match="Truncated"):
            qwen._request([{"role": "user", "content": "fixture"}])
        assert qwen._request([{"role": "user", "content": "fixture"}])["content"] == "{}"
        assert [r["max_tokens"] for r in requests] == [16384, 32768]
        assert journal.get("token_usage")["total_tokens"] == 20
        assert journal.get("qwen_configuration")["output_tokens"] == 32768
    finally:
        qwen.close()
        journal.close()


def test_multiple_native_tools_keep_first_valid_operation_without_json_retry(tmp_path, monkeypatch):
    calls = [{"id": "one", "function": {"name": "finish", "arguments": json.dumps({"reason": "fixture"})}},
        {"id": "two", "function": {"name": "finish", "arguments": json.dumps({"reason": "later"})}}]
    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": calls}}]})
    qwen, journal = client(tmp_path, monkeypatch, handler)
    try:
        command, _ = qwen.next_command("fixture", {})
        assert command.operation_id == "one" and qwen.transport == "native"
        assert journal.get("llm_calls") == 1
        assert journal.db.execute("SELECT COUNT(*) FROM events WHERE kind='extra_native_calls_deferred'").fetchone()[0] == 1
    finally:
        qwen.close()
        journal.close()


@pytest.mark.parametrize("failure", ["connect", "503", "length"])
def test_transient_failure_keeps_native_transport_for_the_next_attempt(tmp_path, monkeypatch, failure):
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert "tools" in body
        assert request.extensions["timeout"]["connect"] <= 15
        if len(requests) == 1:
            if failure == "connect":
                raise httpx.ConnectError("fixture", request=request)
            if failure == "503":
                return httpx.Response(503)
            return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {"content": ""}}]})
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
            {"id": "recovered", "function": {"name": "finish", "arguments": json.dumps({"reason": "fixture"})}}]}}]})
    qwen, journal = client(tmp_path, monkeypatch, handler)
    try:
        with pytest.raises(QwenRetryableError):
            qwen.next_command("fixture", {})
        assert qwen.transport == "auto" and journal.get("llm_calls") == 1
        command, _ = qwen.next_command("fixture", {})
        assert command.operation_id == "recovered" and qwen.transport == "native"
        assert journal.get("llm_calls") == 2
        assert not journal.db.execute("SELECT COUNT(*) FROM events WHERE kind='native_fallback'").fetchone()[0]
    finally:
        qwen.close()
        journal.close()


def test_account_failure_is_distinct_from_transient_api_error(tmp_path, monkeypatch):
    qwen, journal = client(tmp_path, monkeypatch, lambda request: httpx.Response(402))
    try:
        with pytest.raises(QwenUnavailable, match="402"):
            qwen.next_command("fixture", {})
        assert journal.get("llm_calls") == 1
    finally:
        qwen.close()
        journal.close()


def test_three_transient_errors_do_not_end_research(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model-proxy:8000/v1")
    controller = Controller(write_task(tmp_path), Limits(seconds=90, cpu=1))
    calls, recovery = [], []
    def next_command(*args):
        calls.append(1)
        if len(calls) <= 3:
            raise QwenError("temporary HTTP 503")
        return ToolCommand(operation_id="recovered_finish", tool="finish", arguments={
            "reason": "Recovery verified", "result_ids":["baseline"], "imputation_skip_reason": "Transport fixture"}), ""
    monkeypatch.setattr(controller.qwen, "next_command", next_command)
    monkeypatch.setattr(controller, "recover_search", lambda error: recovery.append(str(error)))
    try:
        result = controller.run()
        assert len(calls) == 4 and len(recovery) == 1
        assert result["state"] == "completed" and result["model_error_count"] == 3
        assert result["research_stop_reason"] == "Recovery verified"
    finally:
        controller.close()


def test_finish_requires_completed_result_references_in_long_runs(tmp_path):
    controller = Controller(write_task(tmp_path), Limits(seconds=3600, cpu=1))
    try:
        controller.prepare()
        result = tool_payload(controller.dispatch(ToolCommand(operation_id="premature_finish", tool="finish", arguments={"reason": "Done"})))
        assert result["status"] == "rejected" and "result_ids" in result["error"]
        assert controller.state == "searching"
    finally:
        controller.close()


def test_only_public_csv_gets_readable_permissions_including_recovery(tmp_path):
    controller = Controller(write_task(tmp_path), Limits(seconds=90, cpu=1))
    try:
        controller.prepare()
        output = controller.publisher.output
        assert output.stat().st_mode & 0o777 == 0o644
        output.chmod(0o600)
        controller.publisher.recover()
        assert output.stat().st_mode & 0o777 == 0o644
        private = tmp_path / "private-metadata"
        atomic_bytes(private, b"fixture")
        assert private.stat().st_mode & 0o777 == 0o600
    finally:
        controller.close()


@pytest.mark.parametrize("model,parameters", [
    ("lightgbm", {"n_jobs": 99, "random_state": 999, "device": "gpu"}),
    ("xgboost", {"n_jobs": 99, "random_state": 999, "tree_method": "gpu_hist", "device": "cuda"}),
    ("catboost", {"thread_count": 99, "random_seed": 999, "task_type": "GPU", "verbose": True}),
])
def test_native_adapters_override_managed_parameters_before_validation(model, parameters):
    estimator = make_estimator(PipelineSpec(model=model, hyperparameters=parameters, seed=42), False, {"cpu": 2})
    actual = estimator.get_params()
    if model == "catboost":
        assert actual["thread_count"] == 2 and actual["random_seed"] == 42 and actual["task_type"] == "CPU"
    else:
        assert actual["n_jobs"] == 2 and actual["random_state"] == 42
        if model == "xgboost":
            assert actual["tree_method"] == "hist" and actual["device"] == "cpu"


def test_repeated_computation_cannot_hide_behind_ignored_cpu_overrides(tmp_path):
    controller = Controller(write_task(tmp_path), Limits(seconds=90, cpu=1))
    try:
        controller.prepare()
        base = {"pipeline": {"model": "extra_trees", "hyperparameters": {"n_estimators": 5, "n_jobs": 99}},
            "hypothesis": {"observation": "Fixture", "expected_effect": "Fixture", "changed_component": "Fixture", "acceptance": "Fixture"},
            "budget_seconds": 20}
        first = tool_payload(controller.dispatch(ToolCommand(operation_id="managed_cpu_first", tool="train_model", arguments=base)))
        assert first["result"]["status"] == "completed"
        base["pipeline"]["hyperparameters"]["n_jobs"] = 1
        second = tool_payload(controller.dispatch(ToolCommand(operation_id="managed_cpu_second", tool="train_model", arguments=base)))
        assert second["status"] == "duplicate_computation" and "Duplicate computation" in second["error"]
        assert second["existing_experiment_id"] == first["experiment"]["experiment_id"]
        assert second["computation_hash"] == controller.lookup(second["existing_experiment_id"])[0].computation_hash
        base["budget_seconds"] = 21
        third = tool_payload(controller.dispatch(ToolCommand(operation_id="managed_cpu_more_budget", tool="train_model", arguments=base)))
        assert third["result"]["status"] == "completed"
        assert third["experiment"]["budget_seconds"] > first["experiment"]["budget_seconds"]
        assert controller.budget.counter("candidate_count") == 2
    finally:
        controller.close()
