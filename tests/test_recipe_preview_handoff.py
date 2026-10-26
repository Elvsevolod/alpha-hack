"""A preview must carry the exact application to the next stateless decision."""
import json
from pathlib import Path

import cloudpickle
import httpx
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.controller import Controller
from automl_agent.util import digest
from conftest import write_task


def delivered(context, tool):
    for observation in context["observations"]:
        content = observation.get("content")
        if isinstance(content, str):
            content = json.loads(content)
        if isinstance(content, dict) and content.get("tool") == tool:
            return content["data"]
    pytest.fail("The complete pending tool result was not delivered")


@pytest.mark.parametrize("transport", ["native", "json"])
@pytest.mark.parametrize("pressure", [False, True])
def test_preview_to_apply_uses_only_delivered_receipt(tmp_path, monkeypatch, transport, pressure):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = Controller(write_task(tmp_path, kind="classification"),
        Limits(seconds=180, cpu=1, max_calls=5), run_id="preview_handoff", transport=transport)
    requests = []
    base_hashes, resolved_hashes, preview_payloads = [], [], []
    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        context = json.loads(body["messages"][1]["content"])
        if len(requests) == 1:
            command = dict(operation_id="read_base", tool="read_run_records", arguments={
                "ids": [c.journal.get("leader_id")], "view": "pipeline"})
        elif len(requests) == 2:
            base = delivered(context, "read_run_records")
            command = dict(operation_id="preview", tool="preview_recipe", arguments={
                "base_id": base["experiment_id"], "base_hash": base["recipe_hash"],
                "changes": [{"path": "hyperparameters.class_weight", "value": "balanced"}]})
            base_hashes.append(base["recipe_hash"])
            if pressure:
                c.journal.put("context_compaction_level", 2)
        elif len(requests) == 3:
            preview = delivered(context, "preview_recipe")
            preview_payloads.append(preview)
            # Build this command only from the current request, with no copy
            # of the earlier preview arguments supplied by the fixture.
            application = preview["apply_recipe"]
            assert application["base_hash"] != preview["recipe_hash"]
            resolved_hashes.append(preview["recipe_hash"])
            command = dict(operation_id="apply_preview", tool="train_model", arguments={
                "apply_recipe": application, "budget_seconds": 30, "hypothesis": {
                    "observation": "Offline replay of the delivered preview application",
                    "expected_effect": "Preserve the declared class weight without rereading the base",
                    "changed_component": "class_weight", "acceptance": "A real fit and valid CSV; no quality claim"}})
        else:
            command = dict(operation_id="finish_handoff", tool="finish", arguments={
                "result_ids": [c.journal.get("leader_id")], "reason": "Technical handoff replay complete",
                "imputation_skip_reason": "Fixed fixture", "skipped_directions": {"all": "Not an autonomous quality test"}})
        message = {"role": "assistant", "content": json.dumps(command)}
        if "tools" in body:
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_" + str(len(requests)), "type": "function", "function": {
                    "name": command["tool"], "arguments": json.dumps({"operation_id": command["operation_id"], **command["arguments"]})}}]}
        return httpx.Response(200, json={"choices": [{"message": message,
            "finish_reason": "tool_calls" if "tools" in body else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(handle))
    c.qwen.key, c.qwen.base_url = "offline-fixture", "http://fixture.invalid/v1"
    try:
        report = c.run()
        assert preview_payloads and "apply_recipe" in preview_payloads[0], "Preview delivery lost the exact executable application"
        assert len(requests) == 4, report
        op = c.journal.operation("apply_preview")
        assert op.status == "succeeded", op.result
        spec, result = c.lookup(op.experiment_id)
        assert result.status == "completed"
        assert digest(spec.pipeline.research_recipe()) == resolved_hashes[0] != base_hashes[0]
        with (Path(result.artifacts["directory"]) / "model.pkl").open("rb") as stream:
            model = cloudpickle.load(stream)
        assert model.models["answer"].get_params()["class_weight"] == "balanced"
        assert report["state"] == "completed" and report["refit_status"] == "completed"
        output = pd.read_csv(c.publisher.output, dtype={"row": str})
        sample = pd.read_csv(tmp_path / "sample_submission.csv", dtype={"row": str})
        assert list(output) == list(sample) and output.row.tolist() == sample.row.tolist()
        assert output.answer.notna().all()
        counts = dict(c.journal.db.execute("select kind,count(*) from events group by kind"))
        assert counts.get("native_fallback", 0) == counts.get("qwen_failure", 0) == 0
    finally:
        c.close()
