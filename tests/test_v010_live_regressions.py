from automl_agent.tool_results import payload as tool_payload
from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis, PipelineSpec, ToolCommand
from automl_agent.controller import Controller
from automl_agent.custom import semantic_code_hash
from conftest import write_task

H = Hypothesis(observation="Reproduced live syntax failure", expected_effect="Return a repairable diagnostic",
    changed_component="candidate_syntax", acceptance="Continue search and preserve verified CSV")


def test_invalid_generated_python_is_rejected_readable_and_does_not_fail_controller(tmp_path):
    c = Controller(write_task(tmp_path), Limits(seconds=90, cpu=1))
    try:
        c.prepare()
        publication = c.publisher.output.read_bytes()
        code = "def build_pipeline(context):\n    return (\n"
        result = tool_payload(c.dispatch(ToolCommand(operation_id="broken_python", tool="run_custom_experiment", arguments={
            "pipeline": PipelineSpec(engine="custom", model="new_idea").model_dump(),
            "hypothesis": H.model_dump(), "code": code})))
        assert result["status"] == "rejected" and "Candidate Python syntax" in result["error"]
        assert c.state == "searching" and c.publisher.output.read_bytes() == publication
        details = tool_payload(c.dispatch(ToolCommand(operation_id="read_broken", tool="analyze_experiments", arguments={
            "operation_ids": ["broken_python"], "include_code": True})))
        assert details["operations"][0]["code"] == code
        assert tool_payload(details["operations"][0]["result"])["error"] == result["error"]
        c.freeze("Syntax rejection verified")
        c.finalize()
        assert c.journal.get("control_status") == "completed" and c.journal.get("refit_status") == "completed"
    finally:
        c.close()


def test_formatting_and_comments_are_not_a_new_custom_computation():
    a = "def build_pipeline(context):\n    return 1\n"
    b = "# A new explanation\n\ndef build_pipeline(context):\n    return   1  # unchanged implementation\n"
    assert semantic_code_hash(a) == semantic_code_hash(b)
    assert semantic_code_hash(a) != semantic_code_hash(a.replace("return 1", "return 2"))
import time

import pytest

from automl_agent import progress


def test_rearranging_already_read_results_does_not_restart_progress_timer(tmp_path):
    c = Controller(write_task(tmp_path), Limits(seconds=90, cpu=1))
    try:
        c.prepare()
        trained = tool_payload(c.dispatch(ToolCommand(operation_id="new_model", tool="train_model", arguments={
            "pipeline": PipelineSpec(model="ridge", hyperparameters={"alpha": 3}).model_dump(),
            "hypothesis": H.model_dump(), "budget_seconds": 10})))
        identifier = trained["experiment"]["experiment_id"]
        first = ["baseline", identifier]
        tool_payload(c.dispatch(ToolCommand(operation_id="first_read", tool="analyze_experiments", arguments={"experiment_ids": first})))
        state = c.journal.get("progress_state")
        state["last_useful_at"] = time.time() - 601
        c.journal.put("progress_state", state)
        saved = state["last_useful_at"]
        for i, ids in enumerate([list(reversed(first)), first[:1], first[1:]]):
            tool_payload(c.dispatch(ToolCommand(operation_id=f"reordered_{i}", tool="analyze_experiments", arguments={"experiment_ids": ids})))
            assert c.journal.get("progress_state")["last_useful_at"] == saved
        assert c.journal.get("unproductive_reads") == 3 and progress.exhausted(c)
    finally:
        c.close()


def test_operation_id_conflict_remains_actionable_after_strong_compaction(tmp_path):
    c = Controller(write_task(tmp_path), Limits(seconds=90, cpu=1))
    try:
        c.prepare()
        tool_payload(c.dispatch(ToolCommand(operation_id="used", tool="analyze_experiments", arguments={})))
        with pytest.raises(ValueError, match="new operation_id") as error:
            tool_payload(c.dispatch(ToolCommand(operation_id="used", tool="analyze_experiments", arguments={"limit": 1})))
        c.journal.put("last_tool_result", {"status": "invalid_command_or_api", "error": str(error.value)})
        c.journal.put("context_compaction_level", 2)
        memory = c.context()["mandatory_memory"]
        assert "new operation_id" in memory["last_command_error"]["error"]
        suggested = memory["suggested_new_operation_id"]
        c.budget.counter("llm_calls", 1)
        assert c.context()["mandatory_memory"]["suggested_new_operation_id"] != suggested
    finally:
        c.close()
