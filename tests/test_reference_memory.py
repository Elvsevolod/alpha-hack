from automl_agent.tool_results import payload as tool_payload
from automl_agent.budget import Limits
from automl_agent.contracts import ToolCommand
from automl_agent.controller import Controller
from conftest import write_task


def test_read_sources_survive_another_result_compaction_and_process_resume(tmp_path):
    task = write_task(tmp_path)
    controller = Controller(task, Limits(seconds=90, cpu=1), run_id="reference_memory")
    try:
        controller.prepare()
        tool_payload(controller.dispatch(ToolCommand(operation_id="missing_read", tool="get_reference",
            arguments={"mode": "api", "query": "NoSuchFixtureApi"})))
        tool_payload(controller.dispatch(ToolCommand(operation_id="method_read", tool="get_reference",
            arguments={"mode": "methods", "reference_id": "linear"})))
        tool_payload(controller.dispatch(ToolCommand(operation_id="api_read", tool="get_reference",
            arguments={"mode": "api", "query": "sklearn.linear_model.Ridge", "version": "1.5.2"})))
        tool_payload(controller.dispatch(ToolCommand(operation_id="api_repeat", tool="get_reference",
            arguments={"mode": "api", "query": "sklearn.linear_model.Ridge", "version": "1.5.2"})))
        controller.journal.put("last_tool_result", {"status": "unrelated_later_result"})
        controller.qwen.context_tokens = 131072
        controller.journal.put("context_compaction_level", 2)
        context = controller.context()
        facts = {f["reference_id"]: f for f in context["state"]["facts"] if f.get("source") == "reference_read"}
        assert facts["method:linear"]["kind"] == "method"
        assert facts["method:linear"]["applicability"] == "unknown"
        api = facts["api_sklearn_linear_model_Ridge"]
        assert api["kind"] == "documentation" and api["version"] == "1.5.2" and api["doc_sha256"]
        assert all("content" not in fact for fact in facts.values())
        assert any(f.get("source") == "reference_unavailable" and f.get("query") == "NoSuchFixtureApi"
            for f in context["state"]["facts"])
        assert context["context_compaction"]["full_contracts_and_outcomes_preserved"]
        assert context["state"]["counters"]["unproductive_reads"] == 1
        old_deadline = controller.budget.deadline
    finally:
        controller.close()
    resumed = Controller(task, Limits(seconds=1000, cpu=1), run_id="reference_memory")
    try:
        resumed.prepare()
        restored = {f["reference_id"]: f for f in resumed.context()["state"]["facts"] if f.get("source") == "reference_read"}
        assert restored == facts and resumed.budget.deadline == old_deadline
        assert resumed.budget.counter("unproductive_reads") == 1
        assert any(f.get("source") == "reference_unavailable" and f.get("query") == "NoSuchFixtureApi"
            for f in resumed.context()["state"]["facts"])
    finally:
        resumed.close()
