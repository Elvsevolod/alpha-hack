import json

import pandas as pd
import pytest

from automl_agent.admission import admit, drain
from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis, PipelineSpec, ToolCommand, TrainArgs, KnowledgeSearchArgs, KnowledgeReadArgs
from automl_agent.controller import Controller
from automl_agent.qwen import QwenUnavailable
from automl_agent.tool_results import payload
from conftest import write_task


@pytest.fixture
def prepared(tmp_path):
    controller = Controller(write_task(tmp_path), limits=Limits(seconds=90, cpu=1, memory_mb=1024), run_id="plan")
    controller.no_llm = True
    controller.prepare()
    yield controller
    controller.close()


def candidate(model="ridge", **parameters):
    return TrainArgs(pipeline=PipelineSpec(model=model, hyperparameters=parameters), budget_seconds=10,
        hypothesis=Hypothesis(observation="Confirmed fixed synthetic task", expected_effect="Measure another bounded recipe",
            changed_component=model, acceptance="Compare the same held-out metric"))


def test_explicit_contract_publishes_even_when_semantic_channel_is_down(tmp_path, monkeypatch):
    path = write_task(tmp_path)
    path.write_text("Regression. Target: answer.\nPrimary metric: RMSE\n")
    controller = Controller(path, limits=Limits(seconds=45, cpu=1, memory_mb=1024), run_id="local")
    monkeypatch.setattr(controller.qwen, "resolve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("No semantic API call is needed")))
    try:
        controller.prepare()
        output = pd.read_csv(path.parent / "submission.csv")
        assert len(output) == 25 and list(output) == ["row", "answer"]
        assert controller.journal.get("first_csv_seconds") is not None
    finally:
        controller.close()


def test_conflicting_explicit_metric_publishes_only_the_template(tmp_path):
    path = write_task(tmp_path)
    path.write_text("Primary metric: Macro F1\n")
    controller = Controller(path, limits=Limits(seconds=20, cpu=1, memory_mb=1024), run_id="conflict")
    try:
        report = controller.run(no_llm=True)
        assert report["state"] == "contract_unresolved"
        output = tmp_path / "submission.csv"
        assert output.exists()
        assert output.read_bytes() == (tmp_path / "sample_submission.csv").read_bytes()
    finally:
        controller.close()


@pytest.mark.parametrize("reason", ["HTTP 402", "HTTP 502", "unknown generation outcome", "connection timeout"])
def test_admitted_work_runs_once_without_any_new_model_request(prepared, monkeypatch, reason):
    c = prepared
    monkeypatch.setattr(c.qwen, "critique", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Degraded work must not request a critic")))
    command = ToolCommand(operation_id="admitted", tool="train_model", arguments=candidate("ridge", alpha=3).model_dump(mode="json"))
    assert admit(c, command)
    first = drain(c, reason)
    count = len(c.records())
    second = drain(c, reason)
    assert first == ["admitted"] and second == [] and len(c.records()) == count
    assert c.journal.operation("admitted").status == "succeeded"
    assert c.journal.get("channel_state")["status"] == "degraded"


def test_degraded_queue_rejects_stale_scope_and_never_reopens_control(prepared):
    c = prepared
    command = ToolCommand(operation_id="stale", tool="train_model", arguments=candidate().model_dump(mode="json"))
    assert admit(c, command)
    saved = c.journal.get("admitted_queue")
    saved[0]["protocol"] = "another"
    c.journal.put("admitted_queue", saved)
    assert drain(c, "provider unavailable") == []
    assert c.journal.get("admitted_queue")[0]["status"] == "rejected_stale_admission"
    c.journal.put("control_opened", True)
    assert not admit(c, command)
    assert drain(c, "provider unavailable") == []


def test_action_card_to_preview_to_fit_is_one_canonical_recipe(prepared):
    c = prepared
    found = c.knowledge_tools.discover(KnowledgeSearchArgs(collection="actions", query="KNN neighbors"))
    route = next(item for item in found["entries"] if item["id"] == "action:knn")["read"]["arguments"]
    card = c.knowledge_tools.read(KnowledgeReadArgs.model_validate(route))["entry"]
    application = card["apply"]
    application["changes"] = [{"path": "hyperparameters.n_neighbors", "operation": "set", "value": 3}]
    preview = payload(c.dispatch(ToolCommand(operation_id="preview", tool="preview_recipe", arguments=application)))
    args = candidate().model_dump(mode="json")
    args.pop("pipeline")
    args["apply_recipe"] = application
    trained = payload(c.dispatch(ToolCommand(operation_id="applied", tool="train_model", arguments=args)))
    assert trained["result"]["status"] == "completed"
    assert trained["experiment"]["pipeline"] == PipelineSpec.model_validate(preview["pipeline"]).model_dump(mode="json")


def test_finish_records_optional_directions_and_can_explain_skips(prepared):
    c = prepared
    args = {"reason": "One useful baseline is available", "result_ids": [c.journal.get("baseline_id")],
        "skipped_directions": {"all": "This is a capped technical fixture; the remaining search belongs to the external development experiment."}}
    result = payload(c.dispatch(ToolCommand(operation_id="finish_check", tool="finish", arguments=args)))
    assert c.journal.operation("finish_check").status == "succeeded"
    # A measured boosting family covers the weak-card alternatives, so they are
    # optional rather than a blocker (F18).
    assert result["status"] == "frozen" and c.state == "frozen"
    assert c.journal.get("finish_evidence")["reason"] == args["reason"]


def test_classical_scope_remains_explicit():
    from automl_agent.custom import validate_code
    from automl_agent.dependency_policy import permitted_module
    from automl_agent.research import RESEARCH_POLICY
    assert not permitted_module("torch") and "Classical ML only" in RESEARCH_POLICY
    with pytest.raises(ValueError, match="classical ML"):
        validate_code("from sklearn.neural_network import MLPRegressor\ndef build_pipeline(context): return MLPRegressor()", "pipeline")
