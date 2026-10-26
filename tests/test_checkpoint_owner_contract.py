import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import PipelineSpec, ToolCommand
from automl_agent.controller import Controller
from automl_agent.compatibility import incompatibilities
from automl_agent.tool_results import payload
from conftest import write_task
from test_custom_execution import VALID_CODE


@pytest.mark.parametrize("supplied_engine", [None, "direct", "custom"])
def test_custom_checkpoints_reject_before_feature_fit(tmp_path, monkeypatch, supplied_engine):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1), run_id="checkpoint_owner")
    c.no_llm = c.qwen.disabled = True
    try:
        c.prepare()
        before = c.publisher.output.read_bytes()
        count = c.budget.counter("candidate_count")
        executed = []
        def execute(spec, **kwargs):
            executed.append(spec)
            return c.failed_result(spec, {"status": "error", "error": "offline execution sentinel"})
        monkeypatch.setattr(c, "execute_experiment", execute)
        pipeline = {"model": "lightgbm", "custom_component": "transformer", "checkpoint_counts": [4, 8]}
        if supplied_engine:
            pipeline["engine"] = supplied_engine
        result = payload(c.dispatch(ToolCommand(operation_id="custom_checkpoints", tool="run_custom_experiment",
            arguments={"pipeline": pipeline, "code": VALID_CODE, "budget_seconds": 30, "hypothesis": {
                "observation": "Custom feature fit precedes ready model checkpoint selection",
                "expected_effect": "Require the same verified owner as native early stopping",
                "changed_component": "checkpoint_counts", "acceptance": "Reject without a fit or a paid review"}})))
        assert not executed, "Unverified custom selection reached candidate execution"
        assert "checkpoint_selection_requires_direct_or_optuna_boosting" in result["error"]
        assert c.budget.counter("candidate_count") == count
        assert c.publisher.output.read_bytes() == before and not c.journal.get("control_opened")
    finally:
        c.close()


@pytest.mark.parametrize("engine,model", [("custom", "lightgbm"), ("autogluon", "library_search"),
    ("flaml", "library_search"), ("lightautoml", "library_search"), ("direct", "ridge")])
def test_unsupported_checkpoint_owners_are_explicit(tabular, engine, model):
    _, _, task = tabular
    reasons = incompatibilities(PipelineSpec(engine=engine, model=model, checkpoint_counts=[4, 8]), task)
    assert "checkpoint_selection_requires_direct_or_optuna_boosting" in reasons


@pytest.mark.parametrize("engine", ["direct", "optuna"])
@pytest.mark.parametrize("model", ["lightgbm", "xgboost", "catboost", "hist_gradient_boosting"])
def test_existing_boosting_checkpoint_owners_remain_supported(tabular, engine, model):
    _, _, task = tabular
    reasons = incompatibilities(PipelineSpec(engine=engine, model=model, checkpoint_counts=[4, 8]), task)
    assert "checkpoint_selection_requires_direct_or_optuna_boosting" not in reasons
