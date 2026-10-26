import numpy as np
import pytest
import json
import pandas as pd
import cloudpickle

from automl_agent.contracts import PipelineSpec
from automl_agent.pipelines import DirectPipeline
from automl_agent.budget import Limits
from automl_agent.controller import Controller
from automl_agent.contracts import Hypothesis, ToolCommand
from automl_agent.tool_results import payload
from conftest import write_task
from test_plan_operators import context


def temporal_case(tabular, tmp_path, periods):
    X, y, task = tabular
    X, y = X.copy(), y.abs() + 1
    X["when"] = np.repeat(np.arange(periods), len(X) // periods)
    task = task.model_copy(update={"time_column": "when", "feature_columns": [*task.feature_columns, "when"]})
    ctx = {**context(task, tmp_path), "strategy": "temporal"}
    spec = PipelineSpec(model="lightgbm", hyperparameters={"n_estimators": 12, "min_child_samples": 3},
        early_stopping_rounds=3, evaluation_interval=1, target_transform="log1p", log_inverse="oof_smearing")
    return X, y, task, ctx, spec


def test_three_periods_fit_stopping_then_independent_final_inverse(tabular, tmp_path):
    X, y, task, ctx, spec = temporal_case(tabular, tmp_path, 3)
    naive = DirectPipeline(spec.model_copy(update={"log_inverse": "naive"}), task)
    naive.fit(X, y, ctx)
    model = DirectPipeline(spec, task)
    model.fit(X, y, ctx)
    assert len(model.inverse_state["answer"]["fold_factors"]) == 2
    assert model.inverse_state["answer"]["scope"] == "fit_only_oof_residuals"
    schedule = model.summary.learning_curve["answer"]
    assert schedule["iterations"] == naive.summary.learning_curve["answer"]["iterations"]
    assert schedule["curve"] == naive.summary.learning_curve["answer"]["curve"]
    assert schedule["stopping_log_inverse"] == "naive"
    assert schedule["final_log_inverse"] == "oof_smearing"
    assert schedule["surrogate_differs_from_final_decision"]
    assert np.isfinite(model.predict(X.iloc[-10:], ctx).values["answer"]).all()


def test_stopping_does_not_learn_an_inverse_it_never_scores(tabular, tmp_path, monkeypatch):
    X, y, task, ctx, spec = temporal_case(tabular, tmp_path, 4)
    original = DirectPipeline._fit_inverse
    calls = []
    def tracked(self, fit_X, fit_y, local):
        calls.append((len(fit_X), bool(local.get("schedule_selection"))))
        return original(self, fit_X, fit_y, local)
    monkeypatch.setattr(DirectPipeline, "_fit_inverse", tracked)
    model = DirectPipeline(spec, task)
    model.fit(X, y, ctx)
    assert calls == [(len(X), False)]


def test_two_periods_still_cannot_learn_a_two_fold_inverse(tabular, tmp_path):
    X, y, task, ctx, spec = temporal_case(tabular, tmp_path, 2)
    with pytest.raises(ValueError, match="insufficient_oof_support"):
        DirectPipeline(spec.model_copy(update={"early_stopping_rounds": None}), task).fit(X, y, ctx)


def test_final_output_checkpoints_still_learn_their_inner_inverse(tabular, tmp_path, monkeypatch):
    X, y, task, ctx, spec = temporal_case(tabular, tmp_path, 4)
    original = DirectPipeline._fit_inverse
    calls = []
    def tracked(self, fit_X, fit_y, local):
        calls.append((len(fit_X), local.get("split_id")))
        return original(self, fit_X, fit_y, local)
    monkeypatch.setattr(DirectPipeline, "_fit_inverse", tracked)
    model = DirectPipeline(spec.model_copy(update={"checkpoint_counts": [4, 8]}), task)
    model.fit(X, y, ctx)
    assert any(rows < len(X) and split == "checkpoint" for rows, split in calls)
    assert model.inverse_state["answer"]


def test_controller_three_period_recipe_reaches_frozen_refit_csv(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    task = write_task(tmp_path, temporal=True)
    for name in ("train", "test"):
        frame = pd.read_csv(tmp_path / f"{name}.csv", dtype={"row": str})
        frame["when"] = np.arange(len(frame)) // 36 + 1 if name == "train" else 6
        if name == "train":
            frame["answer"] = frame.answer.abs() + 1
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    c = Controller(task, Limits(seconds=240, cpu=1), run_id="smearing_stopping")
    c.no_llm = c.qwen.disabled = True
    try:
        c.prepare()
        assert c.dataset.features(c.plan.fit)["when"].nunique() == 3
        spec = PipelineSpec(model="lightgbm", hyperparameters={"n_estimators": 12, "min_child_samples": 3},
            early_stopping_rounds=3, evaluation_interval=1, target_transform="log1p", log_inverse="oof_smearing")
        trained = payload(c.dispatch(ToolCommand(operation_id="three_period_fit", tool="train_model", arguments={
            "pipeline": spec.model_dump(), "budget_seconds": 35, "hypothesis": Hypothesis(
                observation="Offline three-period history fixture", expected_effect="Test the complete declared recipe",
                changed_component="inverse", acceptance="Independent final correction and frozen refit").model_dump()})))
        assert trained["result"]["status"] == "completed", trained
        identifier = trained["experiment"]["experiment_id"]
        prototype = json.loads((c.workdir / "attempts" / identifier / "worker_result.json").read_text())
        learned = prototype["frozen_pipeline"]["learned_inverse"]
        assert learned["answer"]["fold_factors"]
        # Pin this finalist only to test its production finalization path;
        # this fixture is not evidence of automatic selection or quality.
        monkeypatch.setattr(c, "prospective_finalists", lambda: ([identifier], {"fixture": "inverse replay"}))
        c.freeze("Technical inverse replay fixture")
        c.finalize()
        assert c.journal.get("refit_status") == "completed"
        final_dir = c.workdir / "finalization" / "refit" / identifier
        raw = json.loads((final_dir / "worker_result.json").read_text())
        assert raw["fit_rows"] == 180 and raw["frozen_pipeline"]["learned_inverse"] == learned
        with (final_dir / "model.pkl").open("rb") as stream:
            model = cloudpickle.load(stream)
        assert model.inverse_state == learned
        output = pd.read_csv(c.publisher.output, dtype={"row": str})
        sample = pd.read_csv(tmp_path / "sample_submission.csv", dtype={"row": str})
        assert output.row.tolist() == sample.row.tolist() and len(output) == 25
        assert np.isfinite(output.answer).all()
    finally:
        c.close()
