import hashlib
import sys

import numpy as np
import pandas as pd

from automl_agent.contracts import MetricSpec, OutputSpec, PipelineSpec
from automl_agent.pipelines import DirectPipeline
from automl_agent.research_evaluation import oof_predictions
from test_plan_operators import context


def classification(tabular):
    X, y, task = tabular
    y = pd.DataFrame({"answer": np.where(y.answer < -.5, "first", np.where(y.answer > .7, "last", "middle"))}, index=y.index)
    task = task.model_copy(update={"kind": "classification", "classes": {"answer": ["middle", "last", "first"]},
        "metric": MetricSpec(name="f1"), "output": OutputSpec(columns=["answer"], prediction_type="labels")})
    return X, y, task


def test_pooled_decision_frozen_refit_keeps_class_order_and_state(tabular, tmp_path):
    X, y, task = classification(tabular)
    spec = PipelineSpec(model="logistic", prediction_policy="class_offsets", postprocess_fit_policy="pooled_oof", decision_method="regularized")
    ctx = context(task, tmp_path)
    model = DirectPipeline(spec, task)
    model.fit(X.iloc[:150], y.iloc[:150], ctx)
    frozen = PipelineSpec.model_validate_json(model.frozen_spec().model_dump_json())
    state = frozen.learned_decision["answer"]
    assert state["classes"] == task.classes["answer"] and state["folds_hash"]
    restored = DirectPipeline(frozen, task)
    restored.fit(X, y, {**ctx, "execution_mode": "refit"})
    assert restored.postprocessors["answer"] == state
    bundle = restored.predict(X.iloc[-20:], ctx)
    assert bundle.prediction_type == "labels" and bundle.classes == task.classes
    assert np.asarray(bundle.raw_probabilities["answer"]).shape == (20, 3)


def test_checkpoint_selection_uses_final_labels_inside_its_fit(tabular, tmp_path):
    X, y, task = classification(tabular)
    spec = PipelineSpec(model="lightgbm", hyperparameters={"n_estimators": 10, "min_child_samples": 3},
        checkpoint_counts=[4, 8], prediction_policy="class_offsets", postprocess_fit_policy="pooled_oof", decision_method="regularized")
    model = DirectPipeline(spec, task)
    model.fit(X, y, context(task, tmp_path))
    schedule = model.summary.learning_curve["answer"]
    assert schedule["iterations"] in [4, 8]
    assert schedule["decision_fit"] == "strictly_inside_checkpoint_fit"
    assert model.frozen_spec().iteration_schedule["answer"] in [4, 8]


def test_log_smearing_refit_retains_fit_only_correction(tabular, tmp_path, monkeypatch):
    X, y, task = tabular
    y = y.abs() + 1
    spec = PipelineSpec(model="lightgbm", hyperparameters={"n_estimators": 8}, target_transform="log1p", log_inverse="oof_smearing")
    model = DirectPipeline(spec, task)
    ctx = context(task, tmp_path)
    model.fit(X.iloc[:150], y.iloc[:150], ctx)
    frozen = model.frozen_spec()
    assert frozen.learned_inverse["answer"]["scope"] == "fit_only_oof_residuals"
    monkeypatch.setattr("automl_agent.research_evaluation.oof_predictions", lambda *a: (_ for _ in ()).throw(AssertionError("Frozen inverse must not retune")))
    refit = DirectPipeline(frozen, task)
    refit.fit(X, y, ctx)
    assert refit.inverse_state == model.inverse_state
    assert np.isfinite(refit.predict(X.iloc[-20:], ctx).values["answer"]).all()


def test_strict_oof_does_not_inherit_a_parent_surrogate_holdout(tabular, tmp_path, monkeypatch):
    X, y, task = tabular
    outsider_X, outsider_y = X.iloc[:1].copy(), y.iloc[:1].copy()
    outsider_X.index = outsider_y.index = ["outside_allowed_fit"]
    original = DirectPipeline.fit
    observed = []
    def checked(model, fit_X, fit_y, ctx):
        if ctx.get("schedule_validation"):
            assert "outside_allowed_fit" not in ctx["schedule_validation"][0].index
            observed.append(True)
        return original(model, fit_X, fit_y, ctx)
    monkeypatch.setattr(DirectPipeline, "fit", checked)
    ctx = {**context(task, tmp_path), "schedule_selection": True, "schedule_validation": (outsider_X, outsider_y)}
    report = oof_predictions(PipelineSpec(model="lightgbm", hyperparameters={"n_estimators": 8}, early_stopping_rounds=2), task, X, y, ctx)
    assert observed and report["prediction_role"] == "strict_oof"


def test_custom_oof_uses_a_new_hash_verified_factory_for_every_fold(tabular, tmp_path):
    X, y, task = tabular
    code = '''from automl_agent.pipelines import DirectPipeline
from automl_agent.contracts import PipelineSpec, TaskSpec
class Fresh:
    def fit(self, X, y, context):
        self.rows = set(X.index)
        self.model = DirectPipeline(PipelineSpec(model="ridge", seed=context["seed"]), TaskSpec.model_validate(context["task"]))
        self.summary = self.model.fit(X, y, context)
        return self.summary
    def predict(self, X, context):
        assert not self.rows.intersection(X.index)
        return self.model.predict(X, context)
def build_pipeline(context): return Fresh()
'''
    (tmp_path / "candidate.py").write_text(code)
    spec = PipelineSpec(engine="custom", model="ridge", custom_hash=hashlib.sha256(code.encode()).hexdigest(), custom_oof=True)
    ctx = {**context(task, tmp_path), "seed": 42}
    profile = sys.getprofile()
    try:
        custom = oof_predictions(spec, task, X, y, ctx)
    finally:
        sys.setprofile(profile)
    builtin = oof_predictions(PipelineSpec(model="ridge"), task, X, y, ctx)
    assert custom["folds_hash"] == builtin["folds_hash"]
    assert np.allclose(custom["values"]["answer"], builtin["values"]["answer"])
