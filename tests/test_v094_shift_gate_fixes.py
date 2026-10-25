"""Composition regressions for the two remaining v093 audit findings."""
import cloudpickle
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis, MetricSpec, OutputSpec, PipelineSpec, TaskSpec, TrainArgs, TrainingSummary
from automl_agent.controller import Controller
from automl_agent.pipelines import DirectPipeline
from automl_agent.shift import ShiftPipeline
from automl_agent.validation import InsufficientPilotSupport, pilot_indices, pilot_population
from conftest import write_task


def multiclass_data(seed=0):
    values, labels = make_classification(n_samples=1800, n_features=8,
        n_informative=6, n_redundant=0, n_classes=6, n_clusters_per_class=1,
        class_sep=2.5, flip_y=0, random_state=seed)
    columns = [f"x{i}" for i in range(8)]
    X = pd.DataFrame(values[:1200], columns=columns, index=[f"r:{i}" for i in range(1200)])
    y = pd.DataFrame({"y": labels[:1200]}, index=X.index)
    future = pd.DataFrame(values[1200:], columns=columns, index=[f"f:{i}" for i in range(600)])
    task = TaskSpec(kind="classification", targets=["y"], feature_columns=columns,
        classes={"y": list(range(6))}, metric=MetricSpec(name="f1", average="macro"),
        output=OutputSpec(columns=["y"], prediction_type="labels"), source_hash="offline_shift_regression")
    spec = PipelineSpec(model="extra_trees", hyperparameters={"n_estimators": 10}, shift_strategy="pseudo_label", seed=42)
    return X, y, future, task, spec


def context(task, future, directory):
    return {"cpu": 1, "memory_mb": 2048, "budget_seconds": 60,
        "strategy": "random", "protocol_hash": "shift_recheck", "data_hash": "synthetic",
        "split_id": "validation", "workdir": str(directory),
        "task": task.model_dump(mode="json"), "unlabelled_features": future}


def test_A25_viable_native_recipe_survives_smoke_support_selection(tmp_path):
    X, y, future, task, spec = multiclass_data()
    positions, support = pilot_population(X, y, task, "random", 80, 42, spec)
    assert positions.tolist() == list(range(len(X)))
    assert support["expanded_for_structural_support"]
    model = ShiftPipeline(spec, task)
    summary = model.fit(X.iloc[positions], y.iloc[positions], context(task, future, tmp_path))
    assert summary.rows == 1200 and model.evidence["student_gate"]["gain"] > 0


def test_A25_insufficient_full_population_is_detected_before_model_fit(tmp_path, monkeypatch):
    X, y, future, task, spec = multiclass_data()
    rows = pilot_indices(X, y, task, "random", 80, 42)
    monkeypatch.setattr(DirectPipeline, "fit", lambda *a, **kw: pytest.fail("Impossible pseudo pool reached training"))
    with pytest.raises(InsufficientPilotSupport, match="pseudo"):
        ShiftPipeline(spec, task).fit(X.iloc[rows], y.iloc[rows], context(task, future, tmp_path))


def test_A25_controller_uses_original_recipe_in_one_bounded_full_attempt(tmp_path, monkeypatch):
    from automl_agent.worker import read_frame
    c = Controller(write_task(tmp_path, kind="classification"), Limits(seconds=240, cpu=1, memory_mb=2048))
    try:
        c.no_llm = True
        c.prepare()
        pipeline = PipelineSpec(model="extra_trees", shift_strategy="pseudo_label", hyperparameters={"n_estimators": 137})
        spec = c.make_spec(TrainArgs(pipeline=pipeline, budget_seconds=30,
            hypothesis=Hypothesis(observation="A25 reduced quality gate rejected a viable recipe",
                expected_effect="Use the complete pseudo recipe", changed_component="pseudo support",
                acceptance="One bounded full attempt with unchanged model complexity")), "pseudo_full_recipe")
        calls = []
        def inspect(directory, request, seconds):
            calls.append((directory, seconds))
            assert directory.name == spec.experiment_id
            assert not request.get("pilot")
            assert request["pipeline"]["hyperparameters"]["n_estimators"] == 137
            assert len(read_frame(directory / "fit_X.json")) == len(c.plan.fit)
            assert .1 <= seconds <= spec.budget_seconds
            return {"status": "timeout", "error": "Controlled stop after checking full-recipe admission"}
        monkeypatch.setattr(c.executor, "run", inspect)
        outcome = c.execute_experiment(spec)
        assert len(calls) == 1 and outcome.status == "timeout"
        assert "Controlled stop" in outcome.error
    finally:
        c.close()


@pytest.mark.parametrize("strategy", ["random", "group", "exact_group", "temporal", "temporal_group"])
def test_A25_pseudo_preflight_preserves_outer_transfer_boundaries(strategy):
    from automl_agent.validation import pseudo_label_split
    rows = np.arange(2400)
    X = pd.DataFrame({"x": rows % 6, "when": rows // 240, "entity": rows // 24})
    y = pd.DataFrame({"y": rows % 6})
    task = TaskSpec(kind="classification", targets=["y"], feature_columns=list(X),
        time_column="when", group_column="entity", classes={"y": list(range(6))},
        metric=MetricSpec(name="f1", average="macro"), output=OutputSpec(columns=["y"], prediction_type="labels"),
        source_hash="split_fixture")
    teacher, pool, gate = pseudo_label_split(X, y, task, strategy, 42)
    assert not (set(teacher) & set(pool) or set(teacher) & set(gate) or set(pool) & set(gate))
    if strategy in {"temporal", "temporal_group"}:
        assert X.iloc[teacher].when.max() < X.iloc[pool].when.min() < X.iloc[gate].when.min()
    if strategy in {"group", "temporal_group"}:
        assert not set(X.iloc[teacher].entity) & set(X.iloc[gate].entity)
        assert not set(X.iloc[pool].entity) & set(X.iloc[gate].entity)


@pytest.mark.parametrize("seed,research_rows", [(3, 1200), (4, 1000)])
def test_A27_native_frozen_refit_rebuilds_independent_gate_state(tmp_path, monkeypatch, seed, research_rows):
    import automl_agent.decision as decision
    X, y, future, task, spec = multiclass_data(seed)
    spec = spec.model_copy(update={"prediction_policy": "class_offsets", "postprocess_fit_policy": "inner_holdout"})
    known, observed = {}, []
    original_decision, original_predict = decision.fit_decision, DirectPipeline.predict
    def capture(task, actual, probabilities, keys, *args, **kwargs):
        state = original_decision(task, actual, probabilities, keys, *args, **kwargs)
        known[state["row_keys_hash"]] = set(map(str, keys))
        return state
    def trace(model, frame, ctx):
        if ctx.get("split_id") == "student_gate":
            state = model.postprocessors["y"]
            assert not known[state["row_keys_hash"]] & set(map(str, frame.index))
            observed.append(state["row_keys_hash"])
        return original_predict(model, frame, ctx)
    monkeypatch.setattr(decision, "fit_decision", capture)
    monkeypatch.setattr(DirectPipeline, "predict", trace)
    ctx = context(task, future, tmp_path)
    fresh = ShiftPipeline(spec, task)
    fresh.fit(X.iloc[:research_rows], y.iloc[:research_rows], ctx)
    frozen = fresh.frozen_spec()
    frozen_state = cloudpickle.loads(cloudpickle.dumps(frozen.learned_decision))
    refit = ShiftPipeline(cloudpickle.loads(cloudpickle.dumps(frozen)), task)
    try:
        refit.fit(X, y, {**ctx, "execution_mode": "refit"})
    except ValueError as error:
        # A larger honest gate may reject the recipe; independence is mandatory
        # before the gain is observed. A negative gain must not become admission.
        assert research_rows < len(X) and "student did not improve" in str(error)
    else:
        assert refit.frozen_spec().learned_decision == frozen_state
    assert len(observed) == 4 and not refit.evidence["student_gate"]["evaluation_labels_used_for_training"]


@pytest.mark.parametrize("strategy", ["temporal", "temporal_group"])
@pytest.mark.parametrize("early_stopping", [None, 2])
def test_A08_A25_temporal_oof_declares_both_nested_pseudo_splits(strategy, early_stopping):
    from automl_agent.research_evaluation import oof_plan
    X, y, _, task, spec = multiclass_data()
    X = X.assign(when=np.arange(len(X)) // (240 if early_stopping else 300), entity=np.arange(len(X)) // 30)
    if early_stopping:
        spec = spec.model_copy(update={"model": "lightgbm", "early_stopping_rounds": early_stopping,
            "hyperparameters": {"n_estimators": 8}})
    task = task.model_copy(update={"feature_columns": list(X), "time_column": "when", "group_column": "entity"})
    parts, skipped = oof_plan(spec, task, X, y, strategy)
    assert len(parts) == 1 and len(skipped) == 2
    for fit, validation, fresh, _ in parts:
        assert X.iloc[fit].when.nunique() >= (4 if early_stopping else 3)
        assert X.iloc[fit].when.max() < X.iloc[validation].when.min()
        assert fresh.shift_strategy == "pseudo_label"


def test_A27_auxiliary_recipe_resets_learned_state_and_context_but_final_policy_is_frozen(tmp_path, monkeypatch):
    from automl_agent.api import prediction_bundle
    X, y, future, task, spec = multiclass_data()
    X = pd.DataFrame({"x": np.tile(np.arange(6), 200)}, index=X.index)
    future = X.iloc[:120].copy(); future.index = [f"f:{i}" for i in range(120)]
    y = pd.DataFrame({"y": X.x}, index=X.index)
    task = task.model_copy(update={"feature_columns": ["x"]})
    spec = spec.model_copy(update={"prediction_policy": "class_offsets", "postprocess_fit_policy": "inner_holdout",
        "class_offsets": [0.] * 6, "learned_decision": {"y": {"offsets": [0.] * 6}},
        "frozen_iterations": 10, "iteration_schedule": {"y": 10}, "iteration_scale_rows": 1200})
    fits = []
    class InspectDirect:
        def __init__(self, spec, task): self.spec, self.task = spec, task
        def fit(self, frame, actual, ctx):
            self.student = "sample_weights" in ctx
            if len(fits) < 2:
                assert not self.spec.learned_decision and self.spec.class_offsets is None
                assert not self.spec.iteration_schedule and self.spec.frozen_iterations is None
                assert self.spec.iteration_scale_rows is None and ctx["execution_mode"] == "train"
                assert all(name not in ctx for name in ("schedule_selection", "schedule_validation", "optuna_trial", "rolling_folds"))
                if len(fits) == 0: assert "sample_weights" not in ctx
            else:
                assert self.spec.learned_decision == spec.learned_decision
            fits.append(set(frame.index))
            return TrainingSummary(rows=len(frame), seconds=.01, budget={"selected_fit_actual_seconds": .01})
        def predict(self, frame, ctx):
            labels = frame.x.to_numpy().copy()
            p = np.full((len(frame), 6), .001)
            p[np.arange(len(frame)), labels] = .995
            if ctx["split_id"] == "student_gate":
                assert not fits[0] & set(frame.index) and not fits[1] & set(frame.index)
                if not self.student: labels[0] = (labels[0] + 1) % 6
            return prediction_bundle({"y": labels.tolist()}, frame, ctx, "labels", task.classes, raw_probabilities={"y": p.tolist()})
        def frozen_spec(self): return self.spec
    monkeypatch.setattr("automl_agent.pipelines.DirectPipeline", InspectDirect)
    ctx = {**context(task, future, tmp_path), "execution_mode": "refit", "schedule_selection": True,
        "schedule_validation": (X, y), "rolling_folds": [], "optuna_trial": object(),
        "sample_weights": pd.Series(1., index=X.index)}
    model = ShiftPipeline(spec, task)
    model.fit(X, y, ctx)
    assert len(fits) == 4 and model.frozen_spec().learned_decision == spec.learned_decision
