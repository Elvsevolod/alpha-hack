from automl_agent.tool_results import payload as tool_payload
import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.compatibility import incompatibilities, require_compatible
from automl_agent.contracts import InspectArgs, PipelineSpec, ReferenceArgs, ToolCommand, Transformation
from automl_agent.controller import Controller
from automl_agent.data import Dataset
from automl_agent.preprocessing import FeatureMatrix
from automl_agent.validation import inner_split, stratified_parts
from conftest import write_task


@pytest.mark.parametrize("representation", ["native", "tabular"])
@pytest.mark.parametrize("operation", ["select_kbest", "drop_constant", "drop_correlated", "standard_scale", "yeo_johnson", "spline"])
def test_native_transform_rejected_before_fitting(tabular, representation, operation):
    X, y, task = tabular
    spec = PipelineSpec(model="lightgbm", representation=representation, transformations=[Transformation(name=operation)])
    with pytest.raises(ValueError, match="native_transformations_unsupported"):
        FeatureMatrix(spec, task).fit(X, y)


def test_transform_order_and_persisted_feature_schema(tabular):
    X, y, task = tabular
    bad = PipelineSpec(model="ridge", transformations=[Transformation(name="select_kbest"), Transformation(name="log1p")])
    with pytest.raises(ValueError, match="stage_order"):
        FeatureMatrix(bad, task).fit(X, y)
    spec = PipelineSpec(model="ridge", transformations=[Transformation(name="robust_scale"),
        Transformation(name="standard_scale"), Transformation(name="select_kbest", parameters={"k": 2})])
    fitted = FeatureMatrix(spec, task).fit(X, y)
    loaded = cloudpickle.loads(cloudpickle.dumps(fitted))
    assert len(loaded.output_features) == loaded.width == 2
    assert loaded.processing_summary["implicit_numeric_transforms"] == ["RobustScaler", "StandardScaler"]
    assert np.allclose(fitted.transform(X).toarray(), loaded.transform(X).toarray())


def test_multioutput_singleton_and_all_indicators():
    y = pd.DataFrame({"first": ["a", "b"] * 60, "second": ["common"] * 116 + ["rare"] * 3 + ["singleton"]})
    fit, val, control = stratified_parts(y, 42)
    assert 119 in fit
    assert len(set(fit + val + control)) == len(y)
    for rows in (fit, val, control):
        assert "rare" in set(y.iloc[rows].second)
    assert not set(fit) & set(val + control)


def test_multioutput_inner_split_keeps_second_target_singleton_in_fit(tmp_path):
    path = write_task(tmp_path, "classification")
    dataset = Dataset(path)
    task, _ = dataset.draft()
    task = dataset.activate(task)
    y = dataset.train[task.targets].copy()
    y["other"] = "common"
    y.loc[y.index[-1], "other"] = "single"
    task = task.model_copy(update={"kind": "multioutput_classification", "targets": ["answer", "other"],
        "classes": {**task.classes, "other": ["common", "single"]}})
    fit, val = inner_split(dataset.features(), y, task, "random", 42)
    assert len(y) - 1 in fit and len(y) - 1 not in val


@pytest.mark.parametrize("description", ["calendar year of observation", "year of observation", "календарный год наблюдения", "год наблюдения"])
def test_renamed_time_semantics(tmp_path, description):
    path = write_task(tmp_path, temporal=True)
    (tmp_path / "task_spec.json").unlink()
    path.write_text("Regression task. Primary metric: RMSE.\ntrain.csv test.csv sample_submission.csv\np: " + description)
    for filename in ("train.csv", "test.csv"):
        frame = pd.read_csv(tmp_path / filename, dtype=str)
        frame = frame.rename(columns={"when": "p"})
        frame.to_csv(tmp_path / filename, index=False)
    task, questions = Dataset(path).draft()
    assert task and task.time_column == "p" and not questions


def test_ambiguous_temporal_transfer_requires_proposal_but_increasing_id_does_not(tmp_path):
    path = write_task(tmp_path, temporal=True)
    (tmp_path / "task_spec.json").unlink()
    path.write_text("Regression. Primary metric: RMSE. Next period transfer.\ntrain.csv test.csv sample_submission.csv")
    dataset = Dataset(path)
    task, questions = dataset.draft()
    assert task is None and "time_column" in questions
    assert "structural_features" in dataset.sources
    for filename in ("train.csv", "test.csv"):
        frame = pd.read_csv(tmp_path / filename, dtype=str).drop(columns="when")
        frame.to_csv(tmp_path / filename, index=False)
    task, questions = Dataset(path).draft()
    assert task and task.time_column is None and not questions


def test_catalog_executor_agree_and_cursor_tracks_registry(tmp_path):
    controller = Controller(write_task(tmp_path), Limits(seconds=120, cpu=1))
    try:
        controller.prepare()
        task = controller.task.model_copy(update={"text_columns": ["code"], "categorical_columns": []})
        args = ReferenceArgs(mode="recipes", reference_id="flaml")
        result = controller.references.catalog_entry("recipes", args.reference_id, task, controller.plan)
        assert result["applicable"] is False
        assert "automl_requires_tabular_nontext_task" in result["exclusion_reasons"]
        with pytest.raises(ValueError, match="nontext"):
            require_compatible(PipelineSpec.model_validate(result["pipeline"]), task, controller.plan.strategy, controller.capabilities)
        first = controller.references.query(ReferenceArgs(mode="api", query="sklearn.linear_model.Ridge", max_chars=2000), task, controller.plan)
        assert first["next_cursor"]
        controller.capabilities.versions["flaml"] = "absent"
        repeated = controller.references.query(ReferenceArgs(mode="api", query="sklearn.linear_model.Ridge",
            max_chars=2000, cursor=first["next_cursor"]), task, controller.plan)
        assert repeated["status"] == "stale_cursor"
    finally:
        controller.close()


@pytest.mark.parametrize("candidate_limit", [None, 8])
def test_protocol_revision_commits_with_unlimited_or_finite_budget(tmp_path, candidate_limit):
    controller = Controller(write_task(tmp_path, temporal=True), Limits(seconds=150, cpu=1, max_candidates=candidate_limit))
    try:
        controller.prepare()
        # The temporal split safely permits a comparison using a random research
        # split while its held-out final period remains untouched.
        old = controller.plan
        result = tool_payload(controller.dispatch(ToolCommand(operation_id="revise", tool="inspect_dataset", arguments={
            "report": "validation_proposal", "proposed_strategy": "random", "rationale": "Compare research split only; retain closed next-period control"})))
        assert result["status"] == "revised", result
        assert controller.plan.control == old.control and controller.plan.hash != old.hash
        assert controller.journal.get("publication")["validation_hash"] == controller.plan.hash
        assert controller.lookup(controller.journal.get("leader_id"))[0].validation_hash == controller.plan.hash
    finally:
        controller.close()


def test_protocol_revision_failure_retains_old_state(tmp_path, monkeypatch):
    controller = Controller(write_task(tmp_path, temporal=True), Limits(seconds=150, cpu=1))
    try:
        controller.prepare()
        old = controller.plan.hash
        publication = controller.publisher.output.read_bytes()
        deadline = controller.budget.deadline
        monkeypatch.setattr(controller, "execute_experiment", lambda spec, **kw: controller.failed_result(spec, {"status": "timeout"}))
        result = controller.revise_validation(InspectArgs(report="validation_proposal", proposed_strategy="random", rationale="test failure"))
        assert result["status"] == "revision_aborted"
        assert controller.plan.hash == old
        assert controller.publisher.output.read_bytes() == publication and controller.budget.deadline == deadline
    finally:
        controller.close()
