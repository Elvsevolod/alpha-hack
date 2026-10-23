import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from automl_agent.api import prediction_bundle
from automl_agent.contracts import ExperimentResult, ExperimentSpec, FeedbackPolicy, Hypothesis, MetricSpec, OutputSpec, PipelineSpec, PredictionBundle, SelectionPolicy, TaskSpec, ValidationPlan
from automl_agent.evaluation import align_predictions, choose_leader, feedback_reports, score_predictions
from automl_agent.util import digest


def test_row_reordering_is_restored(tabular):
    X, y, task = tabular
    selected = X.iloc[[5, 1, 9]]
    context = {"task": task.model_dump(mode="json"), "protocol_hash": "protocol", "split_id": "validation"}
    b = prediction_bundle({"answer": [50, 10, 90]}, selected, context, "values")
    assert align_predictions(b, ["train:1", "train:5", "train:9"], task, "protocol", "validation")["answer"].tolist() == [10, 50, 90]


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra", "order_hash", "protocol", "data", "nan", "inf", "target"])
def test_invalid_prediction_is_rejected(tabular, mutation):
    X, y, task = tabular
    context = {"task": task.model_dump(mode="json"), "protocol_hash": "v", "split_id": "validation"}
    b = prediction_bundle({"answer": [1., 2., 3.]}, X.iloc[:3], context, "values")
    raw = b.model_dump()
    if mutation == "duplicate":
        raw["row_keys"][1] = raw["row_keys"][0]
        raw["row_order_hash"] = digest(raw["row_keys"])
    elif mutation == "missing": raw["row_keys"] = raw["row_keys"][:-1]
    elif mutation == "extra": raw["row_keys"].append("test:9")
    elif mutation == "order_hash": raw["row_order_hash"] = "wrong"
    elif mutation == "protocol": raw["protocol_hash"] = "wrong"
    elif mutation == "data": raw["data_version"] = "wrong"
    elif mutation == "target": raw["target_names"] = ["other"]
    else: raw["values"]["answer"][0] = float(mutation)
    with pytest.raises(ValueError):
        align_predictions(PredictionBundle.model_validate(raw), list(X.index[:3]), task, "v", "validation")


def classification_task():
    return TaskSpec(kind="classification", targets=["y"], id_columns=["id"], feature_columns=["x"],
        metric=MetricSpec(name="log_loss"), classes={"y": ["cat", "dog"]},
        output=OutputSpec(columns=["id", "cat_p", "dog_p"], prediction_type="probabilities",
            probability_columns={"cat_p": ("y", "cat"), "dog_p": ("y", "dog")}), source_hash="data")


def test_probability_class_order_is_explicit():
    task = classification_task()
    X = pd.DataFrame({"x": [1, 2]}, index=["a", "b"])
    b = prediction_bundle({"y": [[.1, .9], [.8, .2]]}, X,
        {"task": task.model_dump(), "protocol_hash": "p", "split_id": "validation"}, "probabilities", {"y": ["dog", "cat"]})
    pred = align_predictions(b, list(X.index), task, "p", "validation")
    assert np.allclose(pred["y"], [[.9, .1], [.2, .8]])
    assert score_predictions(task, pd.DataFrame({"y": ["cat", "dog"]}), pred, "probabilities") < .2


@pytest.mark.parametrize("metric", ["roc_auc", "log_loss", "average_precision"])
def test_probability_metrics_reject_label_output(metric):
    with pytest.raises(ValidationError, match="Probability metrics require probability output"):
        TaskSpec(kind="classification", targets=["y"], id_columns=["id"], feature_columns=["x"],
            metric=MetricSpec(name=metric, positive_label="dog" if metric != "log_loss" else None),
            classes={"y": ["cat", "dog"]},
            output=OutputSpec(columns=["id", "y"], prediction_type="labels"), source_hash="data")


def test_publisher_guards_probability_metric_at_final_boundary(tmp_path):
    from types import SimpleNamespace
    from automl_agent.publication import Publisher

    task = classification_task().model_copy(update={
        "metric": MetricSpec(name="roc_auc", positive_label="dog"),
        "output": OutputSpec(columns=["id", "y"], prediction_type="labels")})
    publisher = Publisher(SimpleNamespace(task=task, task_path=tmp_path / "task.md"), None, tmp_path, None)
    with pytest.raises(ValueError, match="Probability metrics require probability output"):
        publisher.frame(None)


@pytest.mark.parametrize("array", [[[.1, .1]], [[-1., 2.]], [[float('nan'), .5]], [[1.1, -.1]]])
def test_invalid_probabilities(array):
    task = classification_task()
    X = pd.DataFrame({"x": [1]}, index=["a"])
    b = prediction_bundle({"y": array}, X, {"task": task.model_dump(), "protocol_hash": "p", "split_id": "test"}, "probabilities")
    with pytest.raises(ValueError): align_predictions(b, ["a"], task, "p", "test")


@pytest.mark.parametrize("maximize,reference,candidate,expected", [
    (True,.5,.501,False),(True,.5,.50101,True),(False,10,9.99,False),(False,10,9.989,True),
    (True,-1,-.999,False),(True,-1,-.998,True),(False,0,0,False)])
def test_practical_tie_boundary(maximize, reference, candidate, expected):
    p = SelectionPolicy(maximize=maximize,min_delta_abs=.001,min_delta_rel=.001 if abs(reference)>1 else 0)
    assert p.improvement(candidate,reference) is expected


def test_cheap_replacements_cannot_drift_from_raw_best():
    p=SelectionPolicy(maximize=True,min_delta_abs=.01)
    scores=[.9,.895,.889]
    results=[ExperimentResult(experiment_id=str(i),status="completed",validation_hash="p",score=s,finish_seconds=3-i) for i,s in enumerate(scores)]
    leader,best=choose_leader(results,p)
    assert best==.9 and leader.experiment_id=="1"


@pytest.mark.parametrize("count,status", [(29,"suppressed"),(30,"available")])
def test_feedback_support_and_complements(tabular,count,status):
    _,_,task=tabular
    X=pd.DataFrame({c:np.ones(2*count) for c in task.feature_columns})
    X.loc[:count-1,"number"]=np.nan
    y=pd.DataFrame({"answer":np.arange(2*count,dtype=float)})
    plan=ValidationPlan(strategy="random",reason="fixture",fit=[1],validation=[2],control=[3],data_version=task.source_hash,
        feedback_policy=FeedbackPolicy(report_ids=["missingness"]),selection_policy=SelectionPolicy(maximize=False))
    report=feedback_reports(task,plan,X,y,{"answer":y.answer.to_numpy()},"values")["missingness"]
    assert report["status"]==status
    if status=="suppressed": assert "slices" not in report and "rows" not in report


def test_nested_search_and_unknown_fields_rejected():
    with pytest.raises(ValidationError): PipelineSpec(engine="flaml",search_owner="optuna")
    with pytest.raises(ValidationError): PipelineSpec(model="ridge",download_model=True)


def test_computation_hash_tracks_training_budget_but_not_ensemble_composition_budget():
    hypothesis = Hypothesis(observation="fixture", expected_effect="fixture", changed_component="fixture", acceptance="fixture")
    optuna_pipeline = PipelineSpec(engine="optuna", model="lightgbm", search_owner="optuna",
        search_space={"num_leaves": {"type": "int", "low": 8, "high": 16}})
    base = ExperimentSpec(experiment_id="first", hypothesis=hypothesis, pipeline=optuna_pipeline,
        task_hash="task", data_version="data", validation_hash="validation", budget_seconds=60,
        budget_mode="research")
    same_computation = base.model_copy(update={"experiment_id": "second"})
    more_budget = base.model_copy(update={"budget_seconds": 120})
    other_mode = base.model_copy(update={"budget_mode": "full_comparison"})
    assert base.computation_hash == same_computation.computation_hash
    assert base.computation_hash != more_budget.computation_hash
    assert base.computation_hash != other_mode.computation_hash

    ensemble = PipelineSpec(engine="ensemble", model="uniform_average", members=["one", "two"])
    combined = base.model_copy(update={"experiment_id": "ensemble_first", "pipeline": ensemble})
    combined_more_budget = combined.model_copy(update={"budget_seconds": 120, "budget_mode": "full_comparison"})
    assert combined.computation_hash == combined_more_budget.computation_hash
