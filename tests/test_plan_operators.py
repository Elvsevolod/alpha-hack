from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import PipelineSpec, MetricSpec, OutputSpec, TaskSpec
from automl_agent.decision import decide, fit_decision
from automl_agent.pipelines import DirectPipeline, build_pipeline, validate_parameters
from automl_agent.research_evaluation import oof_predictions
from automl_agent.search import SearchPipeline, autogluon_portfolio
from automl_agent.util import read_json


def test_flaml_smoke_keeps_a_replayable_search_and_refuses_one_trial(tmp_path):
    from automl_agent.budget import Limits
    from automl_agent.contracts import ToolCommand
    from automl_agent.controller import Controller
    from automl_agent.compatibility import require_compatible
    from automl_agent.tool_results import payload
    from conftest import write_task
    controller = Controller(write_task(tmp_path), Limits(seconds=120, cpu=1, memory_mb=2048), run_id="flaml_replay")
    try:
        controller.no_llm = True
        controller.prepare()
        with pytest.raises(ValueError, match="n_trials_at_least_two"):
            require_compatible(PipelineSpec(engine="flaml", model="automl", learners=["lgbm"], n_trials=1), controller.task, "random")
        result = payload(controller.dispatch(ToolCommand(operation_id="replayable", tool="run_automl", arguments={
            "pipeline": {"engine": "flaml", "model": "automl", "learners": ["lgbm"], "n_trials": 2},
            "budget_seconds": 45, "hypothesis": {"observation": "Pinned one-trial FLAML writes no training log",
                "expected_effect": "Preserve a replayable selected model", "changed_component": "smoke trial count",
                "acceptance": "Actual smoke and final fit both retain logged configurations"}})))
        assert result["result"]["status"] == "completed", result
        requests = list(controller.workdir.glob("attempts/*/smoke/request.json"))
        flaml = [path for path in requests if read_json(path)["pipeline"]["engine"] == "flaml"]
        assert flaml and all(read_json(path)["pipeline"]["n_trials"] == 2 for path in flaml)
        assert all(any(path.parent.glob("flaml_*.jsonl")) for path in flaml)
    finally:
        controller.close()


def context(task, directory):
    directory.mkdir(exist_ok=True)
    return {"cpu": 1, "memory_mb": 512, "budget_seconds": 30, "strategy": "random", "protocol_hash": "fixed",
        "task": task.model_dump(mode="json"), "split_id": "validation", "workdir": str(directory)}


def test_knn_subset_scaler_and_serialization_are_fit_only(tabular, tmp_path):
    import cloudpickle
    X, y, task = tabular
    spec = PipelineSpec(model="knn", include_features=["number", "second"], hyperparameters={"n_neighbors": 3})
    model = DirectPipeline(spec, task)
    ctx = context(task, tmp_path)
    model.fit(X.iloc[:120], y.iloc[:120], ctx)
    before = model.preprocessor.numeric_transformers[0].mean_.copy()
    query = X.iloc[120:].copy()
    query["number"] *= 1000
    prediction = model.predict(query, ctx)
    assert np.array_equal(before, model.preprocessor.numeric_transformers[0].mean_)
    assert model.preprocessor.feature_columns == ["number", "second"]
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    assert prediction.values == restored.predict(query[list(reversed(query.columns))], ctx).values
    assert np.isfinite(prediction.values["answer"]).all()
    bad = spec.model_copy(update={"include_features": ["answer"]})
    with pytest.raises(ValueError, match="confirmed features"):
        DirectPipeline(bad, task).fit(X, y, ctx)


def test_native_lightgbm_objective_parameters_and_alias_conflicts(tabular, tmp_path):
    X, y, task = tabular
    ctx = context(task, tmp_path)
    for objective, key, value in (("huber", "alpha", .8), ("fair", "fair_c", 1.5), ("tweedie", "tweedie_variance_power", 1.3)):
        spec = PipelineSpec(model="lightgbm", training_objective=objective,
            hyperparameters={"n_estimators": 5, key: value, "cat_smooth": 4., "cat_l2": 2., "min_data_per_group": 5})
        actual = y.abs() + 1 if objective == "tweedie" else y
        model = DirectPipeline(spec, task)
        model.fit(X, actual, ctx)
        assert model.models["answer"].get_params()[key] == value
    for parameters in ({"alpha": .5}, {"fair_c": -1}, {"n_estimators": 5, "num_iterations": 7}, {"min_data_per_group": 0}):
        with pytest.raises(ValueError):
            validate_parameters(PipelineSpec(model="lightgbm", hyperparameters=parameters), task)


def test_short_hpo_continues_unique_trials_and_invalidates_changed_rows(tabular, tmp_path):
    X, y, task = tabular
    spec = PipelineSpec(engine="optuna", model="ridge", n_trials=4,
        search_space={"alpha": {"type": "float", "low": .01, "high": 20., "log": True}})
    ctx = context(task, tmp_path / "first")
    first = SearchPipeline(spec, task)
    summary = first.fit(X, y, ctx)
    history = read_json(Path(ctx["workdir"]) / "hpo_history.json")
    second_ctx = context(task, tmp_path / "second")
    second_ctx["hpo_histories"] = {history["identity"]: history["trials"]}
    second = SearchPipeline(spec, task)
    result = second.fit(X, y, second_ctx)
    saved = read_json(Path(second_ctx["workdir"]) / "hpo_history.json")
    assert summary.budget["sampler"] == "tpe" and not summary.budget["pruning_callbacks"]
    assert result.budget["continued_completed_trials"] == 4
    assert len({item["parameters_hash"] for item in saved["trials"]}) == 8
    changed_ctx = context(task, tmp_path / "changed")
    changed_ctx["hpo_histories"] = second_ctx["hpo_histories"]
    changed = SearchPipeline(spec, task).fit(X.iloc[:-1], y.iloc[:-1], changed_ctx)
    assert changed.budget["continued_completed_trials"] == 0


def test_oof_hash_is_shared_by_recipes_but_provenance_is_not(tabular, tmp_path):
    X, y, task = tabular
    ctx = context(task, tmp_path)
    a = oof_predictions(PipelineSpec(model="ridge"), task, X, y, ctx)
    b = oof_predictions(PipelineSpec(model="ridge", hyperparameters={"alpha": 2}), task, X, y, ctx)
    assert a["folds_hash"] == b["folds_hash"]
    assert a["folds"][0]["scope"]["recipe"] != b["folds"][0]["scope"]["recipe"]


def test_final_decision_regularization_and_common_shift_invariance():
    task = TaskSpec(kind="classification", targets=["y"], feature_columns=["x"], classes={"y": ["a", "b", "c"]},
        metric=MetricSpec(name="f1", average="macro"), output=OutputSpec(columns=["y"], prediction_type="labels"), source_hash="fixture")
    p = np.tile([.45, .35, .2], (100, 1))
    y = pd.DataFrame({"y": ["a"] * 49 + ["b"] * 50 + ["c"]})
    state = fit_decision(task, y, p, list(range(100)))
    assert sum(state["offsets"]) == pytest.approx(0)
    assert state["penalty"][2] > state["penalty"][0]
    shifted = {**state, "offsets": (np.asarray(state["offsets"]) + .1).tolist()}
    assert np.array_equal(decide(p, task.classes["y"], state), decide(p, task.classes["y"], shifted))
    assert np.array_equal(decide(p, task.classes["y"]), np.asarray(task.classes["y"])[p.argmax(axis=1)])


def test_explicit_autogluon_portfolio_preserves_configuration_count(tabular):
    _, _, task = tabular
    spec = PipelineSpec(engine="autogluon", model="automl", n_trials=1, learners=["GBM"],
        autogluon_configurations={"GBM": [{"num_leaves": 7}, {"num_leaves": 15}]})
    configs = autogluon_portfolio(spec, task, ["GBM"])
    assert len(configs["GBM"]) == 2 and configs["GBM"][0]["num_leaves"] != configs["GBM"][1]["num_leaves"]
    with pytest.raises(ValueError, match="managed resources"):
        autogluon_portfolio(spec.model_copy(update={"autogluon_configurations": {"GBM": [{"num_threads": 99}]}}), task, ["GBM"])


def test_rolling_predictions_do_not_read_future_labels(tabular, tmp_path):
    X, y, task = tabular
    ctx = context(task, tmp_path)
    ctx["rolling_folds"] = [{"fit": list(range(60)), "validation": list(range(60, 120))},
        {"fit": list(range(120)), "validation": list(range(120, 180))}]
    model = build_pipeline(PipelineSpec(model="ridge"), task, ctx)
    model.fit(X, y, ctx)
    old = model.predict(X.iloc[60:120], ctx).values
    changed = y.copy()
    changed.iloc[120:] *= 100
    new = build_pipeline(PipelineSpec(model="ridge"), task, ctx)
    new.fit(X, changed, ctx)
    assert old == new.predict(X.iloc[60:120], ctx).values
