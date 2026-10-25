import pandas as pd
import importlib.util
import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis, PipelineSpec, TrainArgs
from automl_agent.controller import Controller
from automl_agent.search import SearchPipeline
from conftest import write_task


@pytest.mark.parametrize("copies", [1, 100])
@pytest.mark.skipif(importlib.util.find_spec('autogluon') is None,
    reason='Real AutoGluon pilot requires organizer dependency environment')
def test_pilot_has_its_own_phase_reserves_on_small_and_large_data(tmp_path, monkeypatch, copies):
    path = write_task(tmp_path, kind="classification")
    frame = pd.read_csv(tmp_path / "train.csv", dtype=str, keep_default_na=False)
    frame = pd.concat([frame] * copies, ignore_index=True)
    frame["row"] = [f"train_{i:07d}" for i in range(len(frame))]
    frame.to_csv(tmp_path / "train.csv", index=False)
    c = Controller(path, Limits(seconds=10500, cpu=1, memory_mb=2048), budget_mode="full_comparison")
    try:
        c.prepare()
        # Reproduce the measured full-data reserve that exceeded the pilot budget.
        monkeypatch.setattr(c, "finalist_costs", lambda: [92.])
        pipeline = PipelineSpec(engine="autogluon", model="library_search", learners=["GBM", "CAT", "XGB"], n_trials=None, seed=17)
        args = TrainArgs(pipeline=pipeline, hypothesis=Hypothesis(observation="Observed pilot reserve overflow",
            expected_effect="Train a finite pilot before the separate full search", changed_component="pilot_phase_budget",
            acceptance="Actual small-data AutoGluon fit with feasible phase reserves"), budget_seconds=120)
        spec = c.make_spec(args, "pilot_budget_regression")
        original = c.executor.run
        observed = []
        full_attempts = []
        def inspect(directory, request, seconds):
            if directory.name != "smoke":
                full_attempts.append(directory)
                return {"status": "timeout", "error": "Controlled stop after the real pilot; no full-data model needed"}
            phases = request["context"]["phase_estimates"]
            assert phases["selected_fit_seconds"] + phases["prediction_seconds"] + phases["publication_seconds"] < seconds
            assert request["context"]["requested_budget_seconds"] == seconds
            assert request["pipeline"]["n_trials"] == 1 and request["context"]["budget_mode"] == "research"
            result = original(directory, request, seconds)
            observed.append(result)
            return result
        monkeypatch.setattr(c.executor, "run", inspect)
        outcome = c.execute_experiment(spec)
        assert observed[0]["status"] == "completed", observed[0]
        budget = observed[0]["training_summary"]["budget"]
        target = budget["targets"][c.task.targets[0]]
        assert budget["n_trials"] == 1 and target["configured_learners"] == ["GBM"]
        assert target["stop_reason"] == "trial_limit_reached"
        assert spec.pipeline.n_trials is None and spec.budget_mode == "full_comparison"
        assert outcome.status == "timeout"
        # Under shared CPU load the measured extrapolation can already reject
        # the full attempt. Both paths preserve the successful real pilot and
        # protect finalization; this test must not require a particular speed.
        if full_attempts:
            assert "Controlled stop" in outcome.error
        else:
            diagnostic = outcome.training_summary["diagnostic"]
            assert diagnostic["category"] == "budget_admission"
            assert diagnostic["scope"] == "fit_only_no_external_metric"
            assert diagnostic["reason"] in {"pilot_completed_candidate_rejected", "pilot_completed_candidate_window_exhausted"}
            assert diagnostic["pilot_seconds"] > 0 and diagnostic["finalization_reserve_seconds"] > 0
            if diagnostic["reason"] == "pilot_completed_candidate_rejected":
                assert diagnostic["candidate_fit_seconds_required"] > diagnostic["candidate_window_seconds_remaining"]
                assert diagnostic["candidate_fit_shortfall_seconds"] > 0
        assert outcome.score is None
    finally:
        c.close()


def test_infeasible_phase_reserve_is_rejected_before_calling_engine(tabular, tmp_path, monkeypatch):
    import flaml
    monkeypatch.setattr(flaml.AutoML, "fit", lambda *a, **kw: pytest.fail("Infeasible search reached the engine"))
    X, y, task = tabular
    pipeline = SearchPipeline(PipelineSpec(engine="flaml", model="library_search", n_trials=None), task)
    with pytest.raises(ValueError, match="phase reserves exceed"):
        pipeline.fit(X, y, {"strategy": "random", "cpu": 1, "budget_seconds": 2, "workdir": str(tmp_path),
            "phase_estimates": {"selected_fit_seconds": 10, "prediction_seconds": 1, "publication_seconds": 1}})


@pytest.mark.skipif(importlib.util.find_spec('autogluon') is None,
    reason='Real AutoGluon seed integration requires organizer dependency environment')
def test_autogluon_actual_training_seed_and_finite_model_policy(tabular, tmp_path):
    X, y, task = tabular
    spec = PipelineSpec(engine="autogluon", model="library_search", learners=["GBM", "CAT", "XGB"], n_trials=None, seed=73)
    pipeline = SearchPipeline(spec, task)
    summary = pipeline.fit(X, y, {"strategy": "random", "cpu": 1, "budget_seconds": 35, "memory_mb":2048, "workdir": str(tmp_path)})
    evidence = summary.budget["targets"][task.targets[0]]
    assert evidence["training_seed"] == 73
    assert summary.budget["selected_fit_rows"] == len(X)
    assert set(pipeline.training_keys) == set(X.index)  # cached fit and tuning rows are already refitted once
    assert evidence["completed_learners"] == ["GBM", "CAT", "XGB"]
    assert evidence["stop_reason"] == "engine_space_exhausted"
    predictor = pipeline.models[task.targets[0]][1]
    assert predictor.model_best == pipeline.selected[task.targets[0]]["refitted_model"]
    info = predictor.info()["model_info"]
    actual = [record["hyperparameters"] for record in info.values()]
    assert any(parameters.get("seed_value") == 73 for parameters in actual)
    assert any(parameters.get("random_seed") == 73 for parameters in actual)
    assert any(parameters.get("random_state") == 73 for parameters in actual)
