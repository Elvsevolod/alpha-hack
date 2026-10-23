import pandas as pd
import pytest

from automl_agent.contracts import PipelineSpec
from automl_agent.search import SearchPipeline


@pytest.mark.parametrize("second_type,expected", [("LGBModel", "engine_stopped_unclassified"),
    ("CatBoostModel", "engine_space_exhausted")])
def test_two_models_of_one_family_do_not_prove_all_learners_completed(tabular, tmp_path, monkeypatch, second_type, expected):
    import autogluon.tabular

    class Predictor:
        def __init__(self, **kwargs):
            self.model_best = "first"

        def fit(self, **kwargs):
            assert kwargs["time_limit"] > 10

        def leaderboard(self, **kwargs):
            assert kwargs["extra_info"] is True
            return pd.DataFrame({"model": ["first", "second"], "model_type": ["LGBModel", second_type],
                "fit_time": [.1, .1], "score_val": [-1., -2.]})

        def model_failures(self):
            return pd.DataFrame(columns=["model", "model_type", "exc_type", "total_time"])

        def info(self):
            return {"model_info": {"first": {"hyperparameters": {}}}}

    monkeypatch.setattr(autogluon.tabular, "TabularPredictor", Predictor)
    X, y, task = tabular
    pipeline = SearchPipeline(PipelineSpec(engine="autogluon", model="library_search", learners=["GBM", "CAT"], n_trials=None), task)
    summary = pipeline.fit(X, y, {"strategy": "random", "cpu": 1, "budget_seconds": 30,
        "budget_mode": "full_comparison", "workdir": str(tmp_path)})
    evidence = summary.budget["targets"][task.targets[0]]
    assert evidence["completed_trials"] == 2
    assert evidence["stop_reason"] == expected
