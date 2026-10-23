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
            self.fit_rows = list(kwargs["train_data"].index)
            self.tuning_rows = list(kwargs["tuning_data"].index)

        def refit_full(self, model, train_data_extra, **kwargs):
            assert kwargs["set_best_to_refit_full"] is True
            assert list(train_data_extra.index) == self.tuning_rows
            return {model: model + "_FULL"}

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


def test_autogluon_refit_uses_inner_holdout_and_later_new_rows(tabular, tmp_path, monkeypatch):
    import autogluon.tabular

    instances = []

    class Predictor:
        def __init__(self, **kwargs):
            self.model_best = "best"
            self.path = kwargs.get("path")
            instances.append(self)

        def fit(self, **kwargs):
            self.fit_rows = list(kwargs["train_data"].index)
            self.tuning_rows = list(kwargs["tuning_data"].index)

        def leaderboard(self, **kwargs):
            return pd.DataFrame({"model": ["best"], "model_type": ["LGBModel"],
                "fit_time": [.2], "score_val": [-1.]})

        def model_failures(self):
            return pd.DataFrame(columns=["model", "model_type", "exc_type", "total_time"])

        def info(self):
            return {"model_info": {"best": {"hyperparameters": {}}}}

        def refit_full(self, model, train_data_extra=None, **kwargs):
            extra = [] if train_data_extra is None else list(train_data_extra.index)
            self.refit_rows = extra
            return {model: model + "_FULL"}

        @classmethod
        def load(cls, path, verbosity=0):
            return cls(path=path)

    monkeypatch.setattr(autogluon.tabular, "TabularPredictor", Predictor)
    X, y, task = tabular
    pipeline = SearchPipeline(PipelineSpec(engine="autogluon", model="library_search",
        learners=["GBM"], n_trials=None), task)
    context = {"strategy": "random", "cpu": 1, "budget_seconds": 30,
        "budget_mode": "full_comparison", "workdir": str(tmp_path)}
    pipeline.fit(X.iloc[:140], y.iloc[:140], context)

    initial = instances[0]
    assert set(initial.fit_rows) == set(pipeline.inner_split["fit_keys"])
    assert set(initial.tuning_rows) == set(pipeline.inner_split["validation_keys"])
    assert set(initial.refit_rows) == set(pipeline.inner_split["validation_keys"])
    assert set(pipeline.training_keys) == set(pipeline.inner_split["fit_keys"])

    pipeline.refit(X, y, context)
    final = instances[-1]
    expected = set(pipeline.inner_split["validation_keys"]) | set(X.index[140:])
    assert set(final.refit_rows) == expected
    assert len(final.refit_rows) == len(expected)
