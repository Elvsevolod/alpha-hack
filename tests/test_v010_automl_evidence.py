import pandas as pd
import pytest
import sys
from types import ModuleType

from automl_agent.contracts import PipelineSpec
from automl_agent.search import SearchPipeline


@pytest.fixture
def predictor_api(monkeypatch):
    """These tests supply a fake predictor already; no real learner is exercised."""
    package = ModuleType('autogluon')
    package.__path__ = []
    tabular = ModuleType('autogluon.tabular')
    tabular.__path__ = []
    tabular.TabularPredictor = None
    trainer = ModuleType('autogluon.tabular.trainer')
    trainer.AutoTrainer = type('AutoTrainer', (), {'_train_single_full': lambda *_a, **_kw: []})
    package.tabular = tabular
    tabular.trainer = trainer
    for module in (package, tabular, trainer):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return tabular


@pytest.mark.parametrize("second_type,expected", [("LGBModel", "engine_stopped_unclassified"),
    ("CatBoostModel", "engine_space_exhausted")])
def test_two_models_of_one_family_do_not_prove_all_learners_completed(tabular, tmp_path, monkeypatch, second_type, expected, predictor_api):
    import autogluon.tabular

    class Predictor:
        def __init__(self, **kwargs):
            self.model_best = "first"

        def fit(self, **kwargs):
            assert kwargs["time_limit"] > 10
            self.fit_rows = list(kwargs["train_data"].index)
            self.tuning_rows = list(kwargs["tuning_data"].index)

        def refit_full(self, model, train_data_extra=None, **kwargs):
            assert kwargs["set_best_to_refit_full"] is True
            # refit_full already merges the saved inner validation split.
            assert train_data_extra is None
            return {model: model + "_FULL"}

        def leaderboard(self, **kwargs):
            assert kwargs["extra_info"] is True
            return pd.DataFrame({"model": ["first", "second"], "model_type": ["LGBModel", second_type],
                "fit_time": [.1, .1], "score_val": [-1., -2.]})

        def model_failures(self):
            return pd.DataFrame(columns=["model", "model_type", "exc_type", "total_time"])

        def info(self):
            return {"model_info": {"first": {"model_type": "LGBModel", "hyperparameters": {},
                "hyperparameters_fit": {}}}}

    monkeypatch.setattr(autogluon.tabular, "TabularPredictor", Predictor)
    X, y, task = tabular
    pipeline = SearchPipeline(PipelineSpec(engine="autogluon", model="library_search", learners=["GBM", "CAT"], n_trials=None), task)
    summary = pipeline.fit(X, y, {"strategy": "random", "cpu": 1, "budget_seconds": 30, "memory_mb":2048,
        "budget_mode": "full_comparison", "workdir": str(tmp_path)})
    evidence = summary.budget["targets"][task.targets[0]]
    assert evidence["completed_trials"] == 2
    assert evidence["stop_reason"] == expected


def test_autogluon_refit_retrains_frozen_selection_on_every_permitted_row(tabular, tmp_path, monkeypatch, predictor_api):
    import autogluon.tabular

    instances = []

    class Predictor:
        def __init__(self, **kwargs):
            self.model_best = "best"
            self.path = kwargs.get("path")
            self.refit_rows = []
            self.fit_called = False
            instances.append(self)

        def fit(self, data=None, **kwargs):
            assert data is None, "Frozen refit must not create another holdout search"
            self.fit_called = True
            self.fit_rows = list(kwargs["train_data"].index)
            self.tuning_rows = list(kwargs["tuning_data"].index)
            self.model_best = "best"

        def leaderboard(self, **kwargs):
            return pd.DataFrame({"model": ["best"], "model_type": ["LGBModel"],
                "fit_time": [.2], "score_val": [-1.]})

        def model_failures(self):
            return pd.DataFrame(columns=["model", "model_type", "exc_type", "total_time"])

        def info(self):
            return {"model_info": {"best": {"model_type": "LGBModel",
                "hyperparameters": {"n_estimators": 4, "min_data_in_leaf": 1},
                "hyperparameters_fit": {"num_boost_round": 3}}}}

        def refit_full(self, model, train_data_extra=None, **kwargs):
            assert train_data_extra is None
            if "X" in kwargs:
                assert kwargs["X_val"] is not None and kwargs["X_val"].empty
                assert kwargs["y_val"] is not None and kwargs["y_val"].empty
                assert kwargs["X"].index.equals(kwargs["y"].index)
                self.refit_rows = list(kwargs["X"].index)
            return {model: model + "_FULL"}

        def clone(self, path, return_clone=False):
            assert return_clone and path != self.path
            return Predictor(path=path)

        def transform_features(self, X):
            return X.copy()

        def transform_labels(self, y):
            return y.copy()

        def set_model_best(self, *args, **kwargs):
            pass

        def model_names(self):
            return ['best', 'best_FULL']

        def delete_models(self, **kwargs):
            assert kwargs['models_to_delete'] == ['best_FULL']

        @classmethod
        def load(cls, path, verbosity=0):
            return cls(path=path)

    monkeypatch.setattr(autogluon.tabular, "TabularPredictor", Predictor)
    X, y, task = tabular
    pipeline = SearchPipeline(PipelineSpec(engine="autogluon", model="library_search",
        learners=["GBM"], n_trials=None), task)
    context = {"strategy": "random", "cpu": 1, "budget_seconds": 30, "memory_mb":2048,
        "budget_mode": "full_comparison", "workdir": str(tmp_path)}
    pipeline.fit(X.iloc[:140], y.iloc[:140], context)

    initial = instances[0]
    assert set(initial.fit_rows) == set(pipeline.inner_split["fit_keys"])
    assert set(initial.tuning_rows) == set(pipeline.inner_split["validation_keys"])
    assert initial.refit_rows == []
    assert set(pipeline.training_keys) == set(X.index[:140])

    # Older pickles lack these metadata fields; the fitted model template still
    # holds the frozen training parameters.
    pipeline.selected[task.targets[0]].pop("learner", None)
    pipeline.selected[task.targets[0]].pop("parameters", None)
    pipeline.refit(X, y, context)
    final = instances[-1]
    assert not final.fit_called and initial.fit_called
    assert set(final.refit_rows) == set(X.index)
    assert initial.refit_rows == []
    assert pipeline.selected[task.targets[0]]["full_refit_rows"] == len(X)
    assert pipeline.selected[task.targets[0]]["full_refit_added_rows"] == len(X) - len(pipeline.autogluon_train_keys[task.targets[0]])
