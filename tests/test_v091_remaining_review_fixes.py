"""Behavioral regressions for the two remaining independent review findings."""
import re
import time

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.clustering_route import _affirmative_mention, resolve_and_run
from automl_agent.contracts import MetricSpec, OutputSpec, PipelineSpec, TaskSpec
from automl_agent.qwen import QwenClient
from automl_agent.search import SearchPipeline


@pytest.mark.parametrize("text,expected", [
    ("Без меток выполните кластеризацию.", True),
    ("Без учителя выполните кластеризацию.", True),
    ("Use no labels for clustering.", True),
    ("Without labels perform clustering.", True),
    ("Do not use labels for clustering.", True),
    ("Perform clustering without labels.", True),
    ("This is not a clustering task. Perform clustering without labels.", True),
    ("Not only clustering is allowed.", True),
    ("This is not a clustering task.", False),
    ("Do not perform clustering.", False),
    ("Never use clustering.", False),
    ("No clustering is required.", False),
    ("Solve without using clustering.", False),
    ("Решите без использования кластеризации.", False),
    ("Не выполняйте кластеризацию.", False),
    ("Кластеризация не требуется.", False),
])
def test_clustering_negation_belongs_to_the_action(text, expected):
    accepted = any(_affirmative_mention(text, match)
        for match in re.finditer(r'\bclustering\b|\bкластеризац\w*', text, re.I))
    assert accepted is expected


@pytest.mark.parametrize("instruction", [
    "Без меток выполните кластеризацию.",
    "Без учителя выполните кластеризацию.",
    "Use no labels for clustering.",
])
def test_clustering_without_labels_reaches_resolver_and_publishes_csv(tmp_path, monkeypatch, instruction):
    text = instruction + " Use two clusters. Metric: silhouette. Files: observations.csv and future.csv. Output columns id,cluster."
    (tmp_path / "task.md").write_text(text)
    rng = np.random.default_rng(42)
    pd.DataFrame({"id": np.arange(80), "x": np.r_[rng.normal(-5, .1, 40), rng.normal(5, .1, 40)]}).to_csv(
        tmp_path / "observations.csv", index=False)
    pd.DataFrame({"id": [80, 81], "x": [-5., 5.]}).to_csv(tmp_path / "future.csv", index=False)
    calls = []

    def resolve(self, *args, **kwargs):
        calls.append(1)
        return {"contract": {"train_file": "observations.csv", "test_file": "future.csv",
            "features": ["x"], "id_columns": ["id"], "n_clusters": 2}, "evidence_quote": text}

    monkeypatch.setattr(QwenClient, "resolve", resolve)
    result = resolve_and_run(tmp_path / "task.md", Limits(seconds=60, cpu=1, memory_mb=2048), time.time(), "labels")
    assert calls == [1] and result and result["state"] == "completed"
    output = pd.read_csv(tmp_path / "submission.csv")
    assert list(output) == ["id", "cluster"] and output.id.tolist() == [80, 81]
    assert output.cluster.isin([0, 1]).all() and output.cluster.nunique() == 2


@pytest.mark.parametrize("seed", [42, 17])
def test_autogluon_refit_keeps_measured_iterations_and_uses_every_row(tmp_path, monkeypatch, seed):
    pytest.importorskip("autogluon.tabular")
    from autogluon.tabular.models.lgb.lgb_model import LGBModel

    history = []
    original = LGBModel._fit

    def watch(self, X, y, *args, **kwargs):
        result = original(self, X, y, *args, **kwargs)
        history.append({"rows": len(X), "unique_rows": X["x"].nunique(),
            "iterations": self.params_trained["num_boost_round"], "has_validation": kwargs.get("X_val") is not None})
        return result

    monkeypatch.setattr(LGBModel, "_fit", watch)
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x": np.arange(120, dtype=float)}, index=[f"train:{i}" for i in range(120)])
    y = pd.DataFrame({"answer": np.r_[2 * np.arange(100) + rng.normal(0, .01, 100), rng.normal(0, 2000, 20)]}, index=X.index)
    task = TaskSpec(kind="regression", targets=["answer"], id_columns=["id"], feature_columns=["x"],
        metric=MetricSpec(name="rmse"), output=OutputSpec(columns=["id", "answer"], prediction_type="values"), source_hash="fixture")
    # No n_estimators alias: observe the iterations selected by the real engine.
    spec = PipelineSpec(engine="autogluon", model="library_search", learners=["GBM"], n_trials=None,
        autogluon_configurations={"GBM": [{"min_data_in_leaf": 1, "learning_rate": .05}]})
    pipeline = SearchPipeline(spec, task)
    context = {"strategy": "random", "cpu": 1, "budget_seconds": 60, "memory_mb": 2048,
        "budget_mode": "full_comparison", "workdir": str(tmp_path), "task": task.model_dump(mode="json"),
        "protocol_hash": "fixture", "data_hash": "fixture", "split_id": "fixture"}
    pipeline.fit(X.iloc[:100], y.iloc[:100], context)
    measured = history[-1]
    initial = pipeline.models["answer"][1]
    initial_names = initial.model_names()
    initial_predictions = initial.predict(X.iloc[-5:]).to_numpy()

    pipeline.refit(X, y, {**context, "execution_mode": "refit"})
    assert history[-1]["rows"] == history[-1]["unique_rows"] == 120
    assert history[-1]["iterations"] == measured["iterations"]
    assert not history[-1]["has_validation"]
    assert len(history) == 3  # Research, its full-data fit, one frozen refit.
    assert initial.model_names() == initial_names
    assert np.array_equal(initial.predict(X.iloc[-5:]).to_numpy(), initial_predictions)
    restored = cloudpickle.loads(cloudpickle.dumps(pipeline))
    assert np.array_equal(restored.predict(X.iloc[-5:], context).values["answer"],
        pipeline.predict(X.iloc[-5:], context).values["answer"])
