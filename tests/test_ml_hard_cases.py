"""Offline regression checks for the research-to-runtime integration."""
from pathlib import Path
from types import SimpleNamespace
import json

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from automl_agent.compatibility import incompatibilities
from automl_agent.contracts import FeedbackPolicy, MetricSpec, OutputSpec, PipelineSpec, TaskSpec
from automl_agent.data import Dataset
from automl_agent.evaluation import feedback_reports
from automl_agent.pipelines import DirectPipeline
from automl_agent.validation import choose_validation, inner_split, pilot_indices
from context_compressor import ContextStore, ProfileCache, ProfileKey
from ml_knowledge.build import build
from ml_knowledge.context import SelectionContext, select
from ml_knowledge.core import canonical, digest, file_hash, write_json
from ml_knowledge.runtime import ReferenceStore

ROOT = Path(__file__).resolve().parents[1]


def classification(n=420, classes=3):
    rng = np.random.default_rng(810)
    X = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)}, index=[f"fit:{i}" for i in range(n)])
    boundaries = np.quantile(X.x, np.arange(1, classes) / classes)
    labels = [f"class_{i}" for i in range(classes)]
    y = pd.DataFrame({"answer": np.asarray(labels)[np.searchsorted(boundaries, X.x)]}, index=X.index)
    task = TaskSpec(kind="classification", targets=["answer"], feature_columns=list(X),
        metric=MetricSpec(name="f1", average="macro"), classes={"answer": labels}, source_hash="fixture",
        output=OutputSpec(columns=["answer"], prediction_type="labels"))
    return X, y, task


def regression_dataset():
    X = pd.DataFrame({"x": [0, 0, 1, 1, 2, 2, 3, 3], "z": [1] * 8})
    y = pd.DataFrame({"answer": [2., 2., 5., 6., 9., 11., 13., 13.]})
    task = TaskSpec(kind="regression", targets=["answer"], feature_columns=list(X),
        metric=MetricSpec(name="rmse"), source_hash="fixture",
        output=OutputSpec(columns=["answer"], prediction_type="values"))
    dataset = SimpleNamespace(train=X.join(y), test=X.iloc[:2].copy(), task=task, source_hash="fixture")
    return dataset


def context(task):
    return {"cpu": 1, "memory_mb": 2048, "strategy": "random", "seed": 42,
        "data_version": "fixture", "protocol_hash": "fixture", "split_id": "inner", "task": task.model_dump(mode="json")}


def routing_corpus(root):
    frozen = root / "frozen"
    frozen.mkdir(parents=True)
    source_text = "sklearn.metrics.mean_squared_error(y_true, y_pred)\n\nMean squared error regression loss."
    source = {"id": "api_sklearn_metrics_mean_squared_error", "type": "documentation",
        "title": "sklearn.metrics.mean_squared_error", "library": "scikit-learn", "version": "1.5.2",
        "symbols": ["sklearn.metrics.mean_squared_error"], "text": source_text,
        "doc_sha256": digest(source_text), "source_url": "https://example.test/sklearn/metrics",
        "source_kind": "test_fixture", "license": "test", "conditions": [], "warnings": [],
        "content_status": "complete", "applicability": "unknown"}
    constraints = "Different targets may reflect noise or missing context; do not alter rows or targets automatically."
    method_text = "Одинаковые признаки имеют разные целевые значения\n\nDiagnostic: inspect fit-only duplicate groups.\n\nAcceptance: compare the unchanged metric on valid internal validation.\n\nContraindications: " + constraints
    method = {"id": "method:hard_case_ml26", "legacy_id": "hard_case_ml26", "type": "method",
        "title": "Одинаковые признаки имеют разные целевые значения", "library": "project", "version": "1",
        "symbols": [], "text": method_text, "doc_sha256": digest(method_text), "source_url": "project:methods/hard_case_ml26",
        "source_kind": "project_recommendation", "license": "project", "conditions": ["Confirmed task; fit-only learning."],
        "warnings": [constraints], "content_status": "complete", "applicability": "unknown",
        "payload": {"routing_signal": "label_conflicts", "routing_aliases": ["duplicate_target_variation"],
            "routing_priority": 0, "references": [source["id"]], "applicability_rules": []}}
    documents = frozen / "documents.jsonl"
    documents.write_text("".join(canonical(row) + "\n" for row in sorted((source, method), key=lambda row: row["id"])))
    write_json(frozen / "expected.json", {"items": []})
    write_json(frozen / "coverage.json", {"complete": True, "test_fixture": True})
    write_json(frozen / "inputs.lock.json", {"documents_sha256": file_hash(documents),
        "expected_sha256": file_hash(frozen / "expected.json"), "coverage_sha256": file_hash(frozen / "coverage.json"),
        "requirements_sha256": digest("fixture")})
    build(frozen, root / "corpus")
    return root / "corpus"


def test_joint_forecast_lag_adapter_does_not_require_an_unused_inner_split():
    from automl_agent.pipelines import build_pipeline
    X = pd.DataFrame({"series": ["one"] * 8, "time": list(range(8))})
    y = pd.DataFrame({"answer": np.arange(8, dtype=float)})
    task = TaskSpec(kind="forecasting", targets=["answer"], feature_columns=list(X),
        categorical_columns=["series"], time_column="time", group_column="series",
        series_columns=["series"], horizon=1, metric=MetricSpec(name="rmse"), source_hash="fixture",
        output=OutputSpec(columns=["answer"], prediction_type="values"))
    spec = PipelineSpec(model="extra_trees", representation="lags", hyperparameters={"n_estimators": 5})
    c = {**context(task), "strategy": "temporal_group"}
    pipeline = build_pipeline(spec, task, c)
    pipeline.fit(X, y, c)
    prediction = pipeline.predict(pd.DataFrame({"series": ["one"], "time": [8]}), c)
    assert np.isfinite(prediction.values["answer"]).all()


@pytest.mark.parametrize("spec", [PipelineSpec(engine="optuna"), PipelineSpec(fit_diagnostics="inner_holdout"),
    PipelineSpec(prediction_policy="class_offsets", postprocess_fit_policy="inner_holdout")])
def test_joint_inner_paths_still_require_both_structural_roles(spec):
    _, _, task = classification()
    assert "joint_split_requires_time_and_group_roles" in incompatibilities(spec, task, "temporal_group")


@pytest.mark.parametrize("component", ["transformer", "pipeline"])
def test_custom_paths_cannot_relearn_class_offsets_or_leak_inner_labels(component):
    _, _, task = classification()
    spec = PipelineSpec(engine="custom", custom_component=component, prediction_policy="class_offsets",
        postprocess_fit_policy="inner_holdout")
    assert "class_offsets_require_direct_or_optuna_frozen_fit_path" in incompatibilities(spec, task, "random")


@pytest.mark.parametrize("component", ["transformer", "pipeline"])
@pytest.mark.parametrize("policy", ["threshold", "calibrated"])
def test_custom_binary_postprocessing_rejects_pretransformed_inner_holdout(component, policy):
    _, _, task = classification(classes=2)
    spec = PipelineSpec(engine="custom", custom_component=component, prediction_policy=policy,
        postprocess_fit_policy="inner_holdout")
    assert "custom_postprocessing_has_no_verified_inner_holdout; use direct or Optuna" in incompatibilities(spec, task, "random")


def test_catalog_contains_generic_scenarios_with_registered_api_provenance():
    paths = sorted((ROOT / "ml_knowledge/catalogs/methods").glob("hard_case_ml*.json"))
    assert len(paths) == 60
    with ReferenceStore(ROOT / "ml_knowledge/corpus", verify=False) as store:
        for path in paths:
            obj = json.loads(path.read_text())
            doc = store.document("method:" + obj["id"])
            assert doc and doc["payload"]["routing_signal"]
            assert all(store.document(identifier) for identifier in obj["references"])
            assert "Diagnostic:" in doc["text"] and "Acceptance:" in doc["text"]
            assert "project hypothesis" in doc["text"]
            for public_literal in ("Worms", "site_eui", "Year_Factor", "UNSW", "1458"):
                assert public_literal not in canonical(doc)


@pytest.mark.parametrize("task,metric,validation,risks,expected", [
    ("classification", "f1_macro", "random", {"class_imbalance": True, "rare_classes": True}, "25"),
    ("regression", "rmse", "random", {"heavy_tail": True}, "33"),
    ("regression", "rmse", "temporal_group", {}, "04"),
    ("classification", "f1_macro", "random", {"label_conflicts": True, "duplicate_features": True}, "26"),
    ("regression", "rmse", "temporal", {"missingness_shift": True}, "19"),
])
def test_routing_chooses_complete_risk_card(task, metric, validation, risks, expected):
    selection = SelectionContext(task, {"rows": 1000, "columns": 10, **risks}, validation, "direct extra_trees", {}, metric=metric)
    with ReferenceStore(ROOT / "ml_knowledge/corpus", verify=False) as store:
        packet = select(store, selection)
    assert packet["materials"][0]["id"] == "method:hard_case_ml" + expected
    material = packet["materials"][0]
    assert "Diagnostic:" in material["content"] and "Acceptance:" in material["content"]
    assert material["warnings"] and "Contraindications:" in material["content"]
    assert material["answer_usable"]
    assert len(canonical(packet)) <= 6000 and len(packet["materials"]) <= 3


def test_current_question_can_change_routed_card_and_cache_binding():
    selection = SelectionContext("classification", {"rows": 1000, "columns": 2, "rare_classes": True},
        "random", "direct extra_trees", {}, metric="f1_macro", research_question="Таймаут OOM")
    with ReferenceStore(ROOT / "ml_knowledge/corpus", verify=False) as store:
        packet = select(store, selection)
    assert packet["materials"][0]["id"] == "method:hard_case_ml57"
    assert selection.binding()["research_question"] == "Таймаут OOM"


def test_fit_risk_flags_do_not_read_control_targets():
    X, y, task = classification(n=100)
    train = X.join(y)
    train.loc[train.index[:3], ["x", "z"]] = [1., 2.]
    train.loc[train.index[:3], "answer"] = ["class_0", "class_1", "class_0"]
    train.loc[train.index[3:70], "answer"] = "class_0"
    dataset = SimpleNamespace(train=train, test=X.iloc[:5].copy(), task=task, source_hash="fixture")
    fit = list(range(70))
    before = Dataset.profile(dataset, fit).observations
    flags = before[0]["flags"]
    assert flags["duplicate_features"] and flags["label_conflicts"] and flags["class_imbalance"]
    dataset.train.loc[train.index[70:], "answer"] = "secret-control-value"
    assert Dataset.profile(dataset, fit).observations == before
    assert "secret-control-value" not in canonical(before)


def test_regression_duplicate_target_variation_is_fit_only_and_needs_two_groups():
    dataset = regression_dataset()
    identical = SimpleNamespace(train=dataset.train.copy(), test=dataset.test.copy(), task=dataset.task, source_hash=dataset.source_hash)
    identical.train["answer"] = 2.
    assert not Dataset.profile(identical, list(range(8))).observations[0]["flags"]["duplicate_target_variation"]
    two_groups = Dataset.profile(dataset, list(range(8))).observations
    assert two_groups[0]["flags"]["duplicate_target_variation"]
    assert not two_groups[0]["flags"]["label_conflicts"]
    selection = SelectionContext("regression", two_groups[0]["flags"], "random", "direct extra_trees", {}, metric="rmse")
    assert selection.fit_profile["duplicate_target_variation"]
    one_group = Dataset.profile(dataset, list(range(4))).observations
    assert not one_group[0]["flags"]["duplicate_target_variation"]
    dataset.train.loc[dataset.train.index[4:], "answer"] = [100., 101., 102., 102.]
    assert Dataset.profile(dataset, list(range(4))).observations == one_group


def test_cached_fit_profiles_route_regression_and_classification_conflicts(tmp_path):
    corpus = routing_corpus(tmp_path / "knowledge")
    profile_root = tmp_path / "profiles"
    profile_root.mkdir()
    regression = regression_dataset()
    X, y, task = classification(n=100)
    train = X.join(y)
    train.loc[train.index[:3], ["x", "z"]] = [1., 2.]
    train.loc[train.index[:3], "answer"] = ["class_0", "class_1", "class_0"]
    classification_data = SimpleNamespace(train=train, test=X.iloc[:5].copy(), task=task, source_hash="class-fixture")

    with ContextStore(profile_root / "profiles.sqlite") as cache_store, ReferenceStore(corpus) as references:
        cases = [
            (regression, list(range(len(regression.train))), "regression", "rmse", "duplicate_target_variation"),
            (classification_data, list(range(70)), "classification", "f1_macro", "label_conflicts"),
        ]
        for index, (dataset, fit, task_kind, metric, signal) in enumerate(cases):
            profile_document = Dataset.profile(dataset, fit).model_dump(mode="json")
            cached = ProfileCache(cache_store).get_or_create(
                ProfileKey("dataset-fixture-" + str(index), "fit-fixture-" + str(index)), lambda: profile_document)
            summary = cached["summary"]
            assert summary["risk_flags"][signal] is True
            assert "answer" not in canonical(summary)
            selection = SelectionContext(task_kind, {"rows": summary["rows"]["fit"],
                "columns": summary["feature_count"], **summary["risk_flags"]}, "random", "direct extra_trees", {}, metric=metric)
            packet = select(references, selection)
            assert packet["materials"][0]["id"] == "method:hard_case_ml26"


def joint_fixture(shared=False):
    n = 100
    train = pd.DataFrame({"x": np.arange(n), "when": np.repeat(np.arange(1, 6), 20),
        "entity": ["shared" if shared or i % 20 == 0 else f"period_{i // 20}_entity_{i % 4}" for i in range(n)],
        "answer": np.arange(n) * .2})
    task = TaskSpec(kind="regression", targets=["answer"], feature_columns=["x", "when"],
        time_column="when", group_column="entity", metric=MetricSpec(name="rmse"), source_hash="joint",
        output=OutputSpec(columns=["answer"], prediction_type="values"))
    return SimpleNamespace(task=task, train=train, test=pd.DataFrame({"x": [101], "when": [6], "entity": ["future"]}), source_hash="joint")


def test_joint_time_and_group_boundaries_and_inner_split():
    dataset = joint_fixture()
    plan = choose_validation(dataset)
    assert plan.strategy == "temporal_group"
    for a, b in ((plan.fit, plan.validation), (plan.fit, plan.control), (plan.validation, plan.control)):
        assert not set(dataset.train.iloc[a].entity) & set(dataset.train.iloc[b].entity)
        assert dataset.train.iloc[a].when.max() < dataset.train.iloc[b].when.min()
    X = dataset.train.iloc[plan.fit].drop(columns="answer")
    y = dataset.train.iloc[plan.fit][["answer"]]
    a, b = inner_split(X, y, dataset.task, plan.strategy, 42)
    assert not set(X.iloc[a].entity) & set(X.iloc[b].entity)
    assert X.iloc[a].when.max() < X.iloc[b].when.min()
    assert len(pilot_indices(X, y, dataset.task, plan.strategy, 5, 42)) == len(X)
    assert any("joint_temporal_group" in s for s in incompatibilities(PipelineSpec(engine="flaml"), dataset.task, plan.strategy))


def test_impossible_joint_split_never_falls_back_to_random():
    with pytest.raises(ValueError, match="joint temporal/group"):
        choose_validation(joint_fixture(shared=True))


def test_inner_diagnostics_fit_preprocessing_on_inner_train_only(monkeypatch):
    from automl_agent.preprocessing import FeatureMatrix
    seen = []
    original = FeatureMatrix.fit_transform
    def record(preprocessor, X, y):
        seen.append(set(X.index))
        return original(preprocessor, X, y)
    monkeypatch.setattr(FeatureMatrix, "fit_transform", record)
    X, y, task = classification()
    spec = PipelineSpec(model="extra_trees", hyperparameters={"n_estimators": 5}, fit_diagnostics="inner_holdout")
    pipeline = DirectPipeline(spec, task)
    summary = pipeline.fit(X, y, context(task))
    report = summary.fit_diagnostics
    assert report["scope"] == "fit_inner_holdout" and report["semantics"] == "full_confusion_per_class"
    a, b = inner_split(X, y, task, "random", spec.seed)
    assert seen[0] == set(X.iloc[a].index) and not seen[0] & set(X.iloc[b].index)
    assert seen[1] == set(X.index) and len(seen) == 2
    assert summary.auxiliary_fits >= 1 and report["not_external_validation"]
    assert sum(row["support"] for row in report["classes"]) == len(b)
    assert all("precision" in row and "recall" in row and "f1" in row for row in report["classes"])
    assert pipeline.frozen_spec().fit_diagnostics == "none"


def test_offsets_saved_in_class_order_and_reused_without_tuning(monkeypatch):
    import automl_agent.pipelines as module
    X, y, task = classification()
    spec = PipelineSpec(model="extra_trees", hyperparameters={"n_estimators": 5},
        prediction_policy="class_offsets", postprocess_fit_policy="inner_holdout", fit_diagnostics="inner_holdout")
    pipeline = DirectPipeline(spec, task)
    pipeline.fit(X, y, context(task))
    frozen = pipeline.frozen_spec()
    assert len(frozen.class_offsets) == len(task.classes["answer"])
    before = pipeline.predict(X, context(task))
    restored = cloudpickle.loads(cloudpickle.dumps(pipeline))
    assert np.array_equal(before.values["answer"], restored.predict(X, context(task)).values["answer"])
    monkeypatch.setattr(module, "inner_split", lambda *a, **kw: pytest.fail("Frozen rule was retuned"))
    final = DirectPipeline(frozen, task)
    summary = final.fit(X, y, context(task))
    assert summary.auxiliary_fits == 0
    assert np.array_equal(final.postprocessors["answer"]["offsets"], frozen.class_offsets)


def test_multiclass_decision_is_rejected_for_probability_output():
    X, y, task = classification()
    task = task.model_copy(update={"output": OutputSpec(columns=["p0", "p1", "p2"], prediction_type="probabilities")})
    spec = PipelineSpec(prediction_policy="class_offsets", postprocess_fit_policy="inner_holdout")
    assert any("class_offsets_require" in r for r in incompatibilities(spec, task, "random"))


def test_optuna_freezes_selected_decision_rule(tmp_path):
    pytest.importorskip("optuna")
    from automl_agent.search import SearchPipeline
    X, y, task = classification()
    spec = PipelineSpec(engine="optuna", model="extra_trees", n_trials=1,
        hyperparameters={"n_estimators": 5}, search_space={"max_depth": {"type": "int", "low": 2, "high": 3}},
        prediction_policy="class_offsets", postprocess_fit_policy="inner_holdout")
    pipeline = SearchPipeline(spec, task)
    pipeline.fit(X, y, {**context(task), "budget_seconds": 60, "workdir": str(tmp_path)})
    frozen = pipeline.frozen_spec()
    assert frozen.engine == "direct" and frozen.class_offsets is not None
    assert frozen.class_offsets == pipeline.pipeline.frozen_spec().class_offsets


def test_v039_full_confusion_diagnostics_and_privacy_are_preserved():
    _, _, task = classification(n=60, classes=2)
    task.classes["answer"] = ["class_0", "class_1"]
    y = pd.DataFrame({"answer": ["class_0"] * 30 + ["class_1"] * 30})
    X = pd.DataFrame({"x": np.arange(60), "z": np.zeros(60)})
    plan = SimpleNamespace(hash="fixture", observations=[], feedback_policy=FeedbackPolicy(report_ids=["classes"]))
    report = feedback_reports(task, plan, X, y, {"answer": np.array(["class_0"] * 45 + ["class_1"] * 15)}, "labels")["classes"]
    assert report["classes"][0]["f1"] == .8
    y.loc[:10, "answer"] = "class_1"
    assert feedback_reports(task, plan, X, y, {"answer": y.answer.to_numpy()}, "labels")["classes"]["status"] == "suppressed"
