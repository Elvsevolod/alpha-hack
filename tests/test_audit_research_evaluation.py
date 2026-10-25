from types import SimpleNamespace

import numpy as np
import pandas as pd

from automl_agent.contracts import FeedbackPolicy, MetricSpec, OutputSpec, TaskSpec
from automl_agent.evaluation import feedback_reports
from automl_agent.research_evaluation import domain_diagnostic, feature_groups, folds, oof_predictions, transfer_profile


def test_one_rare_class_never_removes_other_class_or_slice_metrics():
    task = TaskSpec(kind="classification", targets=["outcome"], feature_columns=["measurement"],
        classes={"outcome": ["common", "rare"]}, metric=MetricSpec(name="f1"),
        output=OutputSpec(columns=["outcome"], prediction_type="labels"), source_hash="fixture")
    X = pd.DataFrame({"measurement": [np.nan] + [1.] * 49})
    y = pd.DataFrame({"outcome": ["common"] * 48 + ["rare"] * 2})
    plan = SimpleNamespace(hash="fixed", observations=[], feedback_policy=FeedbackPolicy(version=3,
        joint_disclosure="research_v3", report_ids=["classes", "missingness"], min_slice_rows=1))
    report = feedback_reports(task, plan, X, y, {"outcome": np.array(["common"] * 50)}, "labels")
    detail = report["classes"]
    assert detail["status"] == report["missingness"]["status"] == "available"
    assert detail["targets"]["outcome"]["confusion_matrix"] == [[48, 0], [2, 0]]
    rare = detail["classes"][1]
    assert rare["support"] == 2 and rare["predicted_support"] == 0
    assert rare["recall"] == rare["precision"] == rare["f1"] == 0
    assert rare["uncertain"]
    assert report["missingness"]["slices"]["missing"]["rows"] == 1


def test_temporal_folds_are_forward_only_and_document_initial_uncovered_rows(tabular):
    X, y, task = tabular
    X = X.assign(period=np.repeat([1, 2, 3], len(X) // 3))
    task = task.model_copy(update={"time_column": "period", "feature_columns": list(X)})
    parts = folds(X, y, task, "temporal")
    for a, b in parts:
        assert X.iloc[a].period.max() < X.iloc[b].period.min()
        assert not set(X.iloc[a].index) & set(X.iloc[b].index)


def test_group_folds_keep_entities_disjoint(tabular):
    X, y, task = tabular
    X = X.assign(entity=np.arange(len(X)) // 5)
    task = task.model_copy(update={"group_column": "entity", "feature_columns": list(X)})
    for a, b in folds(X, y, task, "group"):
        assert not set(X.iloc[a].entity) & set(X.iloc[b].entity)


def test_transfer_conflicts_use_fit_labels_and_features_are_not_entity_assertions(tabular):
    X, y, task = tabular
    fit = pd.concat([X.iloc[:10], X.iloc[:2]], ignore_index=True)
    labels = pd.DataFrame({"answer": np.arange(len(fit), dtype=float)})
    report = transfer_profile(fit, labels, X.iloc[:4], X, X.iloc[-20:], task)
    assert report["overlap"]["validation_vs_fit"]["fraction"] == 1
    assert report["fit_conflicts"]["answer"]["groups"] == 2
    assert report["exact_feature_groups_are_not_entities"]
    assert not report["target_quality_inferred_from_shift"]


def test_domain_classifier_has_no_task_target_input_and_respects_exact_groups():
    rng = np.random.default_rng(3)
    train = pd.DataFrame({"observed": rng.normal(size=100), "category": ["alpha"] * 100})
    test = pd.DataFrame({"observed": rng.normal(5, size=100), "category": ["beta"] * 100})
    test.iloc[:8] = train.iloc[:8].to_numpy()
    report = domain_diagnostic(train, test)
    assert report["auc"] > .8
    assert report["exact_groups_disjoint"] and not report["task_labels_used"]
    assert {row["column"] for row in report["feature_importance"]} == set(train)


def test_oof_predictions_preserve_fold_and_row_identity(tabular):
    from automl_agent.contracts import PipelineSpec
    X, y, task = tabular
    context = {"cpu": 1, "memory_mb": 2048, "budget_seconds": 30,
        "strategy": "random", "protocol_hash": "fixed", "task": task.model_dump(mode="json")}
    result = oof_predictions(PipelineSpec(model="ridge"), task, X, y, context)
    assert result["row_keys"] == X.index.tolist() and result["coverage"] == 1
    for fold in result["folds"]:
        assert not set(fold["fit_keys"]) & set(fold["evaluation_keys"])
    assert len(result["values"]["answer"]) == len(X)
