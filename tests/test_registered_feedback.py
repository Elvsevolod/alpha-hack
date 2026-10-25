import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import MetricSpec, OutputSpec, TaskSpec
from automl_agent.evaluation import feedback_reports, score_predictions
from automl_agent.feedback import freeze_partitions, partition_masks, registered_partitions
from automl_agent.validation import choose_validation


def numeric_task(**kwargs):
    return TaskSpec(kind="regression", targets=["target"], feature_columns=["value", "other"],
        metric=MetricSpec(name="rmse"), output=OutputSpec(columns=["target"], prediction_type="values"),
        source_hash="fixture", **kwargs)


def dataset(task, n=1600):
    rng = np.random.default_rng(9)
    train = pd.DataFrame({"value": rng.normal(size=n), "other": rng.normal(size=n), "target": rng.normal(size=n)})
    return SimpleNamespace(task=task, train=train, test=train.iloc[:20].drop(columns="target"), source_hash="fixture")


def test_numeric_boundaries_are_fit_only_and_policy_covers_the_definition():
    task = numeric_task()
    fit = pd.DataFrame({"value": np.arange(500.), "other": np.arange(500.)})
    validation = pd.DataFrame({"value": np.arange(240.), "other": np.arange(240.)})
    rules = freeze_partitions(task, fit, validation)
    shifted_validation = validation + 10000
    assert rules == freeze_partitions(task, fit, shifted_validation)
    changed = freeze_partitions(task, fit + 10, validation)
    assert rules[0]["report_id"] != changed[0]["report_id"]
    assert rules[0]["definition"]["cuts"] == np.quantile(fit["other"], [.25, .5, .75]).tolist()
    assert all(rule["definition"]["cut_scope"] == "fit" for rule in rules)


def test_partition_policy_is_independent_of_validation_targets():
    data = dataset(numeric_task())
    plan = choose_validation(data)
    policy_hash = plan.feedback_policy.hash
    data.train.loc[plan.validation, "target"] += 100000
    new = choose_validation(data)
    assert new.feedback_policy.hash == policy_hash
    assert new.observations == plan.observations


def test_large_ranges_report_without_mutation_and_small_complements_are_hidden():
    data = dataset(numeric_task())
    plan = choose_validation(data)
    plan.feedback_policy = plan.feedback_policy.model_copy(update={"version":2,"min_slice_rows":30,"joint_disclosure":"conservative_v2"})
    rules = registered_partitions(plan)
    identifier, rule = next(iter(rules.items()))
    X = data.train.iloc[plan.validation].drop(columns="target").reset_index(drop=True)
    y = data.train.iloc[plan.validation][["target"]].reset_index(drop=True)
    before = plan.hash
    predictions = {"target": np.zeros(len(y))}
    reports = feedback_reports(data.task, plan, X, y, predictions, "values")
    assert reports[identifier]["status"] == "available"
    assert sum(s["rows"] for s in reports[identifier]["slices"].values()) == len(X)
    assert plan.hash == before
    assert reports == feedback_reports(data.task, plan, X, y, predictions, "values")
    column = rule["column"]
    X[column] = rule["cuts"][-1] + 1
    X.loc[:9, column] = rule["cuts"][0] - 1
    hidden = feedback_reports(data.task, plan, X, y, predictions, "values")[identifier]
    assert hidden["status"] == "suppressed" and "slices" not in hidden and "rows" not in hidden
    assert np.isfinite(score_predictions(data.task, y, predictions, "values"))


@pytest.mark.parametrize("dates", [False, True])
def test_temporal_blocks_are_predeclared_and_cover_complete_periods(dates):
    task = numeric_task(time_column="when", explicit_validation="temporal")
    task = task.model_copy(update={"kind": "forecasting", "horizon": 3, "feature_columns": ["value", "other", "when"]})
    data = dataset(task, 1200)
    periods = pd.date_range("2025-01-01", periods=30, freq="D") if dates else np.arange(30)
    data.train["when"] = np.repeat(periods, 40)
    data.test["when"] = periods[-1] + (pd.Timedelta(days=1) if dates else 1)
    plan = choose_validation(data)
    identifier, rule = next((k, r) for k, r in registered_partitions(plan).items() if r["kind"] == "periods")
    X = data.train.iloc[plan.validation].drop(columns="target").reset_index(drop=True)
    masks = partition_masks(rule, X)
    assert np.all(np.column_stack(list(masks.values())).sum(axis=1) == 1)
    for period in X.when.unique():
        selected = X.when == period
        assert sum(bool(mask[selected].all()) for mask in masks.values()) == 1
    y = data.train.iloc[plan.validation][["target"]].reset_index(drop=True)
    report = feedback_reports(task, plan, X, y, {"target": np.zeros(len(y))}, "values")[identifier]
    assert report["status"] == "available"


def test_group_blocks_never_split_a_ranking_query_or_expose_its_identifiers():
    task = numeric_task(group_column="query")
    task = task.model_copy(update={"kind": "ranking", "metric": MetricSpec(name="ndcg", k=5)})
    data = dataset(task, 1400)
    data.train["query"] = np.repeat([f"private-query-{i}" for i in range(70)], 20)
    data.train["target"] = np.tile(np.arange(20), 70)
    plan = choose_validation(data)
    rules = registered_partitions(plan)
    assert rules and all(r["kind"] == "group_blocks" for r in rules.values())
    identifier, rule = next(iter(rules.items()))
    X = data.train.iloc[plan.validation].drop(columns="target").reset_index(drop=True)
    y = data.train.iloc[plan.validation][["target"]].reset_index(drop=True)
    masks = partition_masks(rule, X)
    for query in X["query"].unique():
        selected = X["query"] == query
        assert sum(bool(mask[selected].all()) for mask in masks.values()) == 1
    assert all(X.loc[mask, "query"].nunique() >= 2 for mask in masks.values())
    report = feedback_reports(task, plan, X, y, {"target": y.target.to_numpy()}, "scores")[identifier]
    assert report["status"] == "available"
    assert "private-query-" not in json.dumps(report)
    assert "group_hash_to_block" not in json.dumps(report)


def test_ranking_diagnostic_rejects_a_mask_that_cuts_queries():
    task = numeric_task(group_column="query")
    task = task.model_copy(update={"kind": "ranking", "metric": MetricSpec(name="ndcg", k=5)})
    data = dataset(task, 1400)
    data.train["query"] = np.repeat(np.arange(70), 20)
    data.train["target"] = np.tile(np.arange(20), 70)
    plan = choose_validation(data)
    X = data.train.iloc[plan.validation].drop(columns="target").reset_index(drop=True)
    X.loc[X.index[::2], "value"] = np.nan
    y = data.train.iloc[plan.validation][["target"]].reset_index(drop=True)
    report = feedback_reports(task, plan, X, y, {"target": y.target.to_numpy()}, "scores")["missingness"]
    assert report["status"] == "unsupported" and "slices" not in report


def test_a_single_query_is_not_an_allowed_extra_report_even_with_many_rows():
    task = numeric_task(group_column="query")
    task = task.model_copy(update={"kind": "ranking", "metric": MetricSpec(name="ndcg", k=5)})
    data = dataset(task, 1600)
    data.train["query"] = np.repeat(np.arange(8), 200)
    data.train["target"] = np.tile(np.arange(200), 8)
    plan = choose_validation(data)
    assert not registered_partitions(plan)
    X = data.train.iloc[plan.validation].drop(columns="target").reset_index(drop=True)
    X.loc[X["query"] == X["query"].iloc[0], "value"] = np.nan
    y = data.train.iloc[plan.validation][["target"]].reset_index(drop=True)
    predictions = {"target": y.target.to_numpy()}
    report = feedback_reports(task, plan, X, y, predictions, "scores")["missingness"]
    assert report["status"] == "unsupported" and "slices" not in report
    assert score_predictions(task, y, predictions, "scores", X["query"].to_numpy()) == pytest.approx(1)


def test_tampered_partition_is_rejected_instead_of_silently_changing_the_report():
    plan = choose_validation(dataset(numeric_task()))
    corrupted = copy.deepcopy(plan)
    corrupted.observations[0]["definition"]["cuts"][0] += 1
    with pytest.raises(ValueError, match="registered policy"):
        registered_partitions(corrupted)


def test_small_fixture_uses_full_v3_research_policy():
    plan = choose_validation(dataset(numeric_task(), 180))
    assert plan.feedback_policy.report_ids == ["missingness", "seen_unseen", "regression_errors"]
    assert plan.feedback_policy.version == 3 and plan.feedback_policy.min_slice_rows == 1
    assert not registered_partitions(plan)


def test_nullable_numeric_columns_and_new_missingness_are_covered():
    task = numeric_task()
    fit = pd.DataFrame({"value": pd.Series(np.arange(500.), dtype="Float64"), "other": np.arange(500.)})
    validation = fit.iloc[:240].copy()
    validation.loc[validation.index[:35], "value"] = pd.NA
    rules = freeze_partitions(task, fit, validation)
    rule = next(r["definition"] for r in rules if r["definition"]["column"] == "value")
    masks = partition_masks(rule, validation)
    assert masks["missing"].sum() == 35
    assert np.all(np.column_stack(list(masks.values())).sum(axis=1) == 1)


def test_registered_ranges_execute_in_the_complete_controller(tmp_path):
    from automl_agent.budget import Limits
    from automl_agent.controller import Controller
    data = dataset(numeric_task(), 900)
    data.train["target"] = 2 * data.train["value"] + data.train["other"]
    data.train.to_csv(tmp_path / "train.csv", index=False)
    data.test.to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"target": np.zeros(len(data.test))}).to_csv(tmp_path / "sample_submission.csv", index=False)
    (tmp_path / "task_spec.json").write_text(json.dumps(data.task.model_dump(mode="json", exclude={"source_hash"})))
    task_path = tmp_path / "task.md"
    task_path.write_text("Technical regression fixture; metric RMSE, output target. No external data.")
    controller = Controller(task_path, Limits(seconds=90, cpu=1), run_id="ranges")
    try:
        controller.prepare()
        partitions = registered_partitions(controller.plan)
        _, baseline = controller.lookup(controller.journal.get("baseline_id"))
        assert partitions and any(baseline.feedback[name]["status"] == "available" for name in partitions)
        public = controller.public_result(baseline)
        assert "group_hash_to_block" not in json.dumps(public)
        report = controller.run(no_llm=True)
        assert report["state"] == "completed" and report["control_status"] == "completed"
        assert len(pd.read_csv(tmp_path / "submission.csv")) == len(data.test)
    finally:
        controller.close()
