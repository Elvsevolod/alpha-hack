import copy
import json

import pandas as pd
import pytest

from tools.comparison_protocol import (dispatch_data, dispatch_result, main, paired_win, receipt_issues,
    run_one, run_workspace, score_csv, validate_selection)
from automl_agent.contracts import SelectionPolicy, TaskSpec
from automl_agent.util import digest


def test_dispatch_result_reads_wrapped_payload_and_legacy_result():
    wrapped = {"status": "completed", "data": {"experiment": {"experiment_id": "exp-1"},
        "result": {"status": "completed", "score": .7}}}
    assert dispatch_result(wrapped) == {"status": "completed", "score": .7}
    assert dispatch_data(wrapped)["experiment"]["experiment_id"] == "exp-1"
    assert dispatch_result({"result": {"status": "completed", "score": .6}})["score"] == .6
    assert dispatch_result({"status": "rejected"}) == {}


def test_each_comparison_run_gets_an_isolated_workspace(tmp_path):
    source = tmp_path / "task"
    source.mkdir()
    (source / "task.md").write_text("task", encoding="utf-8")
    (source / "train.csv").write_text("x\n1\n", encoding="utf-8")
    (source / "submission.csv").write_text("generated output", encoding="utf-8")
    (source / ".agent_work").mkdir()
    (source / ".agent_work" / "journal.sqlite").write_text("runtime state", encoding="utf-8")
    first = run_workspace(source / "task.md", tmp_path / "receipts" / "reference.json")
    second = run_workspace(source / "task.md", tmp_path / "receipts" / "fixed_automl.json")
    assert first.parent != second.parent
    assert first.read_text(encoding="utf-8") == "task"
    assert (second.parent / "train.csv").read_text(encoding="utf-8") == "x\n1\n"
    assert not (second.parent / "submission.csv").exists()
    assert not (second.parent / ".agent_work").exists()


def test_large_drawdown_rejects_two_positive_macro_f1_pairs():
    policy = SelectionPolicy(maximize=True, min_delta_abs=.001)
    result = paired_win([.702, .702, .55], [.7, .7, .7], policy)
    assert not result["accepted"] and result["meaningful_pairs"] == 2
    assert not result["drawdown_satisfied"][-1]


def test_small_drawdown_allowed_but_all_three_conditions_required():
    policy = SelectionPolicy(maximize=True, min_delta_abs=.001)
    assert paired_win([.703, .702, .6995], [.7] * 3, policy)["accepted"]
    assert not paired_win([.7011, .7011, .6989], [.7] * 3, policy)["accepted"]
    assert not paired_win([.701, .701, .701], [.7] * 3, policy)["accepted"]
    assert not paired_win([None, .8, .8], [.7] * 3, policy)["accepted"]


def test_rmse_direction_relative_tolerance_zero_and_nonpositive_mean():
    policy = SelectionPolicy(maximize=False, min_delta_abs=1e-12, min_delta_rel=.001)
    result = paired_win([9.98, 9.98, 10.005], [10.] * 3, policy)
    assert result["accepted"] and result["differences"][0] > 0
    assert result["epsilon"] == [.01] * 3
    assert not paired_win([.01, .01, .01], [0., 0., 0.], policy)["accepted"]
    # Positive mean alone is insufficient with fewer than two wins.
    assert not paired_win([8, 10, 10], [10.] * 3, policy)["accepted"]


def test_completed_csv_does_not_hide_trial_limit_or_unfinished_refit():
    rows = {"fit": [0, 1], "validation": [2], "control": [3]}
    entry = {"data_version": "data", "validation": rows}
    run = {"arm": "fixed_automl", "kind": "regression", "seed": 42}
    manifest = {"code_sha256": "code", "selected_automl": {"regression": "flaml"},
        "automl_full": {"normal_stops": ["calculated_time_budget_exhausted", "engine_space_exhausted"]}}
    receipt = {"arm": "fixed_automl", "seed": 42, "code_sha256": "code", "data_version": "data",
        "validation_indices_hash": digest(rows), "queue": [{"id": "flaml", "status": "completed", "experiment_id": "exp-1"}],
        "report": {"state": "completed", "control_status": "completed", "refit_status": "completed",
            "publication": {"scope": "full_train_refit"}},
        "research": [{"engine": "flaml", "model": "library_search", "scope": "full", "status": "completed", "n_trials": None,
            "experiment_id": "exp-1",
            "budget_mode": "full_comparison", "budget_seconds": 9000, "budget": {"n_trials": None,
                "targets": {"answer": {"max_iter_argument_present": False, "early_stop": False,
                    "stop_reason": "calculated_time_budget_exhausted"}}}}]}
    assert not receipt_issues(receipt, run, manifest, entry)
    limited = copy.deepcopy(receipt)
    limited["research"][0]["budget"]["targets"]["answer"]["stop_reason"] = "trial_limit_reached"
    assert "full_automl_stop_policy_invalid" in receipt_issues(limited, run, manifest, entry)
    limited["report"]["refit_status"] = "refit_skipped_budget"
    assert "refit_status_incomplete" in receipt_issues(limited, run, manifest, entry)


def test_queue_ids_and_experiment_ids_must_match_frozen_queue_and_research():
    rows = {"fit": [0, 1], "validation": [2], "control": [3]}
    run = {"arm": "fixed_automl", "kind": "regression", "seed": 42}
    manifest = {"code_sha256": "code", "selected_automl": {"regression": "flaml"},
        "automl_full": {"normal_stops": ["calculated_time_budget_exhausted"]}}
    receipt = {"arm": "fixed_automl", "seed": 42, "code_sha256": "code", "data_version": "data",
        "validation_indices_hash": digest(rows), "queue": [{"id": "wrong-engine", "status": "completed", "experiment_id": "missing"}],
        "report": {"state": "completed", "control_status": "completed", "refit_status": "completed",
            "publication": {"scope": "full_train_refit"}}, "research": []}
    entry = {"data_version": "data", "validation": rows}
    issues = receipt_issues(receipt, run, manifest, entry)
    assert "fixed_queue_incomplete" in issues
    assert "fixed_queue_research_mismatch" in issues

    misbound = copy.deepcopy(receipt)
    misbound["queue"] = [{"id": "flaml", "status": "completed", "experiment_id": "exp-direct"}]
    misbound["research"] = [
        {"experiment_id": "exp-direct", "engine": "direct", "model": "library_search", "scope": "full", "status": "completed"},
        {"experiment_id": "exp-flaml", "engine": "flaml", "model": "library_search", "scope": "full", "status": "completed",
            "n_trials": None, "budget_mode": "full_comparison", "budget_seconds": 9000,
            "budget": {"n_trials": None, "targets": {"answer": {"max_iter_argument_present": False,
                "early_stop": False, "stop_reason": "calculated_time_budget_exhausted"}}}}
    ]
    assert "fixed_queue_research_mismatch" in receipt_issues(misbound, run, manifest, entry)


def selection_fixture():
    task = TaskSpec(source_hash="source", kind="classification", targets=["answer"], id_columns=["id"],
        feature_columns=["x"], classes={"answer": ["0", "1"]}, metric={"name": "f1", "average": "macro"},
        output={"columns": ["id", "answer"], "prediction_type": "labels"})
    task_data = {"data_version": "data", "validation_hash": "split", "task": task.model_dump(mode="json")}
    def candidate(engine, score, status="completed"):
        experiment_id = "exp-" + engine
        return {"engine": engine, "data_version": "data", "validation_hash": "split", "control_not_opened": True,
            "outcome": {"data": {"experiment": {"experiment_id": experiment_id, "data_version": "data",
                "task_hash": "source", "validation_hash": "split", "status": "full",
                "pipeline": {"engine": engine}},
                "result": {"status": status, "scope": "full", "score": score, "experiment_id": experiment_id,
                    "validation_hash": "split"}}}}
    record = {"engine": "flaml", "selection": "research_validation_only", "outer_scores_used": False,
        "code_sha256": "old-code", "runs": [candidate("flaml", .8), candidate("autogluon", .7)]}
    return record, task_data, "source"


def test_freeze_rejects_selection_with_wrong_runtime_or_nonwinning_engine():
    record, task_data, expected_task_hash = selection_fixture()
    assert validate_selection(record, task_data, "old-code", expected_task_hash) == "flaml"
    with pytest.raises(ValueError, match="different runtime"):
        validate_selection(record, task_data, "current-code", expected_task_hash)
    losing = copy.deepcopy(record)
    losing["engine"] = "autogluon"
    with pytest.raises(ValueError, match="did not win"):
        validate_selection(losing, task_data, "old-code", expected_task_hash)
    misbound = copy.deepcopy(record)
    misbound["runs"][0]["outcome"]["data"]["experiment"]["pipeline"]["engine"] = "autogluon"
    with pytest.raises(ValueError, match="does not match its recorded experiment"):
        validate_selection(misbound, task_data, "old-code", expected_task_hash)
    wrong_task = copy.deepcopy(record)
    wrong_task["runs"][0]["outcome"]["data"]["experiment"]["task_hash"] = "another-task"
    with pytest.raises(ValueError, match="does not match its recorded experiment"):
        validate_selection(wrong_task, task_data, "old-code", expected_task_hash)
    wrong_split = copy.deepcopy(record)
    wrong_split["runs"][0]["outcome"]["data"]["result"]["validation_hash"] = "another-split"
    with pytest.raises(ValueError, match="does not match its recorded experiment"):
        validate_selection(wrong_split, task_data, "old-code", expected_task_hash)
    string_control = copy.deepcopy(record)
    string_control["runs"][0]["control_not_opened"] = "false"
    with pytest.raises(ValueError, match="different data/protocol"):
        validate_selection(string_control, task_data, "old-code", expected_task_hash)


def test_freeze_calls_selection_validation_before_archive_processing(tmp_path, monkeypatch):
    from tools import comparison_protocol
    record, task_data, expected_task_hash = selection_fixture()
    record["engine"] = "autogluon"
    output = tmp_path / "protocol"
    output.mkdir()
    (output / "DATA_MANIFEST.json").write_text(json.dumps({"tasks": {"classification": task_data}}), encoding="utf-8")
    selected_path = tmp_path / "SELECTED.json"
    selected_path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(comparison_protocol, "code_hash", lambda: "old-code")
    class FrozenDataset:
        source_hash = "data"
        def __init__(self, _path): pass
        def explicit(self): return self
        def activate(self, task): return task
        @property
        def hash(self): return expected_task_hash
    monkeypatch.setattr(comparison_protocol, "Dataset", FrozenDataset)
    with pytest.raises(ValueError, match="did not win"):
        comparison_protocol.freeze(output, {"classification": selected_path}, tmp_path / "not-opened.zip")


def test_fixed_automl_requires_engine_at_cli_and_function_boundary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["benchmark.py", "run-one", "--task", "task.md",
        "--output", str(tmp_path / "receipt.json"), "--arm", "fixed_automl", "--seed", "42"])
    with pytest.raises(SystemExit, match="2"):
        main()
    assert "requires --engine" in capsys.readouterr().err
    with pytest.raises(ValueError, match="requires --engine"):
        run_one(tmp_path / "task.md", "fixed_automl", 42, 600, tmp_path / "receipt.json")


def test_fixed_portfolio_queue_is_bound_to_each_research_model():
    portfolio = [{"id": "catboost", "pipeline": {"model": "catboost"}},
        {"id": "flaml", "pipeline": {"engine": "flaml", "model": "library_search"}}]
    run = {"arm": "fixed_portfolio", "kind": "regression", "seed": 42}
    manifest = {"code_sha256": "code", "selected_automl": {"regression": "flaml"}, "portfolio": portfolio,
        "automl_full": {"normal_stops": ["calculated_time_budget_exhausted"]}}
    rows = {"fit": [0, 1], "validation": [2], "control": [3]}
    entry = {"data_version": "data", "validation": rows}
    research = [
        {"experiment_id": "exp-catboost", "engine": "direct", "model": "catboost", "scope": "full", "status": "completed",
            "budget_mode": "full_comparison", "budget_seconds": 4000},
        {"experiment_id": "exp-flaml", "engine": "flaml", "model": "library_search", "scope": "full", "status": "completed",
            "n_trials": None, "budget_mode": "full_comparison", "budget_seconds": 9000,
            "budget": {"n_trials": None, "targets": {"answer": {"max_iter_argument_present": False,
                "early_stop": False, "stop_reason": "calculated_time_budget_exhausted"}}}}
    ]
    receipt = {"arm": "fixed_portfolio", "seed": 42, "code_sha256": "code", "data_version": "data",
        "validation_indices_hash": digest(rows), "report": {"state": "completed", "control_status": "completed",
            "refit_status": "completed", "publication": {"scope": "full_train_refit"}},
        "queue": [{"id": "catboost", "status": "completed", "experiment_id": "exp-catboost"},
            {"id": "flaml", "status": "completed", "experiment_id": "exp-flaml"}], "research": research}
    assert not receipt_issues(receipt, run, manifest, entry)
    wrong_model = copy.deepcopy(receipt)
    wrong_model["research"][0]["model"] = "lightgbm"
    assert "fixed_queue_research_mismatch" in receipt_issues(wrong_model, run, manifest, entry)
    wrong_engine = copy.deepcopy(receipt)
    wrong_engine["research"][0]["engine"] = "autogluon"
    assert "fixed_queue_research_mismatch" in receipt_issues(wrong_engine, run, manifest, entry)
    wrong_budget = copy.deepcopy(receipt)
    wrong_budget["research"][0]["budget_mode"] = "research"
    assert "fixed_queue_research_mismatch" in receipt_issues(wrong_budget, run, manifest, entry)
    bool_budget = copy.deepcopy(receipt)
    bool_budget["research"][0]["budget_seconds"] = True
    assert "fixed_queue_research_mismatch" in receipt_issues(bool_budget, run, manifest, entry)


def test_final_csv_numeric_string_labels_and_invalid_regression_are_accounted(tmp_path):
    task = TaskSpec(source_hash="source", kind="classification", targets=["answer"], id_columns=["id"],
        feature_columns=["x"], classes={"answer": ["0", "1"]}, metric={"name": "f1", "average": "macro"},
        output={"columns": ["id", "answer"], "prediction_type": "labels"})
    actual = pd.DataFrame({"id": ["a", "b"], "answer": ["0", "1"]})
    path = tmp_path / "final.csv"
    actual.to_csv(path, index=False)
    assert score_csv(path, actual, task) == 1
    regression = task.model_copy(update={"kind": "regression", "classes": {},
        "metric": task.metric.model_copy(update={"name": "rmse"}),
        "output": task.output.model_copy(update={"prediction_type": "values"})})
    pd.DataFrame({"id": ["a", "b"], "answer": ["text", "1"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        score_csv(path, actual, regression)
