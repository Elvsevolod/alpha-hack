import copy

import pytest

from tools.plan_acceptance import evaluate
from tools.plan_agent_comparison import run


def evidence(kind="offline_recipe", seeds=(42,)):
    manifest = {"versions": {name: {} for name in ("v076", "v081", "candidate")},
        "seeds": list(seeds), "resources": {"cpu": 2, "memory_mb": 4096, "seconds": 45},
        "protocol_hash": "frozen", "splits": {}}
    rows = []
    for task, metric, base, candidate in (("regression", "rmse", 50., 49.), ("classification", "f1", .6, .61)):
        schema = {"columns": ["id", "answer"], "prediction_type": "values" if task == "regression" else "labels"}
        manifest["splits"][task] = {"data_hash": "data_" + task, "evaluation_keys_hash": "rows_" + task,
            "task": {"metric": {"name": metric}, "output": schema}}
        for version in manifest["versions"]:
            for seed in seeds:
                value = candidate if version == "candidate" else base
                rows.append({"version": version, "task": task, "seed": seed, "resources": manifest["resources"],
                    "protocol_hash": "frozen", "data_hash": "data_" + task, "evaluation_keys_hash": "rows_" + task,
                    "output_schema": schema, "raw_metric": value, "stress_metric": value, "valid_csv": True,
                    "status": "completed", "fallback": False, "evidence_type": kind, "real_qwen": kind == "full_qwen_agent"})
    return manifest, rows


def test_offline_gain_and_three_repeats_cannot_claim_full_agent_advantage():
    manifest, rows = evidence(seeds=(42, 1729, 314159))
    result = evaluate(manifest, rows)
    assert result["complete"] and result["practical_gain_observed"]
    assert result["status"] == "quality_advantage_unestablished"
    assert not result["automatic_promotion"] and not result["full_agent_evidence"]


@pytest.mark.parametrize("field,value", [("resources", {}), ("data_hash", "changed"),
    ("evaluation_keys_hash", "changed"), ("output_schema", {}), ("protocol_hash", "changed")])
def test_gate_rejects_incomparable_rows(field, value):
    manifest, rows = evidence()
    rows[-1][field] = value
    with pytest.raises(ValueError, match="Incomparable"):
        evaluate(manifest, rows)


def test_failed_runs_and_early_fallbacks_remain_in_the_denominator():
    manifest, rows = evidence()
    rows[-1].update(status="timeout", fallback=True)
    result = evaluate(manifest, rows)
    assert result["valid_submission_rate"] == 1
    rows[-1].update(valid_csv=False, raw_metric=None)
    result = evaluate(manifest, rows)
    assert result["valid_submission_rate"] == pytest.approx(5 / 6)
    assert result["all_failures_and_fallbacks_counted"]


def test_gate_never_mixes_mock_recipe_and_full_agent_results():
    manifest, rows = evidence()
    rows[0]["evidence_type"] = "mock"
    with pytest.raises(ValueError, match="cannot be mixed"):
        evaluate(manifest, rows)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(manifest, rows + [copy.deepcopy(rows[0])])


def test_unexplained_stress_reversal_prevents_quality_review():
    manifest, rows = evidence("full_qwen_agent", (42, 1729, 314159))
    assert evaluate(manifest, rows)["status"] == "eligible_for_quality_review"
    for row in rows:
        if row["version"] == "candidate" and row["task"] == "classification":
            row["stress_metric"] = .5
    result = evaluate(manifest, rows)
    assert result["status"] == "quality_advantage_unestablished"
    assert not result["comparisons"]["classification"]["v076"]["stress_direction_consistent"]


def test_live_runner_without_explicit_cap_stops_before_reading_files_or_network(tmp_path):
    with pytest.raises(ValueError, match="explicitly authorized"):
        run(tmp_path / "does_not_exist", None)


@pytest.mark.parametrize("strategy", ["uniform", "weighted_oof", "stacked_oof"])
def test_fixed_composition_worker_uses_fit_only_oof(tabular, tmp_path, strategy):
    from automl_agent.contracts import PipelineSpec
    from tools.plan_recipe_worker import composition
    from test_pipelines import context
    X, y, task = tabular
    spec = PipelineSpec(engine="ensemble", model="blend", members=["a", "b"], ensemble_strategy=strategy)
    members = [{"model": "ridge", "hyperparameters": {"alpha": alpha}} for alpha in (1., 10.)]
    bundle, summary, frozen = composition(spec, members, task, X, y, X.iloc[:5], context(task, tmp_path))
    assert bundle.row_keys == X.index[:5].tolist()
    assert len(bundle.values["answer"]) == 5 and frozen.ensemble_strategy == strategy
    assert summary.budget["all_meta_stages_fit_only"]
    assert summary.budget["strict_oof_members"] == (0 if strategy == "uniform" else 2)


def test_composition_offsets_follow_combined_probabilities(tabular, tmp_path):
    import numpy as np
    from automl_agent.contracts import PipelineSpec, MetricSpec, OutputSpec
    from automl_agent.decision import decide
    from tools.plan_recipe_worker import composition
    from test_pipelines import context
    X, y, original = tabular
    y["answer"] = np.where(y.answer > .6, "c", np.where(y.answer > -.2, "b", "a"))
    task = original.model_copy(update={"kind": "classification", "classes": {"answer": ["a", "b", "c"]},
        "metric": MetricSpec(name="f1"), "output": OutputSpec(columns=["identifier", "answer"], prediction_type="labels")})
    spec = PipelineSpec(engine="ensemble", model="blend", members=["a", "b"], ensemble_strategy="stacked_oof",
        prediction_policy="class_offsets", postprocess_fit_policy="pooled_oof")
    members = [{"model": "logistic", "hyperparameters": {"C": value}} for value in (.1, 1.)]
    bundle, _, frozen = composition(spec, members, task, X, y, X.iloc[:5], context(task, tmp_path))
    expected = decide(bundle.raw_probabilities["answer"], task.classes["answer"], frozen.learned_decision["answer"])
    assert bundle.values["answer"] == expected.tolist() and bundle.prediction_type == "labels"


def test_offline_worker_accepts_a_relative_result_directory(tmp_path, monkeypatch):
    from pathlib import Path
    from automl_agent.data import Dataset
    from tools.plan_development_benchmark import ROOT, run_one
    from conftest import write_task
    dataset = Dataset(write_task(tmp_path / "task"))
    dataset.activate(dataset.explicit())
    manifest = {"versions": {"candidate": {"runtime_root": str(ROOT)}}, "protocol_hash": "fixed",
        "resources": {"cpu": 1, "memory_mb": 1024, "seconds": 15}, "seed": 42}
    split = {"fit": list(range(100)), "evaluation": list(range(100, 120)), "stress": list(range(100, 120)),
        "data_hash": dataset.source_hash, "task": dataset.task.model_dump(mode="json")}
    monkeypatch.chdir(tmp_path)
    result = run_one(manifest, Path("relative_result"), dataset, split, "candidate", {"model": "ridge"}, "fixed")
    assert result["status"] == "completed" and result["valid_csv"] and not result["fallback"]


@pytest.mark.parametrize("invalid", [False, True])
def test_agent_result_scoring_keeps_stress_and_invalid_csv(tmp_path, invalid):
    from automl_agent.data import Dataset
    from tools.plan_agent_comparison import score_job
    from conftest import write_task
    dataset = Dataset(write_task(tmp_path / "source"))
    dataset.activate(dataset.explicit())
    directory = tmp_path / "isolated_job"
    directory.mkdir()
    evaluation = list(range(20))
    frame = dataset.train.iloc[evaluation][dataset.task.output.columns].copy()
    if invalid:
        frame.loc[frame.index[0], "answer"] = float("inf")
    frame.to_csv(directory / "submission.csv", index=False)
    split = {"data_hash": dataset.source_hash, "evaluation_keys_hash": "fixed", "evaluation": evaluation, "stress": evaluation[:10]}
    manifest = {"protocol_hash": "fixed", "resources": {"cpu": 1}, "splits": {"regression": split}}
    job = {"directory": str(directory), "version": "candidate", "task": "regression", "seed": 42, "entry_mono": 0, "finished_mono": 1}
    result = score_job(manifest, job, dataset)
    if invalid:
        assert not result["valid_csv"] and result["status"] == "invalid_csv" and result["raw_metric"] is None
    else:
        assert result["valid_csv"] and result["raw_metric"] == 0 and result["stress_metric"] == 0
