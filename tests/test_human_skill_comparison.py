"""A better number cannot override mismatched controls or missing outcomes."""
import json

import pytest

from tools.analyze_human_skill_benchmark import analyze


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def finished(tmp_path):
    (tmp_path / "RUN_FINISHED").touch()
    save(tmp_path / "MANIFEST.json", {"seeds": [42]})
    jobs = {}
    for kind, scores, metric in (("regression", [10, 9], "rmse"), ("classification", [.5, .6], "f1")):
        for index, version in enumerate(("base", "skill")):
            arm = f"{kind}_42_{version}"
            task = tmp_path / arm / "task"
            jobs[arm] = {"kind": arm, "directory": str(task), "started": True, "seed": 42, "input_hashes": {"train.csv": "same"}}
            save(task / ".agent_work/diagnostic/task_spec.json", {"kind": kind, "metric": {"name": metric, "average": "macro"}})
            save(task / ".agent_work/diagnostic/validation_plan.json", {"fit": [0, 1], "validation": [2], "control": [3], "seed": 42})
            save(tmp_path / arm / "RESULT.json", {"technical_passed": True, "report": {
                "control_winner": "winner", "control_scores": {"winner": scores[index]},
                "effective_cpu": 2, "effective_memory_mb": 2600,
                "model_configuration": {"model": "same", "reasoning_policy": {"ordinary": "medium", "complex": "xhigh"},
                    "output_tokens": 16384 if index else 32768,
                    "output_by_effort": {},
                    "generation": {"temperature": 1, "reasoning": {"effort": "medium" if index else "xhigh"}}}}})
    save(tmp_path / "CONFIG.json", {"jobs": jobs})
    save(tmp_path / "budget_ledger.json", {"requests": {}, "reported_cost": 0, "inflight": None, "closed": None})
    return tmp_path


def test_rmse_and_f1_have_opposite_directions_and_action_effort_may_differ(finished):
    result = analyze(finished)
    assert result["all_pairs_comparable"]
    assert result["pairs"][0]["directed_gain"] == 1
    assert result["pairs"][1]["directed_gain"] == pytest.approx(.1)
    assert result["mean_relative_gain_by_task"] == pytest.approx({"regression": .1, "classification": .2})
    assert result["quality_advantage_established"] is False


def test_changed_control_cannot_be_treated_as_a_quality_gain(finished):
    path = finished / "regression_42_skill/task/.agent_work/diagnostic/validation_plan.json"
    plan = json.loads(path.read_text())
    plan["control"] = [4]
    save(path, plan)
    result = analyze(finished)
    assert not result["all_pairs_comparable"]
    assert not result["pairs"][0]["checks"]["splits"]
    assert result["pairs"][0]["relative_gain"] is None
    assert result["mean_relative_gain_by_task"]["regression"] is None


def test_missing_selected_control_is_inconclusive(finished):
    path = finished / "classification_42_skill/RESULT.json"
    result = json.loads(path.read_text())
    result["report"]["control_scores"] = None
    save(path, result)
    assert not analyze(finished)["pairs"][1]["comparable"]


def test_live_guest_databases_are_never_read(finished):
    (finished / "RUN_FINISHED").unlink()
    with pytest.raises(ValueError, match="Wait for all containers"):
        analyze(finished)
