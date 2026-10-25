"""Offline cache equality/profile and frozen question-to-action acceptance."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from conftest import write_task
from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis, KnowledgeSearchArgs, KnowledgeReadArgs, ToolCommand, RecipeApplication
from automl_agent.controller import Controller
from automl_agent.learned_blend import member_oof, oof_directory
from automl_agent.recipe_patch import resolve
from automl_agent.tool_results import payload
from automl_agent.util import atomic_json, read_json, file_hash


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    path = write_task(output / "fixture")
    c = Controller(path, Limits(seconds=180, cpu=1, memory_mb=1024), run_id="mechanisms")
    try:
        c.no_llm = True
        c.prepare()
        result = payload(c.dispatch(ToolCommand(operation_id="ridge", tool="train_model", arguments={
            "pipeline": {"model": "ridge", "hyperparameters": {"alpha": 3}}, "budget_seconds": 15,
            "hypothesis": Hypothesis(observation="Fixed synthetic fit-only mechanism fixture", expected_effect="Verify exact cache equivalence",
                changed_component="model", acceptance="Unchanged rows and probabilities on cache hit").model_dump(mode="json")})))
        if result["result"]["status"] != "completed":
            raise ValueError("Mechanism reference did not train")
        identifier = result["experiment"]["experiment_id"]
        started = time.monotonic(); first = member_oof(c, identifier, 30); first_seconds = time.monotonic() - started
        started = time.monotonic(); cached = member_oof(c, identifier, 30); cached_seconds = time.monotonic() - started
        spec, _ = c.lookup(identifier)
        directory = oof_directory(c, spec, c.plan.fit)
        preserved = directory.with_name(directory.name + "_explicit_no_cache_comparison")
        directory.rename(preserved)
        started = time.monotonic(); fresh = member_oof(c, identifier, 30); fresh_seconds = time.monotonic() - started
        equal = first["row_keys"] == fresh["row_keys"] == cached["row_keys"] and all(
            np.allclose(first["values"][target], fresh["values"][target], rtol=1e-8, atol=1e-10)
            and np.array_equal(first["values"][target], cached["values"][target]) for target in c.task.targets)
        if not equal:
            raise ValueError("Cache hit differs from independently recomputed OOF")
        cache = {"evidence_type": "offline_mechanism", "first_seconds": first_seconds, "cached_full_step_seconds": cached_seconds,
            "forced_recompute_full_step_seconds": fresh_seconds, "same_rows_and_values": equal, "numerical_tolerance": {"rtol": 1e-8, "atol": 1e-10},
            "scope": first["scope"], "folds_hash": first["folds_hash"], "task_local": True, "new_cache_layer_added": False,
            "memory_manifest": read_json(directory / "worker_result.json"), "agent_quality_gain_claimed": False}
        questions = [("KNN distance neighbors scaling", "action:knn"), ("regularized linear baseline", "action:linear"),
            ("boosting categorical", "action:boosting"), ("torch pretrained neural", None)]
        answers = []
        for query, expected in questions:
            started = time.monotonic()
            found = c.knowledge_tools.discover(KnowledgeSearchArgs(collection="actions", query=query))
            action = found["entries"][0] if found["entries"] else None
            applied = None
            if action:
                card = c.knowledge_tools.read(KnowledgeReadArgs.model_validate(action["read"]["arguments"]))["entry"]
                applied, _ = resolve(c, RecipeApplication.model_validate(card["apply"]))
            answers.append({"query": query, "expected": expected, "found": action["id"] if action else None,
                "applicable_recipe": applied is not None, "seconds": time.monotonic() - started,
                "passed": (action["id"] if action else None) == expected})
        if not all(row["passed"] for row in answers) or c.budget.counter("llm_calls"):
            raise ValueError("Frozen actionable retrieval failed or unexpectedly requested Qwen")
        report = {"cache": cache, "questions": answers, "source_hashes": {str(p.relative_to(ROOT)): file_hash(p)
            for p in (ROOT / "automl_agent/action_cards.py", ROOT / "automl_agent/learned_blend.py")},
            "qwen_calls": c.budget.counter("llm_calls"), "status": "passed", "autonomous_quality_effect": "unestablished"}
        atomic_json(output / "report.json", report)
        return report
    finally:
        c.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    print(run(parser.parse_args().output)["status"])
