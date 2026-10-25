"""Measure the unchanged v081 recipe catalog on frozen action questions, offline."""
import json
from pathlib import Path
import sys
import time


def run(runtime, flat, task_path, output):
    sys.path[:0] = [str(runtime), str(flat)]
    from automl_agent.budget import Limits
    from automl_agent.contracts import KnowledgeSearchArgs, KnowledgeReadArgs
    from automl_agent.controller import Controller
    from automl_agent.util import atomic_json
    questions = [("KNN distance neighbors scaling", {"knn"}),
        ("regularized linear baseline", {"ridge", "elastic_net", "logistic", "sgd"}),
        ("boosting categorical", {"lightgbm", "catboost", "xgboost", "library_search"}),
        ("torch pretrained neural", set())]
    c = Controller(task_path, Limits(seconds=120, cpu=1, memory_mb=2048), run_id="v081_retrieval")
    try:
        c.no_llm = True
        c.prepare()
        rows = []
        for query, expected in questions:
            started = time.monotonic()
            found = c.knowledge_tools.discover(KnowledgeSearchArgs(collection="recipes", query=query))
            entry = found["entries"][0] if found["entries"] else None
            model, executable, reads = None, False, 0
            if entry:
                c.knowledge_tools.read(KnowledgeReadArgs.model_validate(entry["read"]["arguments"]))
                reads += 1
                recipe = c.references.catalog_entry("recipes", entry["id"].removeprefix("recipe:"), c.task, c.plan)
                model = recipe.get("pipeline", {}).get("model")
                executable = recipe.get("applicable") is not False and model is not None
            rows.append({"query": query, "found": entry["id"] if entry else None, "model": model,
                "applicable_recipe": executable, "reads": reads, "seconds": time.monotonic() - started,
                "correct_next_action_or_abstention": executable and model in expected if expected else not executable})
        assert c.budget.counter("llm_calls") == 0
        result = {"runtime": str(runtime), "evidence_type": "offline_retrieval", "questions": rows, "qwen_calls": 0,
            "corpus": c.references.version, "old_catalog": "recipes", "new_catalog": "actions",
            "same_frozen_questions": True, "autonomous_quality_gain_claimed": False}
        atomic_json(output, result)
        return result
    finally:
        c.close()


if __name__ == "__main__":
    run(*map(lambda value: Path(value).resolve(), sys.argv[1:]))
