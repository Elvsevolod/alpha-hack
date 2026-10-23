"""Short prespecified model assays on permitted research data, never control."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis, PipelineSpec, ToolCommand
from automl_agent.controller import Controller
from automl_agent.data import Dataset
from automl_agent.util import atomic_json, digest, file_hash


def run(source, output, kind):
    output.mkdir(parents=True, exist_ok=False)
    paths = {str(p.relative_to(Path(__file__).resolve().parents[1])): file_hash(p)
        for p in sorted(Path(__file__).resolve().parents[1].joinpath("automl_agent").glob("*.py"))}
    data = Dataset(source)
    task, questions = data.draft()
    if task is None or task.kind != kind:
        raise ValueError(str(questions))
    for name in ("train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv", "task.md"):
        path = data.paths.get(name) if name != "task.md" else source
        if path:
            shutil.copyfile(path, output / name)
    base = {"model": "catboost", "representation": "native", "seed": 42,
        "imputation": {"numeric": "native", "indicators": True},
        "hyperparameters": {"iterations": 500, "depth": 6, "learning_rate": .06, "l2_leaf_reg": 3}}
    variants = []
    if kind == "regression":
        for method in ("native", "median", "iterative_bayesian_ridge"):
            variants.append((method, {**base, "imputation": {"numeric": method, "indicators": True, "max_iter": 3, "n_nearest_features": 20}}))
    else:
        variants.append(("unweighted", base))
        for weights in ("Balanced", "SqrtBalanced"):
            variants.append((weights, {**base, "hyperparameters": {**base["hyperparameters"], "auto_class_weights": weights}}))
        variants.append(("regularized", {**base, "hyperparameters": {**base["hyperparameters"], "l2_leaf_reg": 30}}))
        variants.append(("one_hot_categories", {**base, "hyperparameters": {**base["hyperparameters"], "one_hot_max_size": 32}}))
    manifest = {"stage": "development_assay", "kind": kind, "source_hash": data.source_hash,
        "code_hashes": paths, "code_hash": digest(paths), "variants": [{"id": name, "pipeline": spec} for name, spec in variants],
        "candidate_seconds": 180, "cpu": 6, "memory_limit_gib": 5, "labels_used": "research_validation_only",
        "control_opened": False, "external_benchmark_evaluation_used": False,
        "all_planned_runs_counted": True, "policy": "Existing unchanged metric and fixed split; no claim about final refit or generalization"}
    atomic_json(output / "MANIFEST.json", manifest)
    atomic_json(output / "REGISTRY.json", [{"id": name, "status": "not_started"} for name, _ in variants])
    c = Controller(output / "task.md", Limits(seconds=1800, cpu=6, memory_mb=5120), run_id="model_assay")
    rows = []
    try:
        c.prepare()
        fit = c.dataset.profile(c.plan.fit).model_dump(mode="json")
        categories = {}
        for col in c.task.categorical_columns:
            tr = c.dataset.train.iloc[c.plan.fit][col]
            te = c.dataset.test[col]
            count = tr.value_counts()
            categories[col] = {"fit_distinct": len(count), "fit_rare_le5": int((count <= 5).sum()),
                "unknown_test_fraction": float((~te.isin(count.index)).mean())}
        observations = {"fit_profile": fit, "categories": categories,
            "class_counts_fit": {t: c.dataset.train.iloc[c.plan.fit][t].value_counts().to_dict() for t in c.task.classes},
            "validation_hash": c.plan.hash, "strategy": c.plan.strategy,
            "closed_control_unread": True, "v008": "Prior platform result failed official quality threshold; historical metrics excluded from paired claims"}
        atomic_json(output / "OBSERVATIONS.json", observations)
        for name, config in variants:
            pipeline = PipelineSpec.model_validate(config)
            hypothesis = Hypothesis(observation="Fixed fit profile: missing values and next-period transfer" if kind == "regression" else "Fit class counts and rare/unknown feature categories",
                expected_effect="Measure the isolated component on unchanged research validation",
                changed_component=name, acceptance="Primary metric and complete runtime; do not use closed control or outer evaluation labels")
            result = c.dispatch(ToolCommand(operation_id="assay_" + name, tool="train_model", arguments={
                "pipeline": pipeline.model_dump(mode="json"), "hypothesis": hypothesis.model_dump(), "budget_seconds": 180}))
            rows.append({"id": name, "hypothesis": hypothesis.model_dump(), "actual_pipeline": pipeline.model_dump(mode="json"),
                "outcome": result, "next_agent_choice": "pending_live_research", "status": result.get("result", {}).get("status", result.get("status"))})
            registry = rows + [{"id": other, "status": "not_started"} for other, _ in variants[len(rows):]]
            atomic_json(output / "REGISTRY.json", registry)
            print(json.dumps({"kind": kind, "id": name, "status": rows[-1]["status"],
                "score": result.get("result", {}).get("score")}), flush=True)
        atomic_json(output / "REPORT.json", {"runs": rows, "planned": len(variants), "completed": len(rows),
            "control_status": c.journal.get("control_status", "not_opened"), "final_refit_tested": False,
            "local_quality_goal_achieved": False, "manifest_sha256": file_hash(output / "MANIFEST.json")})
    finally:
        c.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["regression", "classification"], required=True)
    args = parser.parse_args()
    run(args.source, args.output, args.kind)
