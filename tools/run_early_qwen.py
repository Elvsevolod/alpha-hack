"""Live autonomous follow-up to prespecified development assays."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automl_agent.budget import Limits
from automl_agent.controller import Controller
from automl_agent.util import atomic_json, file_hash, digest


def run(source, destination, assay, seconds):
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.name == "task.md" or path.suffix == ".csv" and path.name != "submission.csv":
            shutil.copyfile(path, destination / path.name)
    prior = json.loads(assay.read_text())
    evidence = [{"id": row["id"], "pipeline": row["actual_pipeline"], "status": row["status"],
        "research_validation_score": row.get("outcome", {}).get("result", {}).get("score"),
        "imputation": row.get("outcome", {}).get("result", {}).get("training_summary", {}).get("imputation", {}).get("method")}
        for row in prior["runs"]]
    hashes = {str(p.relative_to(Path(__file__).resolve().parents[1])): file_hash(p)
        for p in sorted(Path(__file__).resolve().parents[1].joinpath("automl_agent").rglob("*.py"))}
    focus = "Development follow-up to a frozen short model assay on this same research protocol. The following are permitted aggregate validation results, not outer evaluation or closed-control error analyses. Choose your next hypotheses freely based on these results and fit observations; do not repeat a failed component without a concrete changed reason. Explain observation, actual configuration change and expected effect. You may consult existing references, tune, ensemble or create Python components. Previous assays: " + json.dumps(evidence)
    atomic_json(destination / "LIVE_MANIFEST.json", {"code_hash": digest(hashes), "code_hashes": hashes,
        "assay_sha256": file_hash(assay), "provided_evidence": evidence, "seconds": seconds, "cpu": 6,
        "memory_gib": 5, "training_seed": 42, "max_calls": None, "max_candidates": None,
        "closed_control_returned_to_research": False, "outer_evaluation_used": False})
    c = Controller(destination / "task.md", Limits(seconds=seconds, cpu=6, memory_mb=5120),
        run_id="early_qwen", training_seed=42, research_focus=focus)
    try:
        report = c.run()
        atomic_json(destination / "REPORT.json", report)
        print(json.dumps({key: report.get(key) for key in ("state", "research_candidates", "model_calls", "control_status", "refit_status", "failure")}))
    finally:
        c.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assay", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1200)
    args = parser.parse_args()
    run(args.source, args.output, args.assay, args.seconds)
