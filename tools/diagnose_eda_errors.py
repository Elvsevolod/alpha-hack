"""Offline development diagnostic; only staged fit labels reach the worker."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from automl_agent.budget import Budget, Limits
from automl_agent.data import Dataset
from automl_agent.executor import Executor, stage_frame
from automl_agent.journal import Journal
from automl_agent.util import atomic_json, digest
from automl_agent.eda.temporal import choose_eda_validation, evaluation_contract


def execute(task_path, contract, output, seconds=180, cpu=2, pipeline=None, baseline_report=None, evaluation=None):
    from automl_agent.contracts import PipelineSpec, EvaluationDesign
    from automl_agent.eda.policy import validate_pipeline
    if pipeline is not None:
        pipeline=PipelineSpec.model_validate(pipeline)
        validate_pipeline(pipeline)
    design=EvaluationDesign.model_validate(evaluation) if evaluation else None
    started = time.time()
    output = output.resolve()
    if seconds <= 0 or cpu < 1:
        raise ValueError("Invalid diagnostic resource limit")
    if output.exists():
        raise ValueError("Choose a new output directory")
    dataset = Dataset(task_path, deferred=True)
    dataset.activate(dataset.explicit(contract))
    plan = choose_eda_validation(dataset, 42)
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output)
    try:
        budget = Budget(journal, Limits(seconds=seconds, cpu=cpu, memory_mb=4096), started)
        executor = Executor(cpu, 4096, budget.deadline, journal)
        directory = output / "worker"
        directory.mkdir()
        X = dataset.features(plan.fit)
        X.index = [dataset.row_keys[i] for i in plan.fit]
        y = dataset.train.iloc[plan.fit][dataset.task.targets].copy()
        y.index = X.index
        stage_frame(directory / "fit_X.json", X)
        stage_frame(directory / "fit_y.json", y)
        context = {"cpu": executor.cpu, "memory_mb": executor.memory_mb, "strategy": plan.strategy,
            "protocol_hash": plan.hash, "task": dataset.task.model_dump(mode="json"), "workdir": str(directory),
            "evaluation_contract":evaluation_contract(dataset, plan,design)}
        request = {"mode": "eda_errors", "task": dataset.task.model_dump(mode="json"), "seed": 42, "context": context}
        if pipeline is not None:
            request["pipeline"]=pipeline.model_dump(mode="json")
        if baseline_report is not None:
            request["baseline_temporal_report"]=baseline_report
        result = executor.run(directory, request, max(.1, budget.remaining))
        result.update(data_version=dataset.source_hash, outer_protocol_hash=plan.hash,
            staged_fit_rows=len(X), staged_fit_keys_hash=digest(list(X.index)), elapsed_seconds=time.time() - started)
        atomic_json(output / "result.json", result)
        return result
    finally:
        journal.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=180)
    parser.add_argument("--evaluation", type=Path, help="Predeclared evaluation design JSON")
    parser.add_argument("--pipeline", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    args = parser.parse_args()
    pipeline=json.loads(args.pipeline.read_text()) if args.pipeline else None
    baseline=json.loads(args.baseline_report.read_text()) if args.baseline_report else None
    if baseline is not None:
        baseline=baseline.get("report",baseline)
    result = execute(args.task, args.task_spec, args.output, args.time_limit, pipeline=pipeline, baseline_report=baseline, evaluation=json.loads(args.evaluation.read_text()) if args.evaluation else None)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "completed" else 1)
