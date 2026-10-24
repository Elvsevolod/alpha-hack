"""Standalone offline EDA: no controller, provider, model fitting or publication."""
from __future__ import annotations

import time
from pathlib import Path

from context_compressor import ContextStore

from ..budget import Budget, Limits
from ..contracts import DatasetReportArgs
from ..data import Dataset
from ..executor import Executor
from ..journal import Journal
from ..util import atomic_bytes, atomic_json
from ..validation import choose_validation
from .repository import EDARepository
from .service import EDAService


def run(task_path: Path, output: Path, contract_path: Path | None = None,
        columns: list[str] | None = None, seconds=120., cpu=1, memory_mb=4096., seed=42):
    started = time.time()
    if seconds <= 0 or cpu < 1 or memory_mb < 128:
        raise ValueError("Invalid EDA resource limit")
    output = output.resolve()
    root = task_path.resolve().parent
    if output.exists():
        raise ValueError("EDA output already exists; choose a new output directory")
    if output == root or root.is_relative_to(output):
        raise ValueError("EDA output cannot contain the input task directory")
    if output.is_relative_to(root) and output.relative_to(root).parts[0] != "outputs":
        raise ValueError("Place EDA output outside the task directory, or inside its outputs/ directory")
    if not (contract_path or root / "task_spec.json").is_file():
        from ..data import ContractUnresolved
        raise ContractUnresolved("Without LLM an explicit task_spec.json contract is required")
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output)
    store = None
    try:
        budget = Budget(journal, Limits(seconds=seconds, cpu=cpu, memory_mb=memory_mb,
            max_calls=0, max_candidates=0), started)
        executor = Executor(cpu, memory_mb, budget.deadline, journal)
        from .loading import BoundedCSVLoader
        dataset = Dataset(task_path, deferred=True)
        dataset.csv_loader = BoundedCSVLoader(executor, budget, output)
        dataset.deadline = budget.deadline
        if budget.remaining > 0:
            dataset.activate(dataset.explicit(contract_path))
            plan = choose_validation(dataset, seed)
        else:
            raise TimeoutError("Original deadline exhausted during inventory")
        store = ContextStore(output / "journal.sqlite", tool_api_version=2)
        service = EDAService(dataset, EDARepository(store), executor, budget, output, lambda: ())
        reference, document = service.profile(plan)
        atomic_json(output / "profile.json", document)
        atomic_json(output / "task_spec.json", dataset.task)
        atomic_json(output / "validation_plan.json", plan)
        correlation = None
        if columns is not None:
            correlation, _ = service.report(DatasetReportArgs(report="correlations", columns=columns,
                include_target=bool(set(columns) & set(dataset.task.targets))), plan)
            atomic_json(output / "correlations.json", correlation)
        status = "completed" if document.get("status", "available") == "available" and budget.remaining > 0 and (correlation is None or correlation.get("status") == "available") else "partial"
        summary = {"state": status, "output": str(output), "scope": "fit", "validation_strategy": plan.strategy,
            "profile": reference["summary"], "correlations_status": correlation.get("status") if correlation else "not_requested",
            "elapsed_seconds": time.time() - started, "model_fits": 0, "api_calls": 0,
            "submission_written": False}
        atomic_json(output / "summary.json", summary)
        lines = ["# Offline EDA", "", f"Status: {status}", f"Validation: {plan.strategy}",
            f"Fit rows: {len(plan.fit)}; features: {len(dataset.task.feature_columns)}", "",
            "No API calls, model fitting or submission.csv publication.", "", "## Findings", ""]
        for finding in document["findings"]:
            lines.extend([f"- **{finding['kind']}**: {finding['observation']}",
                f"  {finding['limitation']} {finding['next_question']}"])
        if not document["findings"]:
            lines.append("No configured risk heuristic triggered; this is not proof of data quality.")
        lines.extend(["", "Details: profile.json, task_spec.json, validation_plan.json, summary.json."])
        if correlation is not None:
            lines.extend(["", f"Correlations: {correlation.get('status')}; see correlations.json."])
        atomic_bytes(output / "REPORT.md", ("\n".join(lines) + "\n").encode())
        return summary
    except TimeoutError as error:
        summary = {"state": "partial", "stop_reason": str(error), "profile": {"status": "budget_too_small"},
            "model_fits": 0, "api_calls": 0, "submission_written": False, "elapsed_seconds": time.time()-started}
        atomic_json(output / "summary.json", summary)
        return summary
    finally:
        if store is not None:
            store.close()
        journal.close()
