"""Execution receipts do not establish explanation or competition quality."""
from .util import atomic_bytes


def verification_dimensions():
    return {"constraints": "Runtime receipts and resource checks; not a new submission certificate.",
        "decision_quality": "unassessed; completing a run does not verify the explanation.",
        "predictive_quality": "Task-local measurements only; competition and external superiority unestablished."}


def write_full_summary(directory, report):
    lines = ["# Research run report", "", f"Execution: {report['state']}.",
        f"Control: {report['control_status']}; refit: {report['refit_status']}.",
        f"Model calls: {report['model_calls']}; research candidates: {report['research_candidates']}.", ""]
    lines += [f"- {key}: {value}" for key, value in report["verification_dimensions"].items()]
    lines += ["", f"Task hash: {report['task_hash']}.",
        "Exact measurements, configuration and stop reason are in run_report.json.",
        "A recorded stop reason is not an independently verified scientific explanation.", ""]
    atomic_bytes(directory / "REPORT.md", "\n".join(lines).encode("utf-8"))
