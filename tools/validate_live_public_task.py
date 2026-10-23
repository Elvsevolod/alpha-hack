"""Validate the flat release with one unchanged public task and the real Qwen proxy.

Runs in the reconstructed exact-dependency image, outside the submitted payload.
Only the three baseline connection variables enter the solution process.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path("/task")


def cgroup_usage():
    root = Path("/sys/fs/cgroup")
    values = {}
    for name in ("memory.current", "memory.peak", "memory.max"):
        path = root / name
        if path.is_file():
            raw = path.read_text().strip()
            values[name] = int(raw) if raw.isdigit() else raw
    path = root / "cpu.stat"
    if path.is_file():
        values["cpu.stat"] = {key: int(value) for key, value in (line.split() for line in path.read_text().splitlines())}
    return values


def connection_environment():
    environment = {key: value for key, value in os.environ.items()
        if not key.startswith(("OPENAI_", "AGENT_", "PYTHONPATH"))}
    provided = []
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        if os.environ.get(key):
            environment[key] = os.environ[key]
            provided.append(key)
    if not environment.get("OPENAI_API_KEY") or not environment.get("OPENAI_BASE_URL"):
        raise RuntimeError("The live validation requires the organizer connection variables")
    return environment, provided


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def snapshot():
    paths = list((ROOT / ".agent_work").glob("*/journal.sqlite"))
    if not paths:
        return {}
    try:
        db = sqlite3.connect("file:" + str(paths[0]) + "?mode=ro", uri=True, timeout=1)
        try:
            data = {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM metadata")}
            counts = dict(db.execute("SELECT kind,COUNT(*) FROM events GROUP BY kind"))
            experiments = [{"id": identifier, "model": json.loads(spec)["pipeline"]["model"],
                "status": json.loads(result)["status"] if result else "running",
                "score": json.loads(result).get("score") if result else None}
                for identifier, spec, result in db.execute("SELECT id,spec,result FROM experiments")]
            return {"state": data.get("state", "preparing"), "model_calls": data.get("llm_calls", 0),
                "model_seconds": data.get("llm_seconds", 0), "actual_qwen_decisions": counts.get("qwen_decision", 0),
                "model_errors": counts.get("qwen_failure", 0), "truncated_responses": counts.get("qwen_response_truncated", 0),
                "candidate_count": data.get("candidate_count", 0), "leader": data.get("leader_id"),
                "validation_best": data.get("raw_best"), "deadline": data.get("deadline"),
                "limits": data.get("limits"), "experiments": experiments,
                "control_status": data.get("control_status"), "refit_status": data.get("refit_status"),
                "first_csv_seconds": data.get("first_csv_seconds"), "first_trained_csv_seconds": data.get("first_trained_csv_seconds"),
                "token_usage": data.get("token_usage"),
                "publication_scope": (data.get("publication") or {}).get("scope"),
                "freeze_reason": (data.get("finalization_plan") or {}).get("reason")}
        finally:
            db.close()
    except sqlite3.Error:
        return {}


def main():
    environment, provided = connection_environment()
    inputs = json.loads((ROOT / "REHEARSAL_INPUT.json").read_text())
    if inputs["kind"] not in {"classification", "regression"}:
        raise RuntimeError("The public rehearsal task family is unknown")
    expected = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        match = re.match(r"([\w.-]+)(?:\[[^]]+\])?==([^\s#]+)", line)
        if match:
            expected[match[1]] = match[2]
    actual = {name: importlib.metadata.version(name) for name in expected}
    mismatches = {name: actual[name] for name in expected if expected[name] != actual[name]}
    if mismatches:
        raise RuntimeError("Dependency versions differ: " + json.dumps(mismatches))
    started = time.time()
    external_deadline = min(started + 10800, inputs["common_external_deadline"])
    if external_deadline <= started:
        raise RuntimeError("The shared external rehearsal deadline has already expired")
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": ["python3", "solution.py", "task.md"], "python": sys.version,
        "dependency_versions": actual, "dependency_mismatches": mismatches,
        "provided_connection_variables": provided, "generation_overrides_provided": False,
        "explicit_task_spec_provided": (ROOT / "task_spec.json").exists(),
        "external_deadline_seconds": 10800, "common_external_deadline": external_deadline,
        "archive_sha256": inputs["archive_sha256"], "source_task_files": inputs["source_task_files"],
        "resource_usage_before": cgroup_usage(), "official_platform_image": False,
        "flat_file_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("solution.py", "agent_bundle.py", "requirements.txt")}}
    save(ROOT / "VALIDATION.json", report)
    if report["flat_file_sha256"] != inputs["flat_file_sha256"]:
        raise RuntimeError("Delivered files differ from the frozen rehearsal input")
    with (ROOT / "RUNTIME.log").open("wb") as log:
        process = subprocess.Popen([sys.executable, "solution.py", "task.md"], cwd=ROOT,
            env=environment, stdout=log, stderr=log, start_new_session=True)
        last_printed = 0
        while process.poll() is None:
            elapsed = time.time() - started
            if time.time() > external_deadline:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                report["external_deadline_exceeded"] = True
                break
            if elapsed - last_printed >= 30:
                current = snapshot()
                save(ROOT / "PROGRESS.json", {"elapsed_seconds": elapsed, "resource_usage": cgroup_usage(), **current})
                print(json.dumps({"elapsed_seconds": round(elapsed), **{key: current.get(key) for key in
                    ("state", "model_calls", "actual_qwen_decisions", "model_errors", "candidate_count", "validation_best", "control_status", "refit_status")}}, ensure_ascii=False), flush=True)
                last_printed = elapsed
            time.sleep(2)
        process.wait()
    report.update(exit_code=process.returncode, elapsed_seconds=time.time() - started, research=snapshot(),
        resource_usage_after=cgroup_usage())
    output = ROOT / "submission.csv"
    checks = {"submission_exists": output.is_file()}
    if output.is_file():
        test = pd.read_csv(ROOT / "test.csv", dtype=str, keep_default_na=False)
        sample = pd.read_csv(ROOT / "sample_submission.csv", dtype=str, keep_default_na=False)
        predictions = pd.read_csv(output, dtype=str, keep_default_na=False)
        target = next(column for column in sample if column != "id")
        checks.update(rows=len(predictions), expected_rows=len(test), columns=list(predictions),
            schema_matches=list(predictions) == list(sample), id_order_identical=predictions["id"].tolist() == test["id"].tolist(),
            duplicate_ids=int(predictions["id"].duplicated().sum()), missing_values=bool((predictions == "").any().any()),
            unique_predictions=int(predictions[target].nunique()), mode=oct(output.stat().st_mode & 0o777),
            sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        if inputs["kind"] == "regression":
            checks["finite_predictions"] = bool(np.isfinite(pd.to_numeric(predictions[target], errors="coerce")).all())
        else:
            training_labels = pd.read_csv(ROOT / "train.csv", usecols=[target], dtype=str)[target]
            checks["unknown_labels"] = sorted(set(predictions[target]) - set(training_labels))
    reports = list((ROOT / ".agent_work").glob("*/run_report.json"))
    report["run_report"] = json.loads(reports[0].read_text()) if reports else None
    report["source_task_unchanged"] = all((ROOT / name).is_file() and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
        for name, expected in inputs["source_task_files"].items())
    report["csv_checks"] = checks
    runtime = report["run_report"] or {}
    report["passed"] = bool(process.returncode == 0 and report["research"].get("actual_qwen_decisions", 0) > 0
        and checks.get("rows") == checks.get("expected_rows") and checks.get("schema_matches")
        and checks.get("id_order_identical") and checks.get("duplicate_ids") == 0
        and not checks.get("missing_values", True) and checks.get("finite_predictions", True)
        and not checks.get("unknown_labels") and checks.get("mode") == "0o644"
        and runtime.get("control_status") == "completed" and runtime.get("refit_status") == "completed"
        and runtime.get("first_trained_csv_seconds") is not None and report["source_task_unchanged"]
        and not report.get("external_deadline_exceeded"))
    report["status"] = "passed" if report["passed"] else "needs_review"
    save(ROOT / "VALIDATION.json", report)
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
        "exit_code": process.returncode, "csv": checks, "research": report["research"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
