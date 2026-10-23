"""Execute registered comparisons without mounting sealed labels into any run."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.comparison_protocol import code_hash, receipt_issues
from automl_agent.util import atomic_json, digest, file_hash, read_json


def docker(*arguments, check=True):
    return subprocess.run(["docker", *map(str, arguments)], check=check, capture_output=True, text=True)


def saved_receipt(directory, run, manifest, source_hash):
    path = directory / "receipt.json"
    if path.is_file():
        return read_json(path)
    work = directory / "work" / ".agent_work" / "comparison"
    report_path = work / "run_report.json"
    report = read_json(report_path) if report_path.is_file() else {}
    metadata, research = {}, []
    journal = work / "journal.sqlite"
    if journal.is_file():
        db = sqlite3.connect("file:" + str(journal) + "?mode=ro", uri=True)
        try:
            metadata = {key: json.loads(value) for key, value in db.execute("SELECT key,value FROM metadata")}
            for spec, result in db.execute("SELECT spec,result FROM experiments ORDER BY rowid"):
                if not result:
                    continue
                spec, result = json.loads(spec), json.loads(result)
                research.append({"experiment_id": spec["experiment_id"], "engine": spec["pipeline"]["engine"],
                    "n_trials": spec["pipeline"]["n_trials"], "budget_mode": spec.get("budget_mode", "research"),
                    "budget_seconds": spec["budget_seconds"], "scope": result["scope"], "status": result["status"],
                    "validation_score": result.get("score"), "budget": result["training_summary"].get("budget"),
                    "error": result.get("error")})
        finally:
            db.close()
    if not report:
        report = {key: metadata.get(key) for key in ("state", "control_status", "refit_status", "publication", "failure", "token_usage")}
    publication = metadata.get("publication") or report.get("publication") or {}
    plan = metadata.get("validation")
    csv = directory / "work" / "submission.csv"
    if csv.is_file():
        shutil.copyfile(csv, directory / "receipt.csv")
    receipt = {"arm": run["arm"], "seed": run["seed"], "report": report,
        "code_sha256": source_hash, "data_version": metadata.get("source_hash"),
        "validation_hash": publication.get("validation_hash"),
        "validation_indices_hash": digest({key: plan[key] for key in ("fit", "validation", "control")}) if plan else None,
        "research": research, "queue": [], "outer_labels_opened": False,
        "csv_sha256": file_hash(csv) if csv.is_file() else None,
        "receipt_recovered_from_original_run": True}
    atomic_json(path, receipt)
    return receipt


def api_limits(phase, directory):
    command = [sys.executable, str(ROOT / "skills/alpha-api-limits/scripts/check_limits.py"), phase,
        "--output", str(directory / ("api_limits_" + phase + ".json"))]
    if phase == "after":
        command += ["--before", str(directory / "api_limits_before.json")]
    return subprocess.run(command, check=False).returncode


def run(protocol, runtime):
    manifest = read_json(protocol / "MANIFEST.json")
    data = read_json(protocol / "DATA_MANIFEST.json")
    registry = read_json(protocol / "REGISTRY.json")
    if any(row["status"] == "running" for row in registry):
        raise ValueError("An interrupted running batch needs outcome recovery; never relaunch it blindly")
    if file_hash(protocol / "DATA_MANIFEST.json") != manifest["data_manifest_sha256"] or code_hash(runtime) != manifest["code_sha256"]:
        raise ValueError("Frozen data manifest or runtime source differs")
    tooling = digest({name: file_hash(runtime / "tools" / name) for name in
        ("benchmark.py", "comparison_protocol.py", "run_comparison_series.py")})
    if tooling != manifest["comparison_tooling_sha256"] or file_hash(runtime / "knowledge_runtime/knowledge.lock.json") != manifest["knowledge_lock_sha256"] or file_hash(runtime / "requirements.txt") != manifest["requirements_sha256"]:
        raise ValueError("Frozen tooling, knowledge or dependency manifest differs")
    archive = ROOT / "dist/solution_v008.zip"
    if file_hash(archive) != manifest["v008_sha256"]:
        raise ValueError("Previous archive changed")
    for kind, entry in data["tasks"].items():
        if any(file_hash(protocol / "tasks" / kind / name) != sha for name, sha in entry["files"].items()):
            raise ValueError("Frozen task files changed")
    load_dotenv(ROOT / ".env", override=False)
    environment = os.environ.copy()
    environment["OPENAI_BASE_URL"] = "http://192.168.97.2:18765/v1"
    (protocol / "receipts").mkdir(exist_ok=True)
    pending = [row for row in registry if row["status"] == "not_started"]
    for offset in range(0, len(pending), 2):
        batch = pending[offset:offset + 2]
        batch_id = "__".join(row["id"] for row in batch)
        checks = protocol / "batches"
        check_number = len(list(checks.glob("*"))) + 1
        batch_dir = checks / (f"{check_number:04d}_" + batch_id)
        batch_dir.mkdir(parents=True, exist_ok=False)
        live = any(row["arm"] in {"agent", "v008"} for row in batch)
        if live and api_limits("before", batch_dir) != 0:
            atomic_json(batch_dir / "NOT_STARTED.json", {"reason": "API check unavailable, blocked or exhausted", "runs": batch})
            return 3
        jobs = []
        try:
            for row in batch:
                directory = protocol / "runs" / row["id"]
                shutil.copytree(protocol / "tasks" / row["kind"], directory / "work")
                name = "alpha-v010-compare-" + row["id"].replace("_", "-")
                command = ["docker", "run", "-d", "--name", name, "--cpus", str(manifest["cpu"]),
                    "--memory", str(manifest["memory_gib"]) + "g", "--network", "alpha-v010-llm" if row["arm"] in {"agent", "v008"} else "none",
                    "-v", str(directory) + ":/run", "--env", "PYTHONDONTWRITEBYTECODE=1"]
                if row["arm"] in {"agent", "v008"}:
                    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_EXTRA_BODY"):
                        if environment.get(key):
                            command += ["--env", key]
                if row["arm"] == "v008":
                    with zipfile.ZipFile(archive) as saved:
                        saved.extractall(directory / "work")
                    source_hash = manifest["v008_code_sha256"]
                    command += ["-w", "/run/work", "qwen-ml-agent:exact", "python", "solution.py", "task.md",
                        "--time-limit", str(manifest["internal_seconds"]), "--cpu", str(manifest["cpu"]),
                        "--memory-mb", str(manifest["memory_gib"] * 1024), "--seed", str(row["seed"]), "--run-id", "comparison"]
                else:
                    source_hash = manifest["code_sha256"]
                    command += ["-v", str(runtime) + ":/runtime:ro", "-w", "/runtime", "qwen-ml-agent:exact", "python",
                        "tools/benchmark.py", "run-one", "--task", "/run/work/task.md", "--output", "/run/receipt.json",
                        "--arm", row["arm"], "--seed", str(row["seed"]), "--seconds", str(manifest["internal_seconds"]),
                        "--engine", manifest["selected_automl"][row["kind"]]]
                row.update(status="running", container=name, started_at=time.time(), source_sha256=source_hash)
                atomic_json(protocol / "REGISTRY.json", registry)
                started = subprocess.run(command, env=environment, capture_output=True, text=True)
                jobs.append((row, directory, name, source_hash, started.returncode))
                if started.returncode:
                    (directory / "launch_error.log").write_text(started.stderr)
            while jobs:
                for job in jobs[:]:
                    row, directory, name, source_hash, launch_error = job
                    state = None
                    if not launch_error:
                        inspected = docker("inspect", "--format", "{{json .State}}", name, check=False)
                        state = json.loads(inspected.stdout) if inspected.returncode == 0 else None
                    if state and state["Running"]:
                        if time.time() - row["started_at"] >= manifest["external_seconds"]:
                            docker("stop", "-t", "2", name)
                            row["external_watchdog_triggered"] = True
                        continue
                    logs = docker("logs", name, check=False)
                    (directory / "process.log").write_text(logs.stdout + logs.stderr)
                    atomic_json(directory / "docker_state.json", state or {"outcome": "unavailable", "launch_exit_code": launch_error})
                    receipt = saved_receipt(directory, row, manifest, source_hash)
                    issues = receipt_issues(receipt, row, manifest, data["tasks"][row["kind"]])
                    exit_code = state.get("ExitCode") if state else launch_error or None
                    if exit_code != 0 or row.get("external_watchdog_triggered"):
                        issues.append("process_failure_or_unknown_outcome")
                    row.update(status="completed" if not issues else "failed", exit_code=exit_code,
                        elapsed_seconds=time.time() - row["started_at"], acceptance_issues=issues,
                        controller_state=receipt.get("report", {}).get("state"),
                        control_status=receipt.get("report", {}).get("control_status"), refit_status=receipt.get("report", {}).get("refit_status"))
                    for suffix in (".json", ".csv"):
                        original = directory / ("receipt" + suffix)
                        if original.is_file():
                            shutil.copyfile(original, protocol / "receipts" / (row["id"] + suffix))
                    atomic_json(protocol / "REGISTRY.json", registry)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    if state:
                        docker("rm", name)
                    jobs.remove(job)
                if jobs:
                    time.sleep(2)
        finally:
            if live:
                # On a launcher error, wait for actual termination before the after snapshot.
                for row, directory, name, source_hash, launch_error in jobs:
                    docker("wait", name, check=False)
                api_limits("after", batch_dir)
        if any(row["status"] != "completed" for row in batch):
            for row in registry:
                if row["status"] == "not_started":
                    row.update(status="cancelled", cancellation_reason="Cohort already failed; repair before a new immutable series")
            atomic_json(protocol / "REGISTRY.json", registry)
            atomic_json(protocol / "SERIES_STATUS.json", {"status": "stopped_incomplete",
                "reason": "At least one run failed the frozen acceptance conditions", "outer_labels_opened": False,
                "local_quality_goal_achieved": False})
            return 1
    return 0 if all(row["status"] == "completed" for row in registry) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.protocol.resolve(), args.runtime.resolve()))
