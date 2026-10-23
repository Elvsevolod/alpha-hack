"""Run both untouched public tasks only after quality and live acceptance pass."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
import zipfile

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_agent.util import atomic_json, file_hash, read_json
from tools.run_comparison_series import api_limits, docker


def stock_generation_profile():
    from automl_agent.journal import Journal
    from automl_agent.qwen import QwenClient
    override = os.environ.pop("OPENAI_EXTRA_BODY", None)
    try:
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory))
            client = QwenClient(journal, None)
            try:
                return client.extra
            finally:
                client.close()
                journal.close()
    finally:
        if override is not None:
            os.environ["OPENAI_EXTRA_BODY"] = override


def require_generation_alignment(comparison):
    profiles = []
    for row in read_json(comparison / "MANIFEST.json")["runs"]:
        if row["arm"] == "agent":
            receipt = read_json(comparison / "receipts" / (row["id"] + ".json"))
            profile = receipt["report"].get("model_configuration")
            if not profile or "generation" not in profile:
                raise ValueError("Comparison generation metadata is missing")
            profiles.append(profile["generation"])
    standard = stock_generation_profile()
    if not profiles or any(profile != standard for profile in profiles):
        raise ValueError("Measured generation settings differ from the stock delivery; establish archive applicability before the final rehearsal")
    return standard


def require_quality(archive, comparison):
    sha = file_hash(archive)
    manifest = read_json(comparison / "MANIFEST.json")
    quality = read_json(comparison / "OUTER_EVALUATION.json")
    if manifest["candidate_archive_sha256"] != sha or quality.get("manifest_sha256") != file_hash(comparison / "MANIFEST.json") or quality.get("local_quality_goal_achieved") is not True:
        raise ValueError("The exact candidate must first pass its frozen quality comparison")
    return sha


def require_prerequisites(archive, comparison, live):
    sha = require_quality(archive, comparison)
    acceptance = read_json(live / "REPORT.json")
    if acceptance.get("archive_sha256") != sha or acceptance.get("manifest_sha256") != file_hash(live / "MANIFEST.json") or acceptance.get("planned") != 33 or acceptance.get("total") != 33 or acceptance.get("passed") != 33 or acceptance.get("all_passed") is not True:
        raise ValueError("The exact candidate must first pass all 33 initial live cases")
    return sha


def run(archive, comparison, live, source, output):
    archive_sha = require_prerequisites(archive, comparison, live)
    load_dotenv(ROOT / ".env", override=False)
    generation = require_generation_alignment(comparison)
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["OPENAI_BASE_URL"] = "http://192.168.97.2:18765/v1"
    if not environment.get("OPENAI_API_KEY"):
        raise ValueError("Organizer model credentials are absent")
    with zipfile.ZipFile(archive) as saved:
        names = saved.namelist()
        if set(names) != {"solution.py", "agent_bundle.py", "requirements.txt"} or len(names) != 3:
            raise ValueError("Rehearsal expects the verified three-file flat candidate")
        flat_hashes = {}
        for name in names:
            flat_hashes[name] = hashlib.sha256(saved.read(name)).hexdigest()
        if saved.read("requirements.txt") != (ROOT / "data_for_participants/requirements.txt").read_bytes():
            raise ValueError("Candidate dependency file differs from the original")
        jobs = []
        for kind, folder in (("regression", "Регрессия"), ("classification", "Многоклассовая классификация")):
            task = source / folder
            if (task / "task_spec.json").exists():
                raise ValueError("The unchanged public task must not contain a manually added task contract")
            directory = output / kind
            directory.mkdir()
            files = {}
            for name in ("task.md", "train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv"):
                if (task / name).is_file():
                    shutil.copyfile(task / name, directory / name)
                    files[name] = file_hash(task / name)
            if not {"task.md", "train.csv", "test.csv", "sample_submission.csv"} <= files.keys():
                raise ValueError("An original public input is missing")
            saved.extractall(directory)
            jobs.append({"kind": kind, "directory": directory, "source_task_files": files, "status": "not_started"})
    manifest = {"archive_sha256": archive_sha, "comparison_manifest_sha256": file_hash(comparison / "MANIFEST.json"),
        "quality_report_sha256": file_hash(comparison / "OUTER_EVALUATION.json"), "live_report_sha256": file_hash(live / "REPORT.json"),
        "validator_sha256": file_hash(ROOT / "tools/validate_live_public_task.py"),
        "command": ["python3", "solution.py", "task.md"], "internal_seconds": 10500, "external_seconds": 10800,
        "cpu_per_task": 6, "memory_gib_per_task": 5, "generation_overrides_provided": False,
        "stock_generation_matches_quality_evidence": True, "stock_generation": generation,
        "scoring_scope": "technical rehearsal of unchanged public tasks; official quality thresholds unknown",
        "planned": [row["kind"] for row in jobs], "technical_failure_rate_allowed": 0}
    atomic_json(output / "MANIFEST.json", manifest)
    if api_limits("before", output) != 0:
        atomic_json(output / "NOT_STARTED.json", {"reason": "API budget check unavailable, exhausted or blocked"})
        return 3
    started = time.time()
    common_deadline = started + 10800
    slug = re.sub(r"[^a-z0-9-]", "-", output.name.lower())[:25]
    try:
        for job in jobs:
            directory = job["directory"]
            atomic_json(directory / "REHEARSAL_INPUT.json", {"archive_sha256": archive_sha, "kind": job["kind"],
                "common_external_deadline": common_deadline, "flat_file_sha256": flat_hashes,
                "source_task_files": job["source_task_files"]})
            name = "alpha-rehearsal-" + slug + "-" + job["kind"]
            command = ["docker", "run", "-d", "--name", name, "--network", "alpha-v010-llm", "--cpus", "6", "--memory", "5g",
                "-v", str(directory) + ":/task", "-v", str(ROOT / "tools/validate_live_public_task.py") + ":/validation.py:ro",
                "-w", "/task", "--env", "PYTHONDONTWRITEBYTECODE=1"]
            for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
                if environment.get(key):
                    command += ["--env", key]
            command += ["qwen-ml-agent:exact", "python", "/validation.py"]
            result = subprocess.run(command, env=environment, capture_output=True, text=True)
            job.update(container=name, status="running" if result.returncode == 0 else "failed", launch_exit_code=result.returncode)
            if result.returncode:
                (directory / "launch_error.log").write_text(result.stderr)
        atomic_json(output / "REGISTRY.json", [{key: str(value) if isinstance(value, Path) else value for key, value in job.items()} for job in jobs])
        while any(job["status"] == "running" for job in jobs):
            for job in jobs:
                if job["status"] != "running":
                    continue
                name, directory = job["container"], job["directory"]
                state = json.loads(docker("inspect", "--format", "{{json .State}}", name).stdout)
                if state["Running"]:
                    if time.time() >= common_deadline:
                        docker("stop", "-t", "2", name)
                        job["common_watchdog_triggered"] = True
                    continue
                atomic_json(directory / "docker_state.json", state)
                logs = docker("logs", name, check=False)
                (directory / "container.log").write_text(logs.stdout + logs.stderr)
                result_path = directory / "VALIDATION.json"
                validation = read_json(result_path) if result_path.is_file() else {}
                job.update(status="completed" if state["ExitCode"] == 0 and validation.get("passed") and not job.get("common_watchdog_triggered") else "failed",
                    exit_code=state["ExitCode"], validation_passed=validation.get("passed", False))
                docker("rm", name)
                print(json.dumps({key: value for key, value in job.items() if key not in {"directory", "source_task_files"}}), flush=True)
            atomic_json(output / "REGISTRY.json", [{key: str(value) if isinstance(value, Path) else value for key, value in job.items()} for job in jobs])
            if any(job["status"] == "running" for job in jobs):
                time.sleep(2)
    finally:
        # Only the two owned jobs are considered; after is taken after actual termination.
        for job in jobs:
            if job["status"] == "running":
                docker("stop", "-t", "2", job["container"], check=False)
                docker("wait", job["container"], check=False)
                state = docker("inspect", "--format", "{{json .State}}", job["container"], check=False)
                atomic_json(job["directory"] / "supervisor_interruption.json", {"docker_state": json.loads(state.stdout) if state.returncode == 0 else None,
                    "reason": "Rehearsal supervisor interrupted; no blind restart"})
                job["status"] = "cancelled"
        atomic_json(output / "REGISTRY.json", [{key: str(value) if isinstance(value, Path) else value for key, value in job.items()} for job in jobs])
        after_status = api_limits("after", output)
    passed = all(job["status"] == "completed" for job in jobs)
    atomic_json(output / "REPORT.json", {"archive_sha256": archive_sha, "manifest_sha256": file_hash(output / "MANIFEST.json"),
        "planned": 2, "completed": sum(job["status"] == "completed" for job in jobs), "all_passed": passed,
        "elapsed_seconds": time.time() - started, "common_external_deadline": common_deadline,
        "api_after_check_exit_code": after_status, "api_after_confirmed": after_status in {0, 3},
        "official_threshold_exceeded": "unknown", "quality_evidence": str(comparison / "OUTER_EVALUATION.json")})
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.archive.resolve(), args.comparison.resolve(), args.live.resolve(), args.source.resolve(), args.output.resolve()))
