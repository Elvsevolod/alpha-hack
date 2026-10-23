"""Host supervisor for the archive-bound 33-case live suite, after a quality win."""
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

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_agent.util import atomic_json, digest, file_hash, read_json
from tools.comparison_protocol import archived_runtime, code_hash
from tools.live_acceptance import SCENARIOS, PROTOCOL_VERSION
from tools.run_comparison_series import api_limits, docker
from tools.run_public_rehearsal import require_generation_alignment, require_quality

HARNESS_FILES = ("live_acceptance.py", "live_acceptance_extensions.py", "acceptance_fixtures.py", "comparison_protocol.py")


def freeze_harness(archive, destination):
    contents = archived_runtime(archive)
    embedded_hash = digest({name: hashlib.sha256(body).hexdigest() for name, body in contents.items()
        if name == "solution.py" or name.startswith(("automl_agent/", "context_compressor/")) and name.endswith(".py")})
    if embedded_hash != code_hash(ROOT):
        raise ValueError("The live supervisor source must match the measured archive")
    for name, body in contents.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unexpected embedded runtime path")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    for name in HARNESS_FILES:
        target = destination / "tools" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "tools" / name, target)
    return {str(path.relative_to(destination)): file_hash(path) for path in sorted(destination.rglob("*")) if path.is_file()}


def container_command(name, output, seconds):
    command = ["docker", "run", "-d", "--name", name, "--network", "alpha-v010-llm", "--cpus", "1", "--memory", "5g",
        "-v", str(output / "runtime") + ":/runtime:ro", "-v", str(output) + ":/results", "-w", "/runtime",
        "--env", "PYTHONDONTWRITEBYTECODE=1"]
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        if os.environ.get(key):
            command += ["--env", key]
    return command + ["qwen-ml-agent:exact", "python", "tools/live_acceptance.py", "--archive", "/results/candidate.zip",
        "--before", "/results/api_limits_before.json", "--output", "/results", "--attempt-id", "cases",
        "--seconds", str(seconds), "--repeats", "3"]


def run(archive, comparison, output, seconds=90):
    archive_sha = require_quality(archive, comparison)
    load_dotenv(ROOT / ".env", override=False)
    generation = require_generation_alignment(comparison)
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Organizer model credentials are absent")
    output.mkdir(parents=True, exist_ok=False)
    runtime_hashes = freeze_harness(archive, output / "runtime")
    shutil.copyfile(archive, output / "candidate.zip")
    registry = [{"id": f"{scenario}_{repeat}", "scenario": scenario, "repeat": repeat, "status": "not_started"}
        for scenario in SCENARIOS for repeat in range(3)]
    watchdog_seconds = len(registry) * (seconds + 30) + 120
    manifest = {"archive_sha256": archive_sha, "comparison_manifest_sha256": file_hash(comparison / "MANIFEST.json"),
        "quality_report_sha256": file_hash(comparison / "OUTER_EVALUATION.json"), "runtime_files": runtime_hashes,
        "supervisor_sha256": file_hash(Path(__file__)), "protocol_version": PROTOCOL_VERSION,
        "planned": [{key: value for key, value in row.items() if key != "status"} for row in registry],
        "seconds_per_case": seconds, "external_watchdog_seconds": watchdog_seconds, "cpu": 1, "memory_gib": 5,
        "generation": generation, "generation_overrides_provided": False, "technical_failure_rate_allowed": 0}
    atomic_json(output / "MANIFEST.json", manifest)
    atomic_json(output / "REGISTRY.json", registry)
    if api_limits("before", output) != 0:
        atomic_json(output / "NOT_STARTED.json", {"reason": "API check unavailable, blocked or exhausted", "planned": len(registry)})
        return 3
    environment = os.environ.copy()
    environment["OPENAI_BASE_URL"] = "http://192.168.97.2:18765/v1"
    # The command forwards names only; the stock solution receives no generation override.
    name = "alpha-live-" + re.sub(r"[^a-z0-9-]", "-", output.name.lower())[:35]
    if docker("inspect", name, check=False).returncode == 0:
        api_limits("after", output)
        raise RuntimeError("The owned container name is already present; no existing process was touched")
    started, launched, watchdog, state, launch_code = time.time(), False, False, None, None
    try:
        launch = subprocess.run(container_command(name, output, seconds), env=environment, capture_output=True, text=True)
        launch_code = launch.returncode
        launched = launch_code == 0
        if not launched:
            (output / "launch_error.log").write_text(launch.stderr)
        while launched:
            state = json.loads(docker("inspect", "--format", "{{json .State}}", name).stdout)
            if not state["Running"]:
                break
            if time.time() - started >= watchdog_seconds:
                watchdog = True
                break
            time.sleep(2)
    finally:
        if launched:
            if not state or state["Running"]:
                docker("stop", "-t", "2", name, check=False)
            docker("wait", name, check=False)
            inspected = docker("inspect", "--format", "{{json .State}}", name, check=False)
            state = json.loads(inspected.stdout) if inspected.returncode == 0 else None
            atomic_json(output / "docker_state.json", state)
            logs = docker("logs", name, check=False)
            (output / "container.log").write_text(logs.stdout + logs.stderr)
            docker("rm", name, check=False)
        after_status = api_limits("after", output)
        result_path = output / "cases/REPORT.json"
        result = read_json(result_path) if result_path.is_file() else {}
        actual_registry = output / "cases/REGISTRY.json"
        if actual_registry.is_file():
            outcomes = {row["id"]: row for row in read_json(actual_registry)}
            for row in registry:
                row.update(outcomes.get(row["id"], {}))
        for row in registry:
            if row["status"] in {"running", "not_started"}:
                row.update(status="cancelled", reason="The owned suite ended without a confirmed terminal case outcome; no blind replay")
        atomic_json(output / "REGISTRY.json", registry)
        technical_pass = bool(state and state.get("ExitCode") == 0 and not watchdog and result.get("all_passed") is True
            and result.get("planned") == result.get("passed") == result.get("total") == len(registry) == 33
            and all(row["status"] == "completed" for row in registry))
        atomic_json(output / "REPORT.json", {"archive_sha256": archive_sha, "manifest_sha256": file_hash(output / "MANIFEST.json"),
            "planned": 33, "total": len(registry), "passed": sum(row["status"] == "completed" for row in registry),
            "all_passed": technical_pass and after_status in {0, 3}, "technical_all_passed": technical_pass,
            "suite_report_sha256": file_hash(result_path) if result_path.is_file() else None,
            "elapsed_seconds": time.time() - started, "launch_exit_code": launch_code, "external_watchdog_triggered": watchdog,
            "api_after_check_exit_code": after_status, "api_after_confirmed": after_status in {0, 3}})
    return 0 if technical_pass and after_status in {0, 3} else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=90)
    args = parser.parse_args()
    raise SystemExit(run(args.archive.resolve(), args.comparison.resolve(), args.output.resolve(), args.seconds))
