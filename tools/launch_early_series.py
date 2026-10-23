"""Run the two approved live development tasks with isolated Linux limits."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
parser = argparse.ArgumentParser()
parser.add_argument("--series", type=Path, default=Path("outputs/v010_implementation/early_qwen01"))
parser.add_argument("--runtime", type=Path, default=Path("outputs/v010_implementation/research_snapshot06"))
parser.add_argument("--kinds", nargs="+", choices=["regression", "classification"], default=["regression", "classification"])
args = parser.parse_args()
OUT = ROOT / args.series
before = json.loads((OUT / "api_limits_before.json").read_text())
if before.get("budget_exhausted") or before.get("limits", {}).get("blocked"):
    raise ValueError("Organizer API is exhausted or blocked")
runtime = ROOT / args.runtime
environment = os.environ.copy()
environment["OPENAI_BASE_URL"] = "http://192.168.97.2:18765/v1"
jobs = []
registry = []
for kind, source, assay in (("regression", "Регрессия", "early_regression03"),
        ("classification", "Многоклассовая классификация", "early_classification01")):
    if kind not in args.kinds:
        continue
    name = "alpha-v010-" + OUT.name.replace("_", "-") + "-" + kind
    command = ["docker", "run", "--rm", "--name", name, "--network", "alpha-v010-llm", "--cpus", "6", "--memory", "5g",
        "-v", str(runtime) + ":/runtime:ro", "-v", str(ROOT / "data_for_participants" / source) + ":/source:ro",
        "-v", str(ROOT / "outputs/v010_implementation" / assay / "REPORT.json") + ":/assay.json:ro",
        "-v", str(OUT) + ":/results", "-w", "/runtime", "--env", "PYTHONDONTWRITEBYTECODE=1"]
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_EXTRA_BODY"):
        if environment.get(key):
            command.extend(["--env", key])
    command.extend(["qwen-ml-agent:exact", "python", "tools/run_early_qwen.py", "--source", "/source",
        "--output", "/results/" + kind, "--assay", "/assay.json", "--seconds", "1200"])
    log = (OUT / (kind + ".log")).open("wb")
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    jobs.append((kind, name, process, log, time.monotonic()))
    registry.append({"kind": kind, "container": name, "status": "running", "started_at": time.time(),
        "internal_seconds": 1200, "external_seconds": 1500, "cpu": 6, "memory_gib": 5})
    (OUT / "REGISTRY.json").write_text(json.dumps(registry, indent=2) + "\n")
while jobs:
    for job in jobs[:]:
        kind, name, process, log, start = job
        if process.poll() is None and time.monotonic() - start > 1500:
            subprocess.run(["docker", "stop", "-t", "2", name], check=False, stdout=subprocess.DEVNULL)
        if process.poll() is not None:
            row = next(r for r in registry if r["kind"] == kind)
            row.update(status="completed" if process.returncode == 0 else "failed", exit_code=process.returncode,
                elapsed_seconds=time.monotonic() - start)
            report_path = OUT / kind / "REPORT.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text())
                row.update(controller_state=report.get("state"), control_status=report.get("control_status"), refit_status=report.get("refit_status"))
                if report.get("state") != "completed" or report.get("control_status") != "completed" or report.get("refit_status") != "completed":
                    row["status"] = "failed"
            log.close()
            jobs.remove(job)
            (OUT / "REGISTRY.json").write_text(json.dumps(registry, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    if jobs:
        time.sleep(2)
subprocess.run([sys.executable, str(ROOT / "skills/alpha-api-limits/scripts/check_limits.py"), "after",
    "--before", str(OUT / "api_limits_before.json"), "--output", str(OUT / "api_limits_after.json")], check=False)
sys.exit(0 if all(r["status"] == "completed" for r in registry) else 1)
