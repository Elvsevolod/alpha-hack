"""Three frozen agent arms on isolated development tasks; live API requires a cap."""
from __future__ import annotations

import argparse
import ast
import base64
from contextlib import ExitStack
import json
import io
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_agent.util import atomic_json, digest, file_hash, read_json
from tools.plan_development_benchmark import tasks
from tools.plan_acceptance import evaluate


def compatible_task_spec(specification, archive):
    """Drop only absent optional fields which an immutable older arm lacks."""
    with zipfile.ZipFile(archive) as outer:
        assignments = {target.id: ast.literal_eval(node.value) for node in ast.parse(outer.read("agent_bundle.py")).body
            if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
    with zipfile.ZipFile(io.BytesIO(base64.b85decode(assignments["PAYLOAD"]))) as bundle:
        tree = ast.parse(bundle.read("automl_agent/contracts.py"))
    fields = {node.name: {field.target.id for field in node.body if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)}
        for node in tree.body if isinstance(node, ast.ClassDef) and node.name in {"TaskSpec", "MetricSpec", "OutputSpec"}}
    def keep(values, schema):
        result = {}
        for key, value in values.items():
            if key in fields[schema]:
                result[key] = value
            elif value is not None and value is not False and value != [] and value != {} and value != "":
                raise ValueError("Immutable arm cannot represent this task field: " + schema + "." + key)
        return result
    result = keep(specification, "TaskSpec")
    for key, schema in (("metric", "MetricSpec"), ("output", "OutputSpec")):
        result[key] = keep(result[key], schema)
    return result


def prepare(development, output, candidate, seeds):
    protocol = read_json(development / "protocol.json")
    output.mkdir(parents=True, exist_ok=False)
    datasets = tasks()
    archives = {"v076": ROOT / "dist/solution_v076.zip", "v081": ROOT / "dist/solution_v081.zip", "candidate": candidate.resolve()}
    for version in ("v076", "v081"):
        if file_hash(archives[version]) != protocol["versions"][version]["zip_sha256"]:
            raise ValueError("Immutable baseline archive changed")
    manifest = {"versions": {version: {"archive": str(path), "sha256": file_hash(path)} for version, path in archives.items()},
        "tasks": list(datasets), "splits": protocol["splits"], "seeds": seeds,
        "resources": {"cpu": 8, "memory_mb": 32768, "seconds": 10800},
        "internal_resources": {"cpu": 7, "memory_mb": 28000}, "development_protocol_hash": protocol["protocol_hash"],
        "source_sha256": file_hash(Path(__file__)), "approval_for_qwen": False, "automatic_promotion": False,
        "evaluation_independence": "previously_viewed_development_labels", "jobs": []}
    for repeat, seed in enumerate(seeds):
        for task_index, (name, dataset) in enumerate(datasets.items()):
            order = list(archives)
            if (repeat + task_index) % 2:
                order.reverse()
            for version in order:
                identifier = f"{name}_{seed}_{version}"
                directory = output / identifier / "task"
                directory.mkdir(parents=True)
                split = protocol["splits"][name]
                fit = dataset.train.iloc[split["fit"]].copy()
                query = dataset.train.iloc[split["evaluation"]].drop(columns=dataset.task.targets).copy()
                fit.to_csv(directory / "train.csv", index=False)
                query.to_csv(directory / "test.csv", index=False)
                sample = query[dataset.task.id_columns].copy()
                for target in dataset.task.targets:
                    sample[target] = 0
                sample = sample[dataset.task.output.columns]
                sample.to_csv(directory / "sample_submission.csv", index=False)
                shutil.copyfile(dataset.paths["data_dictionary.csv"], directory / "data_dictionary.csv")
                text = dataset.text
                for original, replacement in ((len(dataset.train), len(fit)), (len(dataset.test), len(query))):
                    text = text.replace(f"{original:,}".replace(",", " "), f"{replacement:,}".replace(",", " "))
                text = "Development task. Only the files in this directory are available; the sample contains no answers.\n\n" + text
                (directory / "task.md").write_text(text)
                specification = dataset.task.model_dump(mode="json", exclude={"source_hash"})
                specification.update(contract_version=2, file_roles={role: role for role in ("train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv")})
                atomic_json(directory / "task_spec.json", compatible_task_spec(specification, archives[version]))
                with zipfile.ZipFile(archives[version]) as archive:
                    if any(Path(member).name != member or not (member.endswith(".py") or member == "requirements.txt") for member in archive.namelist()):
                        raise ValueError("Only flat verified platform payloads are accepted")
                    archive.extractall(directory)
                manifest["jobs"].append({"id": identifier, "version": version, "task": name, "seed": seed, "directory": str(directory.resolve()),
                    "input_hashes": {name: file_hash(directory / name) for name in ("task.md", "task_spec.json", "train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv")},
                    "external_targets_in_task_directory": False, "started": False, "finished": False})
    manifest["protocol_hash"] = digest(manifest)
    atomic_json(output / "manifest.json", manifest)
    return manifest


def run(output, cap):
    if cap is None or not 20 < cap < 100000:
        raise ValueError("Live Qwen comparison requires an explicitly authorized finite total API budget above the admission reserve")
    manifest = read_json(output / "manifest.json")
    if digest({key: value for key, value in manifest.items() if key != "protocol_hash"}) != manifest["protocol_hash"]:
        raise ValueError("Frozen agent protocol changed")
    if manifest["source_sha256"] != file_hash(Path(__file__)):
        raise ValueError("Comparison implementation changed after freeze")
    from tools import diagnostic_3h as d
    from tools.goal_quality_run import SharedPeriodGuard, decode_response
    from tools.visual_ab_benchmark import Allowance, Handler, response_deadline
    from dotenv import load_dotenv
    info = json.loads(d.docker("info", "--format", "{{json .}}").stdout)
    if info["NCPU"] < 8 or info["MemTotal"] < 33 * d.GIB:
        raise ValueError("The frozen competition resource profile requires 8 available CPU and 33 GiB Docker memory")
    network = "alpha-internal"
    if d.docker("network", "inspect", network, "--format", "{{.Internal}}").stdout.strip() != "true":
        raise ValueError("An internal Docker network is required")
    datasets = tasks()
    for version, source in manifest["versions"].items():
        if file_hash(Path(source["archive"])) != source["sha256"]:
            raise ValueError("Frozen archive changed")
    load_dotenv(ROOT / ".env", override=False)
    key = d.limits.load_key(ROOT / ".env")
    d.register_secret(key)
    before = d.limits.snapshot(d.limits.fetch_info(key), key, "before")
    atomic_json(output / "api_limits_before.json", before)
    jobs, results = {}, []
    server = None
    relay = "alpha-plan-relay-" + secrets.token_hex(4)
    try:
        if before["budget_exhausted"] or before["limits"].get("blocked") or before["remaining_budget"] is None or before["remaining_budget"] < cap:
            raise ValueError("Current known unblocked account remainder does not cover the authorized series cap")
        with ExitStack() as cleanup:
            allowance = Allowance(output / "api_ledger.json", before, cap)
            allowance.cap = allowance.state["initial_spend"] + cap
            allowance.arm_limit = (cap - allowance.reserve) / len(manifest["jobs"])
            cleanup.callback(allowance.release)
            fetch = lambda: d.limits.snapshot(d.limits.fetch_info(key), key, "status")
            guard = SharedPeriodGuard(before, fetch, output, budget_cap=cap, accounting=allowance)
            guard.budget, guard.threshold = cap, cap - allowance.reserve
            token = secrets.token_urlsafe(32)
            d.register_secret(token)
            for index, job in enumerate(manifest["jobs"]):
                jobs[job["id"]] = {**job, "kind": job["id"], "task_kind": job["task"], "archive_sha256": manifest["versions"][job["version"]]["sha256"],
                    "container": "alpha-plan-" + output.name[-8:].replace("_", "-") + "-" + str(index), "deadline_mono": None}
            server = d.Gateway(("0.0.0.0", 0), guard, jobs, os.environ["OPENAI_BASE_URL"].rstrip("/"), key,
                deadline_reader=lambda body, job: response_deadline(body, job), client_key=token,
                response_decoder=decode_response, accounting=allowance)
            server.RequestHandlerClass = Handler
            threading.Thread(target=server.serve_forever, daemon=True).start()
            d.docker("create", "--name", relay, "--network", network, "--cpus", ".25", "--memory", "192m",
                "-v", str(ROOT / "tools/visual_ab_relay.py") + ":/relay.py:ro", "--env", "GATEWAY_PORT=" + str(server.server_port),
                "--env", "ALLOWED_ARMS=" + ",".join(jobs), d.IMAGE, "python", "/relay.py")
            d.docker("start", relay)
            atomic_json(output / "authorized_run.json", {"total_cap": cap, "arm_allowance": allowance.arm_limit,
                "protocol_hash": manifest["protocol_hash"], "price_bound_claimed": False})
            for job in jobs.values():
                if guard.closed:
                    break
                directory = Path(job["directory"])
                if any(file_hash(directory / name) != expected for name, expected in job["input_hashes"].items()):
                    raise ValueError("A frozen task input changed")
                env = {**os.environ, "OPENAI_API_KEY": token, "OPENAI_BASE_URL": f"http://{relay}:18767/{job['id']}/v1"}
                command = ["docker", "create", "--name", job["container"], "--network", network, "--cpus", "8", "--memory", "32g", "--memory-swap", "32g",
                    "-v", job["directory"] + ":/task", "-w", "/task", "--env", "OPENAI_API_KEY", "--env", "OPENAI_BASE_URL", d.IMAGE,
                    "python", "solution.py", "task.md", "--seed", str(job["seed"]), "--run-id", "diagnostic"]
                subprocess.run(command, env=env, capture_output=True, text=True, check=True, timeout=30)
                job.update(started=True, entry_mono=time.monotonic(), deadline_mono=time.monotonic() + 10800)
                d.docker("start", job["container"])
                while True:
                    state = json.loads(d.docker("inspect", "--format", "{{json .State}}", job["container"]).stdout)
                    if not state["Running"]:
                        break
                    guard.allowed()
                    if time.monotonic() >= job["deadline_mono"]:
                        d.docker("kill", job["container"], check=False)
                    atomic_json(output / "progress.json", {"current": job["id"], "gate_reason": guard.reason,
                        "elapsed_seconds": time.monotonic() - job["entry_mono"]})
                    time.sleep(5)
                job.update(finished=True, finished_mono=time.monotonic(), docker_state=state)
                logs = d.docker("logs", job["container"], check=False)
                (directory.parent / "container.log").write_text(logs.stdout + logs.stderr)
                row = score_job(manifest, job, datasets[job["task"]])
                results.append(row)
                atomic_json(output / "results.json", results)
                d.docker("rm", job["container"], check=False)
                print(json.dumps({"job": job["id"], "status": row["status"], "metric": row["raw_metric"]}), flush=True)
            atomic_json(output / "acceptance.json", evaluate(manifest, results))
    finally:
        for job in jobs.values():
            if job.get("started") and not job.get("finished"):
                d.docker("kill", job["container"], check=False)
                try:
                    row = score_job(manifest, {**job, "finished_mono": time.monotonic(), "docker_state": {"ExitCode": -1}}, datasets[job["task"]])
                    row["status"] = "interrupted"
                    results.append(row)
                finally:
                    atomic_json(output / "results.json", results)
            if job.get("started"):
                d.docker("rm", job["container"], check=False)
        d.docker("kill", relay, check=False); d.docker("rm", relay, check=False)
        if server:
            server.shutdown(); server.server_close()
        try:
            after = d.limits.snapshot(d.limits.fetch_info(key), key, "after")
            atomic_json(output / "api_limits_after.json", after)
        except Exception as error:
            atomic_json(output / "api_limits_after.json", {"status": "unknown", "reason": type(error).__name__})


def score_job(manifest, job, dataset):
    import numpy as np
    import pandas as pd
    from automl_agent.evaluation import score_predictions, confusion_details
    directory = Path(job["directory"])
    report = read_json(directory / ".agent_work/diagnostic/run_report.json") if (directory / ".agent_work/diagnostic/run_report.json").exists() else {}
    row = {"version": job["version"], "task": job["task"], "seed": job["seed"], "protocol_hash": manifest["protocol_hash"],
        "data_hash": manifest["splits"][job["task"]]["data_hash"], "evaluation_keys_hash": manifest["splits"][job["task"]]["evaluation_keys_hash"],
        "resources": manifest["resources"], "output_schema": dataset.task.output.model_dump(mode="json"), "evidence_type": "full_qwen_agent", "real_qwen": True,
        "status": report.get("state", "error"), "valid_csv": False, "raw_metric": None, "stress_metric": None,
        "seconds": job["finished_mono"] - job["entry_mono"], "report": report}
    path = directory / "submission.csv"
    if path.is_file():
        try:
            values = pd.read_csv(path, dtype={column: str for column in dataset.task.id_columns})
            split = manifest["splits"][job["task"]]
            actual = dataset.train.iloc[split["evaluation"]]
            if list(values) != dataset.task.output.columns or len(values) != len(actual) or not all(values[column].tolist() == actual[column].astype(str).tolist() for column in dataset.task.id_columns):
                raise ValueError("CSV schema, identifiers or row order changed")
            predictions = {target: values[target].to_numpy() for target in dataset.task.targets}
            for target, prediction in predictions.items():
                if dataset.task.classes:
                    if not np.isin(prediction, dataset.task.classes[target]).all():
                        raise ValueError("CSV contains an unconfirmed label")
                elif not np.isfinite(prediction.astype(float)).all():
                    raise ValueError("CSV contains nonfinite values")
            metric = score_predictions(dataset.task, actual[dataset.task.targets], predictions, dataset.task.output.prediction_type)
            mask = np.isin(split["evaluation"], split["stress"])
            if mask.any():
                row["stress_metric"] = score_predictions(dataset.task, actual.iloc[np.flatnonzero(mask)][dataset.task.targets],
                    {target: value[mask] for target, value in predictions.items()}, dataset.task.output.prediction_type)
            if dataset.task.classes:
                target = dataset.task.targets[0]
                row["per_class"] = confusion_details(actual[target], predictions[target], dataset.task.classes[target])
            row.update(valid_csv=True, raw_metric=metric, final_csv_sha256=file_hash(path), fallback=(report.get("publication") or {}).get("scope") == "fallback")
        except (ValueError, TypeError, KeyError, IndexError) as error:
            row.update(status="invalid_csv", csv_error=str(error), valid_csv=False, raw_metric=None)
    row["disk_bytes"] = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    atomic_json(directory.parent / "measurement.json", row)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1729, 314159])
    parser.add_argument("--authorized-api-budget", type=float)
    args = parser.parse_args()
    prepare(args.development, args.output, args.candidate, args.seeds) if args.mode == "prepare" else run(args.output, args.authorized_api_budget)
