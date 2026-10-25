"""Audit finished paired runs before interpreting their control scores."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def audit_arm(directory, job, ledger):
    result = read(directory / "RESULT.json", {})
    work = Path(job["directory"]) / ".agent_work/diagnostic"
    task = read(work / "task_spec.json", {})
    plan = read(work / "validation_plan.json", {})
    report = result.get("report", {})
    score = (report.get("control_scores") or {}).get(report.get("control_winner"))
    arm = {"started": job["started"], "technical_passed": result.get("technical_passed", False),
        "state": report.get("state"), "failure": report.get("failure"), "selected_control_score": score,
        "control": report.get("control_status"), "refit": report.get("refit_status"),
        "elapsed_seconds": result.get("elapsed_seconds"), "model_calls": report.get("model_calls"),
        "time_accounting": report.get("time_accounting"),
        "cpu": report.get("effective_cpu"), "memory_mb": report.get("effective_memory_mb"),
        "training_seed": job["seed"], "split_seed": plan.get("seed"), "strategy": plan.get("strategy"),
        "model_configuration": report.get("model_configuration"), "input_hashes": job["input_hashes"],
        "split_hashes": {key: digest(plan[key]) for key in ("fit", "validation", "control") if key in plan},
        "split_rows": {key: len(plan[key]) for key in ("fit", "validation", "control") if key in plan},
        "task_semantics": {key: task.get(key) for key in ("kind", "targets", "id_columns", "feature_columns",
            "categorical_columns", "text_columns", "time_column", "group_column", "series_columns", "horizon", "metric", "output",
            "classes", "feature_constraints", "service_columns", "file_roles")},
        "known_api_cost": sum(row["cost"] for row in ledger["requests"].values() if row.get("arm") == job["kind"])}
    if not (work / "journal.sqlite").is_file():
        return arm
    # Call only after RUN_FINISHED: never read a live guest SQLite file from macOS.
    with sqlite3.connect("file:" + str((work / "journal.sqlite").resolve()) + "?mode=ro", uri=True) as db:
        operations = [json.loads(row[0]) for row in db.execute("SELECT record FROM operations")]
        experiments = [{"spec": json.loads(spec), "result": json.loads(outcome) if outcome else None}
            for spec, outcome in db.execute("SELECT spec,result FROM experiments ORDER BY rowid")]
        rejections = [json.loads(row[0]) for row in db.execute(
            "SELECT payload FROM events WHERE kind='task_resolution_rejected' ORDER BY seq")]
    arm["task_resolution_rejections"] = [item.get("errors", []) for item in rejections]
    arm["tools"] = dict(Counter(item["tool"] for item in operations))
    arm["research_guide_reads"] = []
    arm["human_pattern_reads"] = []
    for operation in operations:
        data = operation.get("result") or {}
        data = data.get("data", data)
        if operation["tool"] == "read_run_records":
            content = json.dumps(data, ensure_ascii=False)
            if "# Qwen ML best practices" in content or "Research cycle: observed problem" in content:
                arm["research_guide_reads"].append(operation["operation_id"])
        if operation["tool"] == "read_knowledge" and (data.get("entry") or {}).get("id", "").startswith("method:human_practice_"):
            arm["human_pattern_reads"].append({"operation": operation["operation_id"],
                "id": data["entry"]["id"], "complete": data.get("complete", False)})
    arm["experiments"] = [{"id": item["spec"]["experiment_id"], "parent": item["spec"].get("parent_id"),
        "pipeline": item["spec"]["pipeline"], "hypothesis": item["spec"]["hypothesis"],
        "status": (item["result"] or {}).get("status"), "scope": (item["result"] or {}).get("scope"),
        "score": (item["result"] or {}).get("score"), "seconds": (item["result"] or {}).get("seconds")}
        for item in experiments]
    arm["evidence_limit"] = "A returned guide/card or evidence reference does not prove correct application."
    return arm


def analyze(directory):
    if not (directory / "RUN_FINISHED").is_file():
        raise ValueError("Wait for all containers and the supervisor to finish")
    manifest = read(directory / "MANIFEST.json")
    config = read(directory / "CONFIG.json")
    ledger = read(directory / "budget_ledger.json")
    arms = {name: audit_arm(directory / name, job, ledger) for name, job in config["jobs"].items()}
    for arm in arms.values():
        arm["gateway"] = {"responses": 0, "queue_seconds": 0., "upstream_seconds": 0.}
    for file in sorted((directory / "history/events").glob("*.jsonl.gz")):
        for line in gzip.decompress(file.read_bytes()).decode().splitlines():
            event = json.loads(line)
            if event["kind"] == "request_response" and event.get("task_id") in arms:
                stats = arms[event["task_id"]]["gateway"]
                stats["responses"] += 1
                for field in ("queue_seconds", "upstream_seconds"):
                    stats[field] += event["payload"][field]
    pairs = []
    for seed in manifest["seeds"]:
        for kind in ("regression", "classification"):
            a, b = [arms[f"{kind}_{seed}_{version}"] for version in ("base", "skill")]
            # Effort and its output allowance are selected per action by the
            # same frozen policy. Final actions need not have the same effort.
            def model_settings(arm):
                config = arm["model_configuration"] or {}
                return {"model": config.get("model"), "reasoning_policy": config.get("reasoning_policy"),
                    "output_by_effort": config.get("output_by_effort"), "context_tokens": config.get("context_tokens"),
                    "generation": {key: value for key, value in config.get("generation", {}).items() if key != "reasoning"}}
            checks = {"technical_passed": a["technical_passed"] and b["technical_passed"],
                "inputs": a["input_hashes"] == b["input_hashes"],
                "semantics": bool(a["task_semantics"]["kind"]) and a["task_semantics"] == b["task_semantics"],
                "splits": len(a["split_hashes"]) == 3 and a["split_hashes"] == b["split_hashes"],
                "resources": a["cpu"] == b["cpu"] and a["memory_mb"] == b["memory_mb"],
                "model_settings": model_settings(a) == model_settings(b)}
            scores = [a["selected_control_score"], b["selected_control_score"]]
            finite = all(type(score) in (int, float) and math.isfinite(score) for score in scores)
            comparable = all(checks.values()) and finite
            metric = a["task_semantics"].get("metric") or {}
            maximize = metric.get("name") not in {"rmse", "mse", "mae", "mape", "log_loss"}
            gain = (scores[1] - scores[0]) * (1 if maximize else -1) if comparable else None
            pairs.append({"task": kind, "training_seed": seed, "checks": checks, "comparable": comparable,
                "metric": metric, "base_score": scores[0], "skill_score": scores[1], "directed_gain": gain,
                "relative_gain": gain / abs(scores[0]) if comparable and scores[0] else None})
    means = {}
    for kind in ("regression", "classification"):
        values = [pair["relative_gain"] for pair in pairs if pair["task"] == kind]
        means[kind] = statistics.mean(values) if all(value is not None for value in values) else None
    return {"manifest_sha256": hashlib.sha256((directory / "MANIFEST.json").read_bytes()).hexdigest(),
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arms": arms, "pairs": pairs, "mean_relative_gain_by_task": means,
        "all_pairs_comparable": all(pair["comparable"] for pair in pairs),
        "known_api_cost": ledger["reported_cost"], "unknown_billing": ledger["inflight"] is not None,
        "gate_reason": ledger["closed"], "quality_advantage_established": False,
        "inference_limit": "Descriptive public-development-task comparison; two repeats per task do not establish general superiority."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory.resolve())
    with args.output.open("x") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("all_pairs_comparable", "known_api_cost", "unknown_billing")}))
