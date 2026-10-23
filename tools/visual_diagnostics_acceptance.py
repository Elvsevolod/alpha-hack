"""Exact flat ZIP, synthetic tasks, local SSE endpoint; no organizer API calls."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import zipfile


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def fixture(root, classification):
    for name, start, end in (("train.csv", 0, 600), ("test.csv", 600, 632)):
        with (root / name).open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["identifier", "signal", "category"] + (["outcome"] if start == 0 else []))
            for i in range(start, end):
                x = ((i * 17) % 101 - 50) / 10
                writer.writerow([f"{i:05d}", x, "A" if x > 0 else "B"] +
                    (["positive" if x > 0 else "negative"] if classification and start == 0 else [2 * x] if start == 0 else []))
    with (root / "sample_submission.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identifier", "outcome"])
        writer.writerows((f"{i:05d}", 0) for i in range(600, 632))
    (root / "task.md").write_text("Synthetic technical classification fixture. Maximize Macro F1.\n" if classification else
                                 "Synthetic technical regression fixture. Minimize RMSE.\n")


def run(archive, output):
    output.mkdir(parents=True, exist_ok=False)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    report = {"archive_sha256": archive_hash, "organizer_calls": 0, "quality_test": False, "cases": []}
    with zipfile.ZipFile(archive) as package:
        versions = {}
        for line in package.read("requirements.txt").decode().splitlines():
            match = re.fullmatch(r"([\w.-]+)(?:\[[^]]+\])?==([^\s#]+)", line.strip())
            if match:
                versions[match[1]] = importlib.metadata.version(match[1])
                assert versions[match[1]] == match[2], (match[1], versions[match[1]], match[2])
    report["dependency_versions"] = versions
    for label, model, classification, mode in (("unset", None, False, "native"),
        ("empty", "", True, "native"), ("blank", "   ", True, "json"), ("explicit", "fixture/model", False, "json")):
        case = output / label
        case.mkdir()
        state, received, errors = {"step": 0, "images": []}, [], []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                try:
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    content = body["messages"][-1]["content"]
                    request = json.loads(content[0]["text"] if isinstance(content, list) else content)
                    row = {"model": body["model"], "native": bool(body.get("tools")), "image": isinstance(content, list)}
                    if isinstance(content, list):
                        png = base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1], validate=True)
                        assert png.startswith(b"\x89PNG\r\n\x1a\n")
                        sha = hashlib.sha256(png).hexdigest()
                        assert request["visual_diagnostics"]["image_included"]
                        assert request["visual_diagnostics"]["artifacts"][0]["sha256"] == sha
                        state["images"].append(sha)
                        row["image_sha256"] = sha
                    received.append(row)
                    if "unresolved_fields" in request:
                        task = {"contract_version": 2, "source_hash": request["source_hash"],
                            "file_roles": {name: name for name in ("train.csv", "test.csv", "sample_submission.csv")},
                            "kind": "classification" if classification else "regression", "targets": ["outcome"],
                            "id_columns": ["identifier"], "feature_columns": ["signal", "category"],
                            "categorical_columns": ["category"], "metric": {"name": "f1", "average": "macro"} if classification else {"name": "rmse"},
                            "output": {"columns": ["identifier", "outcome"], "prediction_type": "labels" if classification else "values"}}
                        value = {"request_id": request["request_id"], "source_hash": request["source_hash"], "status": "proposed",
                            "task_spec": task, "evidence": {key: ["task", "inventory"] for key in task}}
                        message, finish = {"content": json.dumps(value)}, "stop"
                    elif "review_id" in request:
                        state["critic_calls"] = state.get("critic_calls", 0) + 1
                        assert state["step"] == 2 and len(state["images"]) == 1
                        assert isinstance(content, str) and not body.get("tools")
                        assert request["candidate"]["hypothesis"]["evidence_refs"] == [state["diagnostic_evidence"]]
                        assert request["validation"]["comparison_scope"] == "research_validation_only"
                        comparator = next((item["experiment_id"] for item in request["comparators"]
                            if item["status"] == "completed" and item["scope"] != "pilot" and item["score"] is not None), None)
                        value = {"review_id": request["review_id"], "verdict": "approve", "revision_requests": [],
                            "evaluation_scope": "research_validation", "comparator_id": comparator,
                            **{name: "Bounded synthetic compatibility check." for name in ("evidence_assessment", "comparison_assessment",
                                "leakage_assessment", "cost_assessment", "falsification_assessment")}}
                        message, finish = {"content": json.dumps(value)}, "stop"
                    else:
                        observed = []
                        for item in request.get("observations", []):
                            try:
                                observed.extend(objects(json.loads(item["content"])))
                            except ValueError:
                                pass  # An explicitly partial source is never treated as complete evidence.
                        step = state["step"]
                        if step == 0:
                            tool, args = "get_diagnostic_report", {"kind": "distributions", "columns": ["signal"], "budget_seconds": 30}
                        elif step == 1:
                            result = next(obj for obj in observed if obj.get("tool") == "get_diagnostic_report" and obj.get("status") == "success")
                            assert result["data"]["summary"]["kind"] == "distributions" and result["data"]["artifacts"]
                            state["diagnostic_evidence"] = result["evidence_ref"]
                            tool, args = "train_model", {"pipeline": {"model": "extra_trees" if classification else "ridge",
                                "hyperparameters": {"n_estimators": 8, "max_depth": 4} if classification else {"alpha": 10}},
                                "budget_seconds": 120, "hypothesis": {"observation": "Inspect the published diagnostic distribution.",
                                    "expected_effect": "Compare a fixed supported estimator", "changed_component": "bounded estimator",
                                    "acceptance": "Valid completed model on the unchanged protocol", "evidence_refs": [result["evidence_ref"]]}}
                        elif step == 2:
                            result = request["core"]["current_work"]["latest_result"]
                            assert result["status"] == "completed" and result["evaluation_scope"] != "pilot"
                            state["experiment"] = result["record_id"]
                            tool, args = "get_diagnostic_report", {"kind": "confusion_matrix" if classification else "residuals",
                                "experiment_id": state["experiment"], "budget_seconds": 60}
                        elif step == 3:
                            result = next(obj for obj in observed if obj.get("tool") == "get_diagnostic_report" and obj.get("data", {}).get("summary", {}).get("kind") != "distributions")
                            assert result["status"] == "success" and result["data"]["report_status"] == "available" and result["data"]["artifacts"]
                            assert len(state["images"]) == 2
                            tool, args = "finish", {"reason": "Exact-archive diagnostic delivery completed", "result_ids": [state["experiment"]],
                                "imputation_skip_reason": "Synthetic technical fixture; no numeric missingness"}
                        else:
                            raise AssertionError("Unexpected additional research call")
                        state["step"] += 1
                        operation = "visual_step_" + str(step)
                        if body.get("tools"):
                            message, finish = {"tool_calls": [{"index": 0, "id": "call_" + operation, "type": "function",
                                "function": {"name": tool, "arguments": json.dumps({"operation_id": operation, **args})}}]}, "tool_calls"
                        else:
                            message, finish = {"content": json.dumps({"operation_id": operation, "tool": tool, "arguments": args})}, "stop"
                    # Exercise the production SSE parser for semantic and research calls.
                    events = [{"choices": [{"index": 0, "delta": message, "finish_reason": None}]},
                              {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                               "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}}]
                    wire = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(wire.encode())
                except Exception as error:
                    errors.append(f"step={state['step']} {type(error).__name__}: {error}")
                    if not (case / "failed_context.json").exists():
                        (case / "failed_context.json").write_text(json.dumps({"step": state["step"], "request": request}, indent=2))
                    self.send_error(400, "Synthetic acceptance mismatch")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="visual-archive-") as temporary:
                root = Path(temporary)
                with zipfile.ZipFile(archive) as package:
                    assert all("/" not in name and (name.endswith(".py") or name == "requirements.txt") for name in package.namelist())
                    package.extractall(root)
                    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in package.namelist()}
                fixture(root, classification)
                env = {"PATH": os.defpath, "LANG": "C.UTF-8", "OPENAI_API_KEY": "offline-fixture",
                       "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1"}
                if model is not None:
                    env["OPENAI_MODEL"] = model
                command = [sys.executable, "solution.py", "task.md"] + (["--transport", "json"] if mode == "json" else [])
                try:
                    process = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=150)
                except subprocess.TimeoutExpired as error:
                    (case / "process.log").write_bytes((error.stdout or b"") + (error.stderr or b""))
                    raise
                (case / "process.log").write_text(process.stdout + process.stderr)
                assert process.returncode == 0 and not errors, (process.returncode, errors, process.stderr[-1000:])
                work = next((root / ".agent_work").iterdir())
                saved = json.loads((work / "run_report.json").read_text())
                assert saved["state"] == saved["control_status"] == saved["refit_status"] == "completed", saved
                db = sqlite3.connect(work / "journal.sqlite")
                limits = json.loads(db.execute("SELECT value FROM metadata WHERE key='limits'").fetchone()[0])
                db.close()
                assert limits["seconds"] == 10800 and limits["cpu"] == 7 and limits["memory_mb"] == 28000
                assert limits["max_calls"] is limits["max_candidates"] is None
                assert len(state["images"]) == 2 and state["step"] == 4
                assert state.get("critic_calls") == 1
                assert all(row["model"] == ((model or "").strip() or "qwen/qwen3.8-27b") for row in received)
                assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == sha for name, sha in hashes.items())
                with (root / "submission.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                assert list(rows[0]) == ["identifier", "outcome"] and [row["identifier"] for row in rows] == [f"{i:05d}" for i in range(600, 632)]
                assert all(row["outcome"] in {"negative", "positive"} if classification else math.isfinite(float(row["outcome"])) for row in rows)
                for index, path in enumerate(sorted((work / "diagnostics").glob("*/diagnostic.png"))):
                    sha = hashlib.sha256(path.read_bytes()).hexdigest()
                    assert sha in state["images"]
                    shutil.copyfile(path, case / f"diagnostic_{index}.png")
                    shutil.copyfile(path.with_name("diagnostic_report.json"), case / f"diagnostic_{index}.json")
                (case / "run_report.json").write_text(json.dumps(saved, indent=2))
                (case / "requests_summary.json").write_text(json.dumps(received, indent=2))
                result = {"case": label, "transport": mode, "task": "classification" if classification else "regression",
                    "passed": True, "images_delivered": 2, "critic_calls": state["critic_calls"],
                    "calls": len(received), "control": saved["control_status"],
                    "refit": saved["refit_status"], "limits": limits}
                report["cases"].append(result)
                print(json.dumps(result), flush=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            (case / "errors.json").write_text(json.dumps(errors, indent=2))
            (output / "REPORT.json").write_text(json.dumps(report, indent=2))
    report["all_passed"] = len(report["cases"]) == 4 and all(row["passed"] for row in report["cases"])
    (output / "REPORT.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.archive.resolve(), args.output.resolve())
