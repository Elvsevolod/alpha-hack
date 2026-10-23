"""Exact flat-archive research-cycle check; stdlib, synthetic tasks, local mock only."""
import argparse
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
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
    for name, start, end in [("train.csv", 0, 240), ("test.csv", 240, 264)]:
        with (root / name).open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["identifier", "signal", "category"] + (["outcome"] if start == 0 else []))
            for i in range(start, end):
                signal = math.sin(i / 7)
                target = ["low", "medium", "high"][int((signal + 1) * 1.49)] if classification else 3 * signal + .1 * math.cos(i)
                writer.writerow([f"{i:05d}", signal, "even" if i % 2 == 0 else "odd"] + ([target] if start == 0 else []))
    with (root / "sample_submission.csv").open("w") as stream:
        writer = csv.writer(stream); writer.writerow(["identifier", "outcome"])
        writer.writerows((f"{i:05d}", "low" if classification else 0) for i in range(240, 264))
    (root / "task.md").write_text("Synthetic technical fixture: predict outcome from signal and category; "
        + ("multiclass classification, primary metric Macro F1; output labels." if classification else "regression, primary metric RMSE; output values.")
        + " Preserve identifier and sample column order. No external data.")


def run(archive, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "organizer_calls": 0,
        "command": ["python", "solution.py", "task.md"], "cases": [], "quality_test": False}
    with zipfile.ZipFile(archive) as package:
        requirements = package.read("requirements.txt").decode()
    versions = {}
    for line in requirements.splitlines():
        match = re.fullmatch(r"([\w.-]+)(?:\[[^]]+\])?==([^\s#]+)", line.strip())
        if match:
            actual = importlib.metadata.version(match[1]); assert actual == match[2], (match[1], actual, match[2])
            versions[match[1]] = actual
    report["dependency_versions"] = versions
    for label, model, classification, transport in [("unset", None, False, "native"),
        ("empty", "", True, "native"), ("blank", "   ", True, "json"),
        ("explicit", "fixture/model", False, "json")]:
        requests, errors, state = [], [], {"step": 0}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                try:
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    requests.append(body)
                    request = json.loads(body["messages"][-1]["content"])
                    if "unresolved_fields" in request:
                        task = {"contract_version": 2, "source_hash": request["source_hash"],
                            "file_roles": {name: name for name in ("train.csv", "test.csv", "sample_submission.csv")},
                            "kind": "classification" if classification else "regression", "targets": ["outcome"],
                            "id_columns": ["identifier"], "feature_columns": ["signal", "category"],
                            "categorical_columns": ["category"], "metric": {"name": "f1", "average": "macro"} if classification else {"name": "rmse"},
                            "output": {"columns": ["identifier", "outcome"], "prediction_type": "labels" if classification else "values"}}
                        content = {"request_id": request["request_id"], "source_hash": request["source_hash"],
                            "status": "proposed", "task_spec": task, "evidence": {key: ["task", "inventory"] for key in task}}
                        message = {"content": json.dumps(content)}
                    else:
                        step = state["step"]
                        related = request["core"]["current_work"]["related_ids"]
                        observation_objects = [obj for item in request.get("observations", [])
                            for obj in objects(json.loads(item["content"]))]
                        if step == 0:
                            tool, args = "read_run_records", {"ids": [next(x for x in related if x.startswith("research_guide:"))], "max_chars": 12000}
                        elif step == 1:
                            tool, args = "search_knowledge", {"collection": "documents", "strategy": "exact_symbol", "limit": 1,
                                "query": "sklearn.ensemble.ExtraTreesClassifier.fit" if classification else "sklearn.linear_model.Ridge.fit"}
                        elif step == 2:
                            card = next(obj for obj in observation_objects if obj.get("read", {}).get("tool") == "read_knowledge")
                            state["reference"] = card["reference"]
                            tool, args = "read_knowledge", {**card["read"]["arguments"], "max_chars": 12000}
                        elif step == 3:
                            assert any(obj.get("reference") == state["reference"] for obj in observation_objects)
                            tool, args = "train_model", {"pipeline": {"model": "extra_trees" if classification else "ridge",
                                "hyperparameters": {"n_estimators": 8, "max_depth": 3} if classification else {"alpha": 10}},
                                "budget_seconds": 20, "hypothesis": {"observation": "Synthetic baseline leaves measurable errors",
                                    "expected_effect": "Improve the fixed primary metric", "changed_component": "bounded model capacity",
                                    "acceptance": "Compare to incumbent under the unchanged protocol and measured cost",
                                    "evidence_refs": [state["reference"]]}}
                        elif step == 4:
                            tool, args = "read_run_records", {"ids": [next(x for x in related if x.startswith("research_review:"))], "max_chars": 12000}
                        else:
                            assert "task_local_requires_revalidation" in json.dumps(observation_objects)
                            tool, args = "finish", {"reason": "Finite synthetic research-cycle check completed; no further fixture hypotheses",
                                "result_ids": [request["core"]["current_work"]["latest_id"]],
                                "imputation_skip_reason": "Synthetic fixture has no missing features; no quality claim"}
                        state["step"] += 1
                        note = {"question": "Which observed fit error justifies the next bounded experiment?",
                            "reading_goal": "Read a pinned local API source and retain conditions",
                            "expected_evidence": "Measured primary metric and an honest conditional review",
                            "status": "closed" if step >= 4 else "awaiting_evidence"}
                        if body.get("tools"):
                            message = {"tool_calls": [{"id": "mock_" + str(step), "type": "function", "function": {
                                "name": tool, "arguments": json.dumps({"operation_id": "research_" + str(step), **args, "research_note": note})}}]}
                        else:
                            message = {"content": json.dumps({"operation_id": "research_" + str(step), "tool": tool, "arguments": args, "research_note": note})}
                    response = json.dumps({"choices": [{"finish_reason": "stop", "message": message}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 10}}).encode()
                    self.send_response(200); self.send_header("Content-Length", str(len(response))); self.end_headers(); self.wfile.write(response)
                except Exception as error:
                    errors.append(repr(error)); self.send_error(500, str(error))
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="research-archive-") as temporary:
                root = Path(temporary)
                with zipfile.ZipFile(archive) as package:
                    package.extractall(root)
                fixture(root, classification)
                env = {"PATH": os.defpath, "LANG": "C.UTF-8", "OPENAI_API_KEY": "offline-fixture",
                    "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1"}
                if model is not None:
                    env["OPENAI_MODEL"] = model
                # Explicit transport is selected only for the JSON case; no time/resource/call caps override production defaults.
                command = [sys.executable, "solution.py", "task.md"] + (["--transport", "json"] if transport == "json" else [])
                proc = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=150)
                case = output / label; case.mkdir()
                (case / "process.log").write_text(proc.stdout + proc.stderr)
                assert not errors, errors
                assert proc.returncode == 0, proc.stderr[-1500:]
                work = next((root / ".agent_work").iterdir())
                saved = json.loads((work / "run_report.json").read_text())
                assert saved["state"] == "completed" and saved["control_status"] == saved["refit_status"] == "completed"
                db = sqlite3.connect(work / "journal.sqlite")
                limits = json.loads(db.execute("SELECT value FROM metadata WHERE key='limits'").fetchone()[0]); db.close()
                assert limits["seconds"] == 10800 and limits["cpu"] == 7 and limits["max_calls"] is None and limits["max_candidates"] is None
                with (root / "submission.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                assert list(rows[0]) == ["identifier", "outcome"] and [r["identifier"] for r in rows] == [f"{i:05d}" for i in range(240, 264)]
                assert all(r["outcome"] in {"low", "medium", "high"} if classification else math.isfinite(float(r["outcome"])) for r in rows)
                reviews = [json.loads(p.read_text()) for p in (work / "research_reviews").glob("*.json")]
                assert any(r["hypothesis"]["evidence_refs"] for r in reviews)
                assert state["step"] == 6
                (case / "reviews.json").write_text(json.dumps(reviews, indent=2))
                (case / "run_report.json").write_text(json.dumps(saved, indent=2))
                result = {"case": label, "task": "multiclass" if classification else "regression", "exit_code": proc.returncode,
                    "transport": transport, "calls": len(requests), "limits": limits, "reviews": len(reviews), "passed": True}
                report["cases"].append(result); print(json.dumps(result), flush=True)
        finally:
            server.shutdown(); server.server_close(); thread.join()
    report["all_passed"] = all(r["passed"] for r in report["cases"])
    (output / "REPORT.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--archive", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); run(args.archive.resolve(), args.output.resolve())
