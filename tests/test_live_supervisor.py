import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from automl_agent.util import digest
from tools import run_live_acceptance as supervisor


def test_live_container_forwards_only_standard_connection_names(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor.os, "environ", {"OPENAI_API_KEY": "offline_key",
        "OPENAI_BASE_URL": "http://offline.invalid/v1", "OPENAI_MODEL": "fixture_model",
        "OPENAI_EXTRA_BODY": '{"temperature":0.2}', "PYTHONPATH": "/development"})
    command = supervisor.container_command("offline_container", tmp_path, 90)
    forwarded = [command[i + 1] for i, value in enumerate(command) if value == "--env"]
    assert forwarded == ["PYTHONDONTWRITEBYTECODE=1", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]
    assert "offline_key" not in command
    assert "/runtime:ro" in " ".join(command)
    assert command[-2:] == ["--repeats", "3"] and "--scenario" not in command


def test_live_harness_is_bound_to_embedded_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "tools").mkdir(parents=True)
    for name in supervisor.HARNESS_FILES:
        (source / "tools" / name).write_text("# offline fixture\n")
    embedded = {"solution.py": b"# archive source\n", "automl_agent/__init__.py": b"",
        "knowledge_runtime/knowledge.lock.json": b"{}"}
    source_hash = digest({name: hashlib.sha256(body).hexdigest() for name, body in embedded.items()
        if name == "solution.py" or name.startswith("automl_agent/")})
    monkeypatch.setattr(supervisor, "ROOT", source)
    monkeypatch.setattr(supervisor, "archived_runtime", lambda path: embedded)
    monkeypatch.setattr(supervisor, "code_hash", lambda path: source_hash)
    hashes = supervisor.freeze_harness(tmp_path / "archive.zip", tmp_path / "runtime")
    assert hashes["solution.py"] == hashlib.sha256(embedded["solution.py"]).hexdigest()
    assert all("tools/" + name in hashes for name in supervisor.HARNESS_FILES)
    monkeypatch.setattr(supervisor, "code_hash", lambda path: "different_source")
    with pytest.raises(ValueError, match="match the measured archive"):
        supervisor.freeze_harness(tmp_path / "archive.zip", tmp_path / "other_runtime")


def test_interrupt_stops_owned_suite_before_api_after_and_keeps_all_33(tmp_path, monkeypatch):
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"Offline interruption fixture; never executed")
    output = tmp_path / "results"
    monkeypatch.setenv("OPENAI_API_KEY", "offline_fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://offline.invalid/v1")
    monkeypatch.setattr(supervisor, "require_quality", lambda *args: "fixture_archive_sha")
    monkeypatch.setattr(supervisor, "require_generation_alignment", lambda *args: {})
    monkeypatch.setattr(supervisor, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(supervisor, "freeze_harness", lambda *args: {})
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    for name in ("MANIFEST.json", "OUTER_EVALUATION.json"):
        (comparison / name).write_text("{}")
    events = []
    def limits(phase, directory):
        events.append("api_" + phase)
        return 0
    def docker(*args, **kwargs):
        events.append(args[0])
        if args[0] == "inspect" and "--format" not in args:
            return subprocess.CompletedProcess(args, 1, "", "")
        if args[0] == "inspect":
            running = "stop" not in events
            return subprocess.CompletedProcess(args, 0, json.dumps({"Running": running, "ExitCode": 137 if not running else 0}), "")
        return subprocess.CompletedProcess(args, 0, "", "")
    def interrupt(seconds):
        raise KeyboardInterrupt("Declared offline supervisor interruption")
    monkeypatch.setattr(supervisor, "api_limits", limits)
    monkeypatch.setattr(supervisor, "docker", docker)
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "fixture_container_id", ""))
    monkeypatch.setattr(supervisor.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        supervisor.run(archive, comparison, output)
    assert events.index("stop") < events.index("wait") < events.index("api_after")
    registry = json.loads((output / "REGISTRY.json").read_text())
    report = json.loads((output / "REPORT.json").read_text())
    assert len(registry) == report["planned"] == report["total"] == 33
    assert all(row["status"] == "cancelled" for row in registry)
    assert report["passed"] == 0 and not report["all_passed"] and report["api_after_confirmed"]
