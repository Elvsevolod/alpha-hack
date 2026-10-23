import hashlib
import json
import sys

import pytest

from automl_agent.util import atomic_json, digest
from tools import comparison_protocol
from tools import live_acceptance as live


def test_live_registry_counts_all_33_and_never_replays_a_failed_case(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline_fixture")
    monkeypatch.delenv("OPENAI_EXTRA_BODY", raising=False)
    monkeypatch.setattr(live, "load_dotenv", lambda *a, **kw: None)
    contents = {"solution.py": b"# controlled registry fixture\n",
        "knowledge_runtime/knowledge.lock.json": (live.ROOT / "knowledge_runtime/knowledge.lock.json").read_bytes()}
    monkeypatch.setattr(comparison_protocol, "archived_runtime", lambda path: contents)
    monkeypatch.setattr(comparison_protocol, "code_hash", lambda root: digest({"solution.py": hashlib.sha256(contents["solution.py"]).hexdigest()}))
    archive = tmp_path / "mock_archive.zip"
    archive.write_bytes(b"Offline bookkeeping fixture, not a deployable archive")
    before = tmp_path / "mock_before.json"
    atomic_json(before, {"phase": "before", "key_fingerprint": hashlib.sha256(b"offline_fixture").hexdigest()[:16],
        "limits": {"blocked": False}, "budget_exhausted": False})
    argv = ["live_acceptance.py", "--archive", str(archive), "--before", str(before),
        "--output", str(tmp_path / "results"), "--attempt-id", "mocked_registry_cohort"]
    calls = []
    def mock_case(scenario, repeat, output, seconds, transport):
        calls.append((scenario, repeat))
        if len(calls) == 2:
            raise RuntimeError("Declared offline registry failure")
        value = {"scenario": scenario, "repeat": repeat, "passed": True, "state": "mocked",
            "scope": "offline bookkeeping test; no model or organizer API", "protocol_version": live.PROTOCOL_VERSION,
            "code_fingerprint": live.code_fingerprint()}
        atomic_json(output / f"{scenario}_{repeat}" / "acceptance.json", value)
        return value
    monkeypatch.setattr(live, "run_case", mock_case)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exited:
        live.main()
    assert exited.value.code == 1 and len(calls) == 33
    directory = tmp_path / "results/mocked_registry_cohort"
    report = json.loads((directory / "REPORT.json").read_text())
    assert report["planned"] == report["total"] == 33 and report["passed"] == 32 and not report["all_passed"]
    registry = json.loads((directory / "REGISTRY.json").read_text())
    assert sum(row["status"] == "failed" for row in registry) == 1
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    with pytest.raises(SystemExit) as resumed:
        live.main()
    assert resumed.value.code == 1 and len(calls) == 33
    repeated = json.loads((directory / "REPORT.json").read_text())
    assert repeated["results"][1]["error"] == report["results"][1]["error"]
