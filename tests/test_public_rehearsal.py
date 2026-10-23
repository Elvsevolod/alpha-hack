import json
from pathlib import Path

import pytest

from automl_agent.util import atomic_json, file_hash
from tools import run_public_rehearsal as rehearsal
from tools import validate_live_public_task as validation


def test_standard_connection_drops_research_overrides(monkeypatch):
    environment = {"PATH": "/test/bin", "OPENAI_API_KEY": "offline_fixture",
        "OPENAI_BASE_URL": "http://offline.invalid/v1", "OPENAI_MODEL": "fixture_model",
        "OPENAI_EXTRA_BODY": '{"temperature":0.2}', "OPENAI_OTHER": "excluded",
        "AGENT_CONTEXT_TOKENS": "10", "AGENT_MAX_CALLS": "1", "PYTHONPATH": "/development"}
    monkeypatch.setattr(validation.os, "environ", environment)
    actual, names = validation.connection_environment()
    assert actual == {key: environment[key] for key in
        ("PATH", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")}
    assert names == ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]
    del environment["OPENAI_API_KEY"]
    with pytest.raises(RuntimeError, match="connection variables"):
        validation.connection_environment()


def test_resource_snapshot_retains_unlimited_marker_and_cpu_counters(tmp_path, monkeypatch):
    (tmp_path / "memory.current").write_text("1234\n")
    (tmp_path / "memory.peak").write_text("4321\n")
    (tmp_path / "memory.max").write_text("max\n")
    (tmp_path / "cpu.stat").write_text("usage_usec 500\nnr_throttled 2\n")
    monkeypatch.setattr(validation, "Path", lambda path: tmp_path if path == "/sys/fs/cgroup" else Path(path))
    assert validation.cgroup_usage() == {"memory.current": 1234, "memory.peak": 4321,
        "memory.max": "max", "cpu.stat": {"usage_usec": 500, "nr_throttled": 2}}
    for path in tmp_path.iterdir():
        path.unlink()
    assert validation.cgroup_usage() == {}


@pytest.fixture
def accepted_evidence(tmp_path):
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"Offline hash-binding fixture; not a deployable archive")
    comparison, live = tmp_path / "comparison", tmp_path / "live"
    atomic_json(comparison / "MANIFEST.json", {"candidate_archive_sha256": file_hash(archive), "runs": []})
    atomic_json(comparison / "OUTER_EVALUATION.json", {"manifest_sha256": file_hash(comparison / "MANIFEST.json"),
        "local_quality_goal_achieved": True})
    atomic_json(live / "MANIFEST.json", {"archive_sha256": file_hash(archive)})
    atomic_json(live / "REPORT.json", {"archive_sha256": file_hash(archive),
        "manifest_sha256": file_hash(live / "MANIFEST.json"), "planned": 33, "total": 33, "passed": 33, "all_passed": True})
    return archive, comparison, live


def test_rehearsal_requires_exact_archive_quality_and_all_live_outcomes(accepted_evidence):
    archive, comparison, live = accepted_evidence
    assert rehearsal.require_prerequisites(archive, comparison, live) == file_hash(archive)
    report = json.loads((live / "REPORT.json").read_text())
    report.update(passed=32, all_passed=False)
    atomic_json(live / "REPORT.json", report)
    with pytest.raises(ValueError, match="all 33"):
        rehearsal.require_prerequisites(archive, comparison, live)
    archive.write_bytes(b"A different immutable candidate")
    with pytest.raises(ValueError, match="quality comparison"):
        rehearsal.require_quality(archive, comparison)


def test_rehearsal_rejects_changed_manifests(accepted_evidence):
    archive, comparison, live = accepted_evidence
    atomic_json(live / "MANIFEST.json", {"changed_after_acceptance": True})
    with pytest.raises(ValueError, match="all 33"):
        rehearsal.require_prerequisites(archive, comparison, live)
    atomic_json(comparison / "MANIFEST.json", {"candidate_archive_sha256": file_hash(archive), "changed_after_scoring": True})
    with pytest.raises(ValueError, match="quality comparison"):
        rehearsal.require_quality(archive, comparison)


def test_stock_generation_is_measured_without_request_and_restores_override(monkeypatch):
    from automl_agent.qwen import QwenClient, GENERATION
    def forbidden(*args, **kwargs):
        pytest.fail("The generation audit must never make a model request")
    monkeypatch.setattr(QwenClient, "_request", forbidden)
    override = '{"temperature":0.2,"reasoning":{"enabled":true}}'
    monkeypatch.setenv("OPENAI_EXTRA_BODY", override)
    assert rehearsal.stock_generation_profile() == {**GENERATION, "reasoning": {"enabled": True, "effort": "medium"}}
    assert rehearsal.os.environ["OPENAI_EXTRA_BODY"] == override


def test_generation_gate_requires_every_agent_receipt(tmp_path, monkeypatch):
    atomic_json(tmp_path / "MANIFEST.json", {"runs": [{"id": "one", "arm": "agent"},
        {"id": "two", "arm": "agent"}, {"id": "fixed", "arm": "reference"}]})
    for identifier in ("one", "two"):
        atomic_json(tmp_path / "receipts" / (identifier + ".json"), {"report": {"model_configuration": {"generation": {}}}})
    monkeypatch.setattr(rehearsal, "stock_generation_profile", lambda: {})
    assert rehearsal.require_generation_alignment(tmp_path) == {}
    atomic_json(tmp_path / "receipts/two.json", {"report": {"model_configuration": {"generation": {"temperature": 0.2}}}})
    with pytest.raises(ValueError, match="differ from the stock"):
        rehearsal.require_generation_alignment(tmp_path)
    atomic_json(tmp_path / "receipts/two.json", {"report": {}})
    with pytest.raises(ValueError, match="metadata is missing"):
        rehearsal.require_generation_alignment(tmp_path)
