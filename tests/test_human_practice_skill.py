"""Skill compilation, addressable retrieval and durable delivered context."""
import json
from pathlib import Path

import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import ToolCommand
from automl_agent.controller import Controller, SYSTEM
from automl_agent.human_practice import GUIDE
from automl_agent.research import RESEARCH_GUIDE
from automl_agent.tool_results import payload
from ml_knowledge.context import SelectionContext, select
from ml_knowledge.core import canonical
from ml_knowledge.normalize import normalize
from ml_knowledge.runtime import ReferenceStore
from tools.build_human_practice import compile_skill
from conftest import write_task

ROOT = Path(__file__).resolve().parents[1]


def test_compiled_skill_matches_its_maintained_sources():
    module, cards = compile_skill()
    assert (ROOT / "automl_agent/human_practice.py").read_text() == module
    for identifier, expected in cards.items():
        assert json.loads((ROOT / "ml_knowledge/catalogs/methods" / (identifier + ".json")).read_text()) == expected
    assert RESEARCH_GUIDE.endswith(GUIDE)


@pytest.mark.parametrize("section", ["validation", "representation", "sparse", "decision", "cost"])
def test_published_patterns_are_addressable_with_constraints(section):
    with ReferenceStore(ROOT / "ml_knowledge/corpus") as store:
        record = store.document("method:human_practice_" + section)
        assert record is not None
        assert record["text"] == normalize(record["payload"]["skill_text"])
        assert record["payload"]["constraints"] in record["text"]
        assert record["warnings"] == [record["payload"]["constraints"]]
        assert record["payload"]["runtime_status"] == "unverified"
        assert record["payload"]["research_sources"]


def test_related_patterns_do_not_evict_the_routed_card_or_break_packet_bounds():
    context = SelectionContext("regression", {"rows": 1000, "columns": 5, "time": True},
        "temporal", "ridge", {"cpu": 1, "memory_mb": 2048}, metric="rmse")
    with ReferenceStore(ROOT / "ml_knowledge/corpus") as store:
        packet = select(store, context)
        assert len(canonical(packet)) <= 6000 and len(packet["materials"]) <= 3
        method = packet["materials"][0]
        assert method["id"].startswith("method:hard_case_ml")
        assert "method:human_practice_validation" in method["related_practice_ids"]
        assert method["warnings"]


@pytest.mark.parametrize("mode", ["native", "json"])
def test_guide_and_read_pattern_survive_context_compression_and_restart(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fixture.invalid/v1")
    task = write_task(tmp_path)
    c = Controller(task, Limits(seconds=180, cpu=1), run_id="human_skill")
    try:
        c.prepare()
        c.context_memory.snapshot(SYSTEM)
        guide_id = c.context_memory.research_guide_id
        guide = payload(c.dispatch(ToolCommand(operation_id="read_human_guide", tool="read_run_records",
            arguments={"ids": [guide_id], "max_chars": 12000})))
        assert GUIDE in json.dumps(guide, ensure_ascii=False).replace("\\n", "\n")
        # Each real model turn follows one tool operation. Inspect the guide's
        # delivery before queuing another large observation ahead of it.
        guide_snapshot = c.context_memory.snapshot(SYSTEM)
        for stage in (0, 3):
            prepared = c.context_memory.compressor.prepare(guide_snapshot, mode, budget=262144, minimum_stage=stage)
            assert "Use human ML cases to choose a useful next experiment" in prepared.wire_json
        result = payload(c.dispatch(ToolCommand(operation_id="read_human_pattern", tool="read_knowledge",
            arguments={"collection": "methods", "id": "method:human_practice_decision",
                "corpus_version": c.references.version, "max_chars": 12000})))
        assert result["complete"] and result["entry"]["warnings"]
        snapshot = c.context_memory.snapshot(SYSTEM)
        for stage in (0, 3):
            prepared = c.context_memory.compressor.prepare(snapshot, mode, budget=262144, minimum_stage=stage)
            wire = json.loads(prepared.wire_json)
            assert c.knowledge.audit(prepared, wire)["send_allowed"]
            decoded = json.loads(wire["messages"][1]["content"])
            observations = json.dumps(decoded, ensure_ascii=False)
            assert "human_practice_decision" in observations
            assert "tuning and evaluating a decision rule on the same labels inflates evidence" in observations.lower()
        assert c.journal.operation("read_human_pattern").status == "succeeded"
    finally:
        c.close()
    resumed = Controller(task, Limits(seconds=180, cpu=1), run_id="human_skill")
    try:
        resumed.prepare()
        resumed.context_memory.snapshot(SYSTEM)
        assert resumed.context_memory.research_guide_id == guide_id
        assert resumed.journal.operation("read_human_pattern").status == "succeeded"
    finally:
        resumed.close()
