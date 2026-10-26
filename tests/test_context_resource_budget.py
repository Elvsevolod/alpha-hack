"""Finite controller limits must survive the stateless decision projection."""
import json
import time
from dataclasses import replace

import pytest

from automl_agent.budget import Budget, Limits
from automl_agent.controller import Controller
from context_compressor import ContextError
from context_compressor.contracts import canonical, text_hash
from conftest import write_task


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1, max_calls=6,
        max_candidates=2, max_llm_seconds=90), run_id="resource_budget")
    c.no_llm = True
    c.qwen.disabled = True
    c.prepare()
    yield c
    c.close()


@pytest.mark.parametrize("mode", ["native", "json"])
def test_last_generation_budget_survives_every_compaction_stage(controller, mode):
    c = controller
    c.journal.put("llm_calls", 5)
    c.journal.put("candidate_count", 1)
    c.journal.put("llm_seconds", 31.5)
    snapshot = c.context_memory.snapshot(c.system_prompt)
    for stage in range(4):
        prepared = c.context_memory.compressor.fit(snapshot, mode, budget=262144, minimum_stage=stage)
        body = json.loads(json.loads(prepared.wire_json)["messages"][1]["content"])
        resources = body["core"]["policies_resources"]
        budget = resources["budget"]
        assert budget["local_qwen_calls_limit"] == 6
        assert budget["qwen_calls_remaining_at_snapshot"] == 1
        assert budget["qwen_calls_remaining_after_one_generation"] == 0
        assert budget["local_candidate_limit"] == 2 and budget["candidates_remaining"] == 1
        assert budget["local_qwen_seconds_limit"] == 90 and budget["qwen_seconds_used"] == 31.5
        assert budget["critic_call_cost_if_required_and_uncached"] == 1
        assert budget["critic_disabled"] is True
        assert resources["counters"]["calls"] == 5
    assert c.budget.counter("llm_calls") == 5, "Projection must not claim a paid call"


def test_resource_budget_changes_after_consumption_and_keeps_original_deadline(controller):
    c = controller
    first = json.loads(c.context_memory.snapshot(c.system_prompt).core_json)["policies_resources"]
    c.budget.claim_call()
    c.budget.claim_candidate()
    second = json.loads(c.context_memory.snapshot(c.system_prompt).core_json)["policies_resources"]
    assert second["budget"]["qwen_calls_remaining_at_snapshot"] == 5
    assert second["budget"]["qwen_calls_remaining_after_one_generation"] == 4
    assert second["budget"]["candidates_remaining"] == 1
    assert second["deadline_epoch"] == first["deadline_epoch"] == c.budget.deadline


def test_unset_caps_are_not_misrepresented_as_zero_or_provider_limits(controller):
    c = controller
    c.budget.limits.max_calls = c.budget.limits.max_candidates = c.budget.limits.max_llm_seconds = None
    budget = json.loads(c.context_memory.snapshot(c.system_prompt).core_json)["policies_resources"]["budget"]
    for name in ("local_qwen_calls_limit", "local_candidate_limit", "local_qwen_seconds_limit",
            "qwen_calls_remaining_at_snapshot", "qwen_calls_remaining_after_one_generation", "candidates_remaining"):
        assert budget[name] is None


def test_resume_projects_saved_limits_instead_of_new_cli_allowance(controller):
    c = controller
    deadline = c.budget.deadline
    c.budget.claim_call()
    c.budget = Budget(c.journal, Limits(seconds=999, max_calls=999, max_candidates=999), time.time())
    resources = json.loads(c.context_memory.snapshot(c.system_prompt).core_json)["policies_resources"]
    assert resources["deadline_epoch"] == deadline
    assert resources["budget"]["local_qwen_calls_limit"] == 6
    assert resources["budget"]["qwen_calls_remaining_at_snapshot"] == 5
    assert resources["budget"]["local_candidate_limit"] == 2


def test_serialized_budget_cannot_be_changed_by_rehashing_the_wire(controller):
    c = controller
    snapshot = c.context_memory.snapshot(c.system_prompt)
    prepared = c.context_memory.compressor.fit(snapshot, "native", budget=262144)
    wire = json.loads(prepared.wire_json)
    context = json.loads(wire["messages"][1]["content"])
    context["core"]["policies_resources"]["budget"]["local_qwen_calls_limit"] = 999
    wire["messages"][1]["content"] = canonical(context)
    text = canonical(wire)
    changed = replace(prepared, wire_json=text, request_hash=text_hash(text),
        wire_bytes=len(text.encode()), conservative_size=len(text.encode()) + prepared.response_reserve)
    with pytest.raises(ContextError, match="Protected snapshot"):
        c.context_memory.store.register_request(changed, json.loads(snapshot.observations_json))
    assert c.context_memory.store.db.execute("select count(*) from cc_requests").fetchone()[0] == 0
    assert c.budget.counter("llm_calls") == 0
