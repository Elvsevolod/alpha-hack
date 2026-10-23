"""Reproducible scripted scenarios. These are not evidence of live Qwen behavior."""
import json
from pathlib import Path

from context_compressor import (ContextCompressor, ContextError, ContextStore, Emission, ProfileCache,
                                ProfileKey, REQUIRED_FLAGS, Source)
from context_compressor.contracts import canonical
from context_compressor.tests.helpers import allowed, emit, note, response, seed_history, setup, task, transport, user_context


def four_attempt_fixture(store):
    """A supported wide core plus real bounded records that require all four stages."""
    full_task = task(wide=True)
    setup(store, task=full_task)
    compressor = ContextCompressor(store, transport(real=True))
    first = compressor.prepare(compressor.snapshot(), "json")
    large_note = {"question": "q" * 512, "reading_goal": "r" * 512, "expected_evidence": "e" * 512,
                  "related_ids": ["x" * 95 + str(i) for i in range(8)]}
    compressor.accept_response(first.request_id, response(note=large_note), allowed)
    calls = []

    def profile():
        calls.append(1)
        return {"rows": {"fit": 256}, "columns": {f"f{i}": {"dtype": "float64"} for i in range(10000)}}

    cached = ProfileCache(store).get_or_create(ProfileKey("d1", "f1"), profile)
    setup(store, task=full_task, profile=cached, cpu=4, memory_mib=9536,
          work={"diagnostic": "\x00" * 512},
          policy_flags={**REQUIRED_FLAGS, **{f"condition_{i}": "C" * 100 for i in range(16)}})
    emit(store, content="\x00" * 65536, kind="diagnostic", scope="code")
    seed_history(store, 40, detail_bytes=16000, record_options={"component": "c" * 128, "engine": "e" * 128,
                 "protocol": "p" * 128, "environment": "v" * 128, "checked_change": "\x00" * 300})
    renderer = transport(real=True)
    raw_system = json.loads((Path(__file__).parent / "fixtures/controller_schemas.json").read_text())["system"]
    renderer = transport(real=True, system=raw_system + "X" * (49100 - renderer.overhead_bytes("native")))
    return ContextCompressor(store, renderer), calls


def pressure_scenario(directory: Path, mode="json"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"pressure-{mode}.sqlite"
    store = ContextStore(path)
    try:
        setup(store)
        compressor = ContextCompressor(store, transport(real=True))
        initial = compressor.prepare(compressor.snapshot(), mode)
        compressor.accept_response(initial.request_id, response(mode, note=note(), operation_id="begin-research"), allowed)
        intent = store.active_note()
        deadline = store.get_meta("controller_state")["policies_resources"]["deadline_epoch"]
        cycles = []
        for cycle in range(3):
            seed_history(store, 20, first_index=cycle * 20, detail_bytes=16000)
            setup(store, counters={"calls": cycle * 2 + 1, "candidates": 20 * (cycle + 1), "stagnation": cycle},
                  work={"baseline_id": "experiment-1", "leader_id": "experiment-2", "parent_id": "experiment-1"})
            oid = emit(store, f"evidence-{cycle}", f"Permitted evidence {cycle}: comparison uses the same fit protocol.")
            snapshot = compressor.snapshot()
            prepared = compressor.prepare(snapshot, mode)
            before_wire_bytes = prepared.wire_bytes
            # Compact summaries now fit locally; emulate a stricter provider
            # rejecting that exact request, then retry the immutable snapshot.
            store.channel_error(prepared.request_id, "provider_context_limit", "Injected provider limit")
            prepared = compressor.prepare(snapshot, mode, minimum_stage=2)
            original_observations = user_context(prepared)["observations"]
            if cycle == 1:
                store.channel_error(prepared.request_id, "timeout", "Injected channel failure")
                try:
                    compressor.accept_response(prepared.request_id, response(mode, finish_reason="length"), allowed)
                    raise AssertionError("A truncated response was accepted")
                except ContextError:
                    pass
                store.close()
                store = ContextStore(path)
                compressor = ContextCompressor(store, transport(real=True))
                prepared = compressor.prepare(snapshot, mode, minimum_stage=2)
                assert user_context(prepared)["observations"] == original_observations
            assert prepared.stage > 0, "Insufficient context pressure"
            body = user_context(prepared)
            assert body["core"]["research_note"] == intent
            assert body["core"]["policies_resources"]["deadline_epoch"] == deadline
            assert body["core"]["policies_resources"]["counters"]["candidates"] == 20 * (cycle + 1)
            assert body["observations"][0]["observation_id"] == oid
            arguments = {"experiment_ids": ["experiment-1", "experiment-2"]} if cycle == 2 else {}
            compressor.accept_response(prepared.request_id, response(mode, operation_id=f"continue-{cycle}", arguments=arguments), allowed)
            assert store.observation_status(oid)["state"] == "delivered"
            assert store.active_note() == intent
            trace = [json.loads(item) for item in prepared.fit_trace]
            assert before_wire_bytes > prepared.wire_bytes
            cycles.append({"cycle": cycle + 1, "request_id": prepared.request_id,
                           "snapshot_id": snapshot.snapshot_id, "request_sha256": prepared.request_hash,
                           "before_wire_bytes": before_wire_bytes, "after_wire_bytes": trace[-1]["wire_bytes"],
                           "attempts": prepared.attempts, "stage": prepared.stage,
                           "selected_ids": list(prepared.selected_ids), "max_record_seq": snapshot.max_record_seq,
                           "observation_id": oid, "intent_note_id": intent["note_id"]})
        # The scripted controller actually produces a confirmed comparison result
        # after the third reduction; no model-selected experiment is fabricated.
        left, right = store.record("experiment-1"), store.record("experiment-2")
        comparison = {"left_id": left["record_id"], "right_id": right["record_id"],
                      "protocol": left["protocol"], "evaluation_scope": left["evaluation_scope"],
                      "score_delta": right["score"] - left["score"], "status": "confirmed_comparison"}
        source = Source("continued-comparison", "v1", canonical(comparison), "registered_feedback",
                        "controller_fact", origin="controller_comparison")
        store.complete_operation("continue-2", "v1", comparison, [Emission(source, kind="analysis")])
        final = compressor.prepare(compressor.snapshot(), mode)
        assert json.loads(user_context(final)["observations"][0]["content"]) == comparison
        found = store.search({"component": "component-0", "engine": "engine-0", "error_code": "ERR_0", "protocol": "protocol-v1"})
        assert found["records"][0]["record_id"] == "experiment-0"
        old_diagnostic = store.handle_read(found["records"][0]["read"]["arguments"])
        assert "Report 0:" in old_diagnostic["content"]
        return {"status": "passed_scripted_offline", "transport": mode, "cycles": cycles,
                "actual_reductions_on_distinct_steps": len(cycles), "channel_failures": 1, "restarts": 1,
                "comparison_after_third": comparison, "old_failure_discovery_reads": 2,
                "original_deadline_preserved": True, "live_qwen_behavior": "not_tested"}
    finally:
        store.close()
