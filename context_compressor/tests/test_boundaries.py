from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from context_compressor import ContextCompressor, ContextError, ContextStore, Emission, Source
from context_compressor.contracts import canonical, text_hash
from context_compressor.tests.helpers import allowed, emit, response, setup, transport, user_context


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = ContextStore(Path(self.directory.name) / "state.sqlite")
        setup(self.store)
        self.compressor = ContextCompressor(self.store, transport())

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_result_version_changes_observation_identity_and_preserves_both_results(self):
        source = Source("report", "v1", "original", "reference")
        first = self.store.complete_operation("read", "r1", {"revision": 1}, [Emission(source)])[0]
        newer = replace(source, version="v2", content="revised")
        second = self.store.complete_operation("read", "r2", {"revision": 2}, [Emission(newer)])[0]
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.pending_count(), 2)
        self.assertEqual(self.store.operation_result("read", "r1"), {"revision": 1})
        self.assertEqual(self.store.operation_result("read", "r2"), {"revision": 2})

    def test_requested_subrange_has_exact_bytes_and_does_not_claim_full_document(self):
        source = Source("unicode-code", "v1", "prefix🙂текст suffix", "code", origin="code_file")
        self.store.complete_operation("read-range", "v1", {"status": "ok"}, [Emission(source, start=len(source.content[:6].encode()), end=len(source.content[:12].encode()), kind="code")])
        prepared = self.compressor.prepare(self.compressor.snapshot(), "json")
        piece = user_context(prepared)["observations"][0]
        self.assertEqual(piece["content"], source.content[6:12])
        self.assertEqual(piece["range"]["start"], len(source.content[:6].encode()))
        self.assertLess(piece["requested_end"], piece["source_bytes"])
        self.assertFalse(piece["document_complete"])
        self.assertEqual(piece["source"]["trust"], "external_data")

    def test_usage_includes_api_retries_and_is_never_inferred_from_bytes(self):
        emit(self.store)
        snapshot = self.compressor.snapshot()
        first = self.compressor.prepare(snapshot, "native")
        second = self.compressor.prepare(snapshot, "native")
        self.store.record_usage(first.request_id, prompt_tokens=100, completion_tokens=10, elapsed_seconds=1.2)
        self.store.record_usage(second.request_id, prompt_tokens=105, completion_tokens=11, elapsed_seconds=1.4)
        values = [json.loads(row[0]) for row in self.store.db.execute("SELECT usage_json FROM cc_usage")]
        self.assertEqual(sum(value["total_tokens"] for value in values), 226)
        self.assertTrue(all(value["source"] == "server_usage" for value in values))
        with self.assertRaises(ContextError):
            self.store.record_usage(first.request_id, prompt_tokens=0, completion_tokens=0, elapsed_seconds=0)
        with self.assertRaises(ContextError):
            self.store.record_usage(first.request_id, prompt_tokens=-1, completion_tokens=0, elapsed_seconds=0)
        self.assertEqual(self.store.pending_count(), 1)

    def test_empty_queue_is_valid_but_request_and_operation_ids_cannot_be_reassigned(self):
        snapshot = self.compressor.snapshot()
        prepared = self.compressor.prepare(snapshot, "json", request_id="fixed-request")
        self.assertEqual(user_context(prepared)["observations"], [])
        with self.assertRaises(ContextError):
            self.compressor.prepare(snapshot, "native", request_id="fixed-request")
        self.compressor.accept_response(prepared.request_id, response(), allowed)
        with self.assertRaises(ContextError):
            self.compressor.accept_response(prepared.request_id, response(operation_id="different"), allowed)

    def test_invalid_provider_shapes_are_controlled_errors(self):
        renderer = transport()
        cases = [None, {}, {"choices": [None]}, {"choices": [{"finish_reason": "stop", "message": None}]}]
        bad = response("native")
        bad["choices"][0]["message"]["tool_calls"] = ["bad"]
        cases.append(bad)
        bad = response("native")
        bad["choices"][0]["message"]["tool_calls"][0]["function"] = None
        cases.append(bad)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ContextError):
                renderer.parse_response(value, "native")

    def test_actual_wire_size_includes_json_escaping_and_preserves_output_reserve(self):
        emit(self.store, content="\x00" * 65536, scope="code", kind="diagnostic")
        prepared = self.compressor.prepare(self.compressor.snapshot(), "json")
        self.assertEqual(prepared.wire_bytes, len(prepared.wire_json.encode()))
        self.assertEqual(prepared.request_hash, text_hash(prepared.wire_json))
        self.assertEqual(prepared.conservative_size, prepared.wire_bytes + 32768)
        self.assertLessEqual(prepared.conservative_size + 8192, prepared.budget)
        self.assertEqual(json.loads(prepared.wire_json)["max_tokens"], 32768)
        audit = json.loads(self.store.request(prepared.request_id)["audit_json"])
        self.assertEqual(audit["wire_bytes"], prepared.wire_bytes)
        self.assertEqual(audit["source_reads"], 1)

    def test_observation_reference_cannot_claim_another_source_hash_or_trust(self):
        emit(self.store)
        snapshot = self.compressor.snapshot()
        body = json.loads(snapshot.observations_json)
        body[0]["source"]["trust"] = "controller_fact"
        forged = replace(snapshot, observations_json=canonical(body))
        with self.assertRaises(ContextError):
            self.compressor.prepare(forged, "native")

    def test_full_task_metric_parameters_and_large_positive_label_have_lossless_routes(self):
        from context_compressor.tests.helpers import task
        full = task(wide=True)
        full["metric"]["positive_label"] = "класс" * 100
        setup(self.store, task=full)
        core = json.loads(self.compressor.snapshot().core_json)
        self.assertIsInstance(core["task"]["metric"]["positive_label"], dict)
        version = core["identity"]["task_hash"]
        original = self.store.db.execute("SELECT content FROM cc_sources WHERE source_id='task' AND version=? AND section='metric'", (version,)).fetchone()[0]
        self.assertEqual(json.loads(original), full["metric"])

    def test_stale_protocol_response_does_not_acknowledge_observation(self):
        oid = emit(self.store)
        prepared = self.compressor.prepare(self.compressor.snapshot(), "json")
        setup(self.store, protocol_hash="protocol-v2")
        with self.assertRaises(ContextError):
            self.compressor.accept_response(prepared.request_id, response(), allowed)
        self.assertEqual(self.store.observation_status(oid)["state"], "included")

    def test_record_publication_failure_rolls_back_result_and_observation(self):
        from context_compressor import Record
        source = Source("atomic-result", "v1", "permitted result", "registered_feedback")
        invalid_record = Record("new-record", "missing-source", "v1")
        from context_compressor import AccessDenied
        with self.assertRaises(AccessDenied):
            self.store.complete_operation("atomic-op", "v1", {"status": "done"}, [Emission(source)], records=[invalid_record])
        self.assertIsNone(self.store.operation_result("atomic-op", "v1"))
        self.assertEqual(self.store.pending_count(), 0)

    def test_unknown_record_id_cannot_collide_with_a_known_error_code(self):
        from context_compressor import Record
        source = Source("collision", "v1", "same text", "registered_feedback")
        self.store.put_source(source)
        self.store.publish_records([Record("COLLISION", "collision", "v1", error_code=None),
                                    Record("known", "collision", "v1", error_code="COLLISION")])
        groups = self.store.failure_groups()
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(group["count"] == 1 for group in groups))
