from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from context_compressor import ContextCompressor, ContextError, ContextStore, Emission, Source
from context_compressor.contracts import canonical, text_hash
from context_compressor.tests.helpers import allowed, emit, note, response, setup, transport, user_context


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "journal.sqlite"
        self.store = ContextStore(self.path)
        setup(self.store)
        self.compressor = ContextCompressor(self.store, transport())

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def restart(self):
        self.store.close()
        self.store = ContextStore(self.path)
        self.compressor = ContextCompressor(self.store, transport())

    def test_api_error_keeps_exact_fragment_and_does_not_reexecute(self):
        oid = emit(self.store)
        original = self.store.operation_result("read-1", "result-v1")
        for mode in ("native", "json"):
            with self.subTest(mode=mode):
                snapshot = self.compressor.snapshot()
                first = self.compressor.prepare(snapshot, mode)
                self.assertEqual(self.store.observation_status(oid)["state"], "included")
                self.store.channel_error(first.request_id, "timeout", "Transport failed; this must not replace tool output")
                second = self.compressor.prepare(snapshot, mode)
                self.assertEqual(user_context(first)["observations"], user_context(second)["observations"])
                self.assertIn("Справка:", user_context(second)["observations"][0]["content"])
                self.assertEqual(self.store.operation_result("read-1", "result-v1"), original)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM cc_operations").fetchone()[0], 1)

    def test_restart_after_each_durable_boundary(self):
        oid = emit(self.store)
        self.restart()  # after result + outbox commit
        self.assertEqual(self.store.observation_status(oid)["state"], "pending")
        prepared = self.compressor.prepare(self.compressor.snapshot(), "json")
        self.restart()  # after actual request persistence
        self.assertEqual(self.store.observation_status(oid)["state"], "included")
        self.compressor.accept_response(prepared.request_id, response(note=note()), allowed)
        self.restart()  # after command + note + receipt commit, before execution
        self.assertEqual(self.store.observation_status(oid)["state"], "delivered")
        self.assertEqual(self.store.pending_commands()[0]["operation_id"], "step-1")
        self.assertEqual(self.store.active_note()["source"], "agent_assumption")
        emit(self.store, "step-1", "completed permitted analysis")
        self.restart()
        self.assertEqual(self.store.pending_commands(), [])

    def test_result_outbox_transaction_rolls_back_together(self):
        self.store.db.executescript("CREATE TRIGGER fail_outbox BEFORE INSERT ON cc_observations BEGIN SELECT RAISE(ABORT,'injected'); END;")
        with self.assertRaises(sqlite3.IntegrityError):
            emit(self.store)
        self.assertIsNone(self.store.operation_result("read-1", "result-v1"))
        self.assertEqual(self.store.pending_count(), 0)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM cc_sources WHERE source_id='doc-read-1'").fetchone()[0], 0)

    def test_ack_command_and_note_transaction_rolls_back_together(self):
        oid = emit(self.store)
        prepared = self.compressor.prepare(self.compressor.snapshot(), "native")
        self.store.db.executescript("CREATE TRIGGER fail_note BEFORE INSERT ON cc_notes BEGIN SELECT RAISE(ABORT,'injected'); END;")
        with self.assertRaises(sqlite3.IntegrityError):
            self.compressor.accept_response(prepared.request_id, response("native", note=note()), allowed)
        self.assertEqual(self.store.observation_status(oid)["state"], "included")
        self.assertEqual(self.store.pending_commands(), [])
        self.assertIsNone(self.store.active_note())
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM cc_receipts").fetchone()[0], 0)

    def test_replayed_result_and_accepted_command_are_idempotent(self):
        first = emit(self.store)
        self.assertEqual(first, emit(self.store))
        request = self.compressor.prepare(self.compressor.snapshot(), "json")
        for _ in range(3):
            self.compressor.accept_response(request.request_id, response(note=note()), allowed)
        self.assertEqual(len(self.store.pending_commands()), 1)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM cc_notes").fetchone()[0], 1)
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM cc_observations").fetchone()[0], 1)
        with self.assertRaises(ContextError):
            emit(self.store, content="Changed result under the same ID and version")

    def test_unvalidated_or_truncated_response_never_acknowledges(self):
        oid = emit(self.store)
        for mode in ("native", "json"):
            request = self.compressor.prepare(self.compressor.snapshot(), mode)
            for reason in ("length", "content_filter", None):
                bad = response(mode)
                bad["choices"][0]["finish_reason"] = reason
                with self.subTest(mode=mode, reason=reason), self.assertRaises(ContextError):
                    self.compressor.accept_response(request.request_id, bad, allowed)
            with self.assertRaises(ValueError):
                self.compressor.accept_response(request.request_id, response(mode, tool="finish"), allowed)
            with self.assertRaises(ContextError):
                self.compressor.accept_response(request.request_id, response(mode), None)
            self.assertNotEqual(self.store.observation_status(oid)["state"], "delivered")

    def test_pointer_cannot_masquerade_as_delivered_text(self):
        emit(self.store)
        snapshot = self.compressor.snapshot()
        prepared = self.compressor.fit(snapshot, "json")
        wire = json.loads(prepared.wire_json)
        body = json.loads(wire["messages"][1]["content"])
        body["observations"][0]["content"] = "See saved source"
        wire["messages"][1]["content"] = canonical(body)
        changed = canonical(wire)
        forged = replace(prepared, wire_json=changed, request_hash=text_hash(changed), wire_bytes=len(changed.encode()),
                         conservative_size=len(changed.encode()) + prepared.response_reserve)
        with self.assertRaises(ContextError):
            self.store.register_request(forged, body["observations"])
        self.assertEqual(self.store.pending_count(), 1)

    def test_snapshot_core_cannot_be_changed_before_registration(self):
        snapshot = self.compressor.snapshot()
        core = json.loads(snapshot.core_json)
        core["policies_resources"]["flags"]["control_labels_accessible"] = True
        forged = replace(snapshot, core_json=canonical(core))
        with self.assertRaises(ContextError):
            self.compressor.prepare(forged, "json")

    def test_long_unicode_and_control_text_delivered_once_without_holes(self):
        content = ("🙂я\x00\n\"é\\" * 400) + "END"
        oid = emit(self.store, content=content)
        received = []
        step = 0
        while self.store.pending_count():
            prepared = self.compressor.prepare(self.compressor.snapshot(), "json")
            fragment = user_context(prepared)["observations"][0]
            self.assertEqual(fragment["range"]["start"], len("".join(received).encode()))
            self.assertEqual(fragment["content_sha256"], text_hash(fragment["content"]))
            self.restart()
            retry = self.compressor.prepare(self.compressor.snapshot(), "native")
            self.assertEqual(user_context(retry)["observations"][0], fragment)
            self.compressor.accept_response(retry.request_id, response("native", operation_id=f"page-{step}"), allowed)
            received.append(fragment["content"])
            step += 1
        self.assertGreater(step, 3)
        self.assertEqual("".join(received), content)
        self.assertEqual(self.store.observation_status(oid)["delivered_until"], len(content.encode()))

    def test_empty_source_is_still_acknowledged_explicitly(self):
        oid = emit(self.store, content="")
        prepared = self.compressor.prepare(self.compressor.snapshot(), "native")
        self.assertEqual(user_context(prepared)["observations"][0]["content"], "")
        self.compressor.accept_response(prepared.request_id, response("native"), allowed)
        self.assertEqual(self.store.observation_status(oid)["state"], "delivered")

    def test_bounded_queue_keeps_remaining_observations(self):
        ids = [emit(self.store, f"read-{i}", f"unique page {i}") for i in range(11)]
        prepared = self.compressor.prepare(self.compressor.snapshot(), "json")
        self.assertEqual(len(user_context(prepared)["observations"]), 4)
        self.compressor.accept_response(prepared.request_id, response(), allowed)
        self.assertEqual(self.store.pending_count(), 7)
        self.assertEqual([self.store.observation_status(x)["state"] for x in ids].count("delivered"), 4)

    def test_conflicting_commands_for_same_decision_are_rejected(self):
        oid = emit(self.store)
        snapshot = self.compressor.snapshot()
        a = self.compressor.prepare(snapshot, "native")
        b = self.compressor.prepare(snapshot, "json")
        self.compressor.accept_response(a.request_id, response("native", operation_id="first"), allowed)
        with self.assertRaises(ContextError):
            self.compressor.accept_response(b.request_id, response("json", operation_id="second"), allowed)
        self.assertEqual(self.store.observation_status(oid)["state"], "delivered")

    def test_hard_process_exit_rolls_back_uncommitted_result(self):
        script = """
import os, sys
from context_compressor import ContextStore
s=ContextStore(sys.argv[1])
s.db.execute('BEGIN IMMEDIATE')
s.db.execute("INSERT INTO cc_operations VALUES('crash','v1','hash','emissions','{}')")
os._exit(17)
"""
        completed = subprocess.run([sys.executable, "-c", script, str(self.path)], capture_output=True)
        self.assertEqual(completed.returncode, 17, completed.stderr.decode())
        self.restart()
        self.assertIsNone(self.store.operation_result("crash", "v1"))

    def test_old_code_and_diagnostic_are_literal_next_request_content(self):
        code = Source("old-code", "revision-7", "def transform(X):\n    return X.fillna(0)\n", "code", origin="generated_code")
        diagnostic = Source("old-trace", "revision-7", "ValueError: transformer returned wrong row order", "code", origin="worker_output")
        self.store.complete_operation("debug-read", "v7", {"status": "succeeded"}, [Emission(code, kind="code"), Emission(diagnostic, kind="diagnostic")])
        prepared = self.compressor.prepare(self.compressor.snapshot(), "json", minimum_stage=3)
        contents = [x["content"] for x in user_context(prepared)["observations"]]
        self.assertCountEqual(contents, [code.content, diagnostic.content])

    def test_newest_observation_has_a_reserved_slot_in_backlog(self):
        ids = [emit(self.store, f"backlog-{i}", f"page {i}") for i in range(10)]
        prepared = self.compressor.prepare(self.compressor.snapshot(), "json")
        observed = [x["observation_id"] for x in user_context(prepared)["observations"]]
        self.assertEqual(observed[0], ids[-1])
        self.assertEqual(observed[1:], ids[:3])

    def test_reread_page_becomes_observation_without_republishing_or_loading_whole_source(self):
        from context_compressor import SourceSlice
        content = "Large old code " + "я" * 40000
        source = Source("old-long-code", "revision-3", content, "code", origin="code_file")
        self.store.put_source(source)
        fragment = self.store.read_source(source_id=source.source_id, version=source.version, scope="code", offset=len("Large old code ".encode()))
        observation = SourceSlice.from_fragment(fragment, kind="code")
        oid = self.store.complete_operation("read-page", "v1", fragment, [observation])[0]
        prepared = self.compressor.prepare(self.compressor.snapshot(), "json", minimum_stage=3)
        delivered = user_context(prepared)["observations"][0]
        self.assertEqual(delivered["content"], fragment["content"])
        self.assertEqual(delivered["source"], fragment["source"])
        self.assertEqual(delivered["range"], fragment["range"])
        self.assertTrue(delivered["complete"])
        self.assertFalse(delivered["document_complete"])
        self.assertIsNotNone(delivered["continuation"])
        self.compressor.accept_response(prepared.request_id, response(), allowed)
        self.assertEqual(self.store.observation_status(oid)["state"], "delivered")
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM cc_sources WHERE source_id=?", (source.source_id,)).fetchone()[0], 1)

    def test_changed_source_slice_hash_is_rejected_atomically(self):
        from context_compressor import SourceSlice
        source = Source("page", "v1", "Original page", "reference")
        self.store.put_source(source)
        fragment = self.store.handle_read(source.reference["arguments"])
        source_slice = SourceSlice.from_fragment(fragment)
        forged = replace(source_slice, content_sha256="0" * 64)
        with self.assertRaises(ContextError):
            self.store.complete_operation("forged-read", "v1", fragment, [forged])
        self.assertIsNone(self.store.operation_result("forged-read", "v1"))
        self.assertEqual(self.store.pending_count(), 0)
