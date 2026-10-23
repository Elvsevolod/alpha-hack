from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from context_compressor import (AccessDenied, ContextCompressor, ContextError, ContextStore, Record, Source)
from context_compressor.tests.helpers import emit, seed_history, setup, transport, user_context


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "journal.sqlite"
        self.store = ContextStore(self.path)
        setup(self.store)
        self.compressor = ContextCompressor(self.store, transport())

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_old_failure_is_found_without_id_among_1000_in_two_reads(self):
        seed_history(self.store, 1000)
        source = Source("old-unique-diagnostic", "v3", "Confirmed incompatible sparse format: DENSE_ONLY, use a dense inner-fit representation.", "registered_feedback")
        self.store.put_source(source)
        self.store.publish_records([Record("hidden-old-failure", source.source_id, source.version,
            component="categorical_encoder", engine="engine-old", status="incompatible", error_code="DENSE_ONLY", protocol="protocol-old")])
        first = self.store.handle_read({"context_search": {"filters": {"component": "categorical_encoder", "engine": "engine-old",
                                                   "status": "incompatible", "error_code": "DENSE_ONLY", "protocol": "protocol-old"}}})
        self.assertEqual(len(first["records"]), 1)
        second = self.store.handle_read(first["records"][0]["read"]["arguments"])
        self.assertEqual(second["content"], source.content)
        self.assertEqual(second["source"]["scope"], "registered_feedback")
        self.assertEqual(first["records"][0]["protocol"], "protocol-old")
        self.assertFalse(first["ambiguous"])

    def test_cursor_is_stable_across_appends_and_bound_to_query(self):
        seed_history(self.store, 25)
        first = self.store.search(limit=3)
        self.assertTrue(first["ambiguous"])
        seed_history(self.store, 10, first_index=25)
        cursor, records = first["cursor"], list(first["records"])
        while cursor:
            page = self.store.search(cursor=cursor, limit=3)
            records.extend(page["records"])
            cursor = page["cursor"]
        self.assertEqual(len(records), 25)
        self.assertEqual(len({x["record_id"] for x in records}), 25)
        with self.assertRaises(ContextError):
            self.store.search({"engine": "different"}, cursor=first["cursor"])
        bad = first["cursor"][:-5] + "ABCDE"
        with self.assertRaises(ContextError):
            self.store.search(cursor=bad)

    def test_cursor_survives_restart_and_does_not_survive_permission_change(self):
        seed_history(self.store, 20)
        first = self.store.search(limit=2)
        self.store.close()
        self.store = ContextStore(self.path)
        self.assertTrue(self.store.search(cursor=first["cursor"], limit=2)["records"])
        with ContextStore(self.path, allowed_scopes=frozenset({"reference"})) as restricted:
            with self.assertRaises(ContextError):
                restricted.search(cursor=first["cursor"])

    def test_access_is_enforced_for_search_and_source_pages(self):
        source = Source("saved-feedback", "v1", "registered metrics only", "registered_feedback")
        self.store.put_source(source)
        self.store.publish_records([Record("saved-evaluation", source.source_id, "v1")])
        with ContextStore(self.path, allowed_scopes=frozenset({"reference"})) as restricted:
            self.assertEqual(restricted.search()["records"], [])
            for scope in ("registered_feedback", "reference", "control_labels"):
                with self.subTest(scope=scope), self.assertRaises(AccessDenied):
                    restricted.read_source(source_id="saved-feedback", version="v1", scope=scope)
        with self.assertRaises(AccessDenied):
            Source("labels", "v1", "forbidden", "control_labels")

    def test_unknown_errors_are_not_merged_and_known_groups_keep_conditions(self):
        source = Source("common-diagnostic", "v1", "Identical error text", "registered_feedback")
        self.store.put_source(source)
        common = Record("known-0", source.source_id, "v1", component="encoder", engine="direct", status="incompatible",
                        error_code="TYPE_CONFLICT", protocol="p1", environment="e1")
        records = [common, replace(common, record_id="known-1"), replace(common, record_id="known-2", environment="e2"),
                   replace(common, record_id="unknown-1", error_code=None), replace(common, record_id="unknown-2", error_code=None),
                   replace(common, record_id="oom", status="memory_limit", error_code="OOM"),
                   replace(common, record_id="timeout", status="timeout", error_code="TIMEOUT"),
                   replace(common, record_id="unknown-outcome", status="outcome_unknown", error_code="UNKNOWN")]
        self.store.publish_records(records)
        groups = self.store.failure_groups()
        self.assertEqual(len(groups), 7)
        self.assertEqual(sorted(group["count"] for group in groups), [1, 1, 1, 1, 1, 1, 2])
        self.assertEqual({group["status"] for group in groups}, {"incompatible", "memory_limit", "timeout", "outcome_unknown"})

    def test_parent_delta_requires_matching_scope_protocol_environment_and_metric(self):
        source = Source("parent-source", "v1", "permitted metrics", "registered_feedback")
        self.store.put_source(source)
        parent = Record("parent", source.source_id, "v1", status="completed", metric="rmse", maximize=False, score=2,
                        protocol="p1", environment="e1")
        child = replace(parent, record_id="child", parent_id="parent", score=1.5)
        records = [parent, child, replace(child, record_id="pilot", evaluation_scope="pilot"),
                   replace(child, record_id="different-protocol", protocol="p2"),
                   replace(child, record_id="different-environment", environment="e2"),
                   replace(child, record_id="different-metric", metric="mae")]
        self.store.publish_records(records)
        setup(self.store, work={"active_id": "child", "related_ids": [r.record_id for r in records]})
        cards = {json.loads(x)["record_id"]: json.loads(x) for x in self.compressor.snapshot().cards}
        self.assertEqual(cards["child"]["parent_comparison"]["gain"], 0.5)
        self.assertIn("not_causal", cards["child"]["parent_comparison"]["interpretation"])
        for item in records[2:]:
            self.assertFalse(cards[item.record_id]["parent_comparison"]["comparable"])
            self.assertNotIn("score_delta", cards[item.record_id]["parent_comparison"])

    def test_selection_keeps_active_intent_baseline_leader_parent_failure_and_dependencies(self):
        seed_history(self.store, 100)
        setup(self.store, work={"active_id": "experiment-4", "baseline_id": "experiment-1", "leader_id": "experiment-2",
                               "parent_id": "experiment-3", "related_failure_id": "experiment-0", "dependencies": ["experiment-5"]})
        snapshot = self.compressor.snapshot()
        ids = [json.loads(value)["record_id"] for value in snapshot.cards]
        self.assertEqual(ids[0], "experiment-4")
        self.assertTrue({f"experiment-{i}" for i in range(6)} <= set(ids))
        self.assertLessEqual(len(snapshot.cards), 32)
        self.assertLessEqual(len(json.loads(snapshot.failures_json)), 8)
        self.assertLessEqual(snapshot.source_reads, 12)

    def test_excluded_manifest_enumerates_exact_snapshot_ids_without_full_scan_in_fit(self):
        seed_history(self.store, 55)
        snapshot = self.compressor.snapshot()
        request = self.compressor.prepare(snapshot, "json", minimum_stage=2)
        clause, params = self.store._scope_sql()
        expected = {row[0] for row in self.store.db.execute(f"SELECT record_id FROM cc_records WHERE seq<=? AND {clause}", (snapshot.max_record_seq, *params))}
        seed_history(self.store, 5, first_index=55)
        excluded, cursor = set(), 0
        while True:
            page = self.store.excluded_ids(request.request_id, after_seq=cursor, limit=7)
            excluded.update(page["ids"])
            if page["next_after_seq"] is None:
                break
            cursor = page["next_after_seq"]
        selected = set(request.selected_ids)
        self.assertFalse(selected & excluded)
        self.assertEqual(selected | excluded, expected)
        self.assertEqual(len(excluded), user_context(request)["compaction"]["exclusions"]["count"])

    def test_read_only_routes_cannot_mutate_policies_or_read_arbitrary_files(self):
        original = self.store.get_meta("controller_state")
        invalid = [{"context_read": {"path": "/etc/passwd"}}, {"context_search": {"filters": {"sql": "select *"}}},
                   {"context_read": {}, "policy": {"control_labels_accessible": True}}, {"set_policy": {"worker_network_allowed": True}}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ContextError):
                self.store.handle_read(value)
        self.assertEqual(self.store.get_meta("controller_state"), original)

    def test_source_catalog_discovers_versioned_documents_without_known_id(self):
        self.store.put_source(Source("old-guide", "v7", "Useful earlier reference", "reference", origin="offline_reference"))
        page = self.store.search({"kind": "source", "component": "offline_reference"})
        self.assertEqual(len(page["records"]), 1)
        document = self.store.handle_read(page["records"][0]["read"]["arguments"])
        self.assertEqual(document["source"]["version"], "v7")
        self.assertEqual(document["source"]["trust"], "external_data")

    def test_read_ranges_reject_invalid_offsets_without_mutating_delivery(self):
        emit(self.store, content="🙂абв")
        for offset in (-1, 1, 1000, True):
            with self.subTest(offset=offset), self.assertRaises(ContextError):
                self.store.read_source(source_id="doc-read-1", version="v1", scope="reference", offset=offset)
        self.assertEqual(self.store.pending_count(), 1)
