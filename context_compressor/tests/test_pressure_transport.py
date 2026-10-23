import json
from pathlib import Path
import tempfile
import unittest

from context_compressor import ContextCompressor, ContextError, ContextStore
from context_compressor.contracts import canonical
from context_compressor.tests.helpers import allowed, emit, note, response, setup, transport, user_context
from context_compressor.tests.scenarios import four_attempt_fixture, pressure_scenario


class PressureTransportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = ContextStore(Path(self.directory.name) / "journal.sqlite")
        setup(self.store)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_three_real_reductions_on_distinct_steps_with_restart_and_comparison(self):
        for mode in ("native", "json"):
            with self.subTest(mode=mode):
                report = pressure_scenario(Path(self.directory.name) / mode, mode)
                self.assertEqual(report["actual_reductions_on_distinct_steps"], 3)
                self.assertEqual(report["comparison_after_third"]["status"], "confirmed_comparison")

    def test_four_attempts_use_one_snapshot_and_one_profile_calculation(self):
        compressor, calls = four_attempt_fixture(self.store)
        before = self.store.db.execute("SELECT count(*) FROM cc_snapshots").fetchone()[0]
        snapshot = compressor.snapshot()
        statements = []
        self.store.db.set_trace_callback(statements.append)
        prepared = compressor.fit(snapshot, "native")
        self.store.db.set_trace_callback(None)
        self.assertEqual(prepared.attempts, 4)
        self.assertEqual(prepared.stage, 3)
        self.assertEqual(len(prepared.fit_trace), 4)
        self.assertEqual(len(calls), 1)
        self.assertEqual(statements, [])
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM cc_snapshots").fetchone()[0], before + 1)
        self.assertEqual(user_context(prepared)["core"]["research_note"], self.store.active_note())
        self.assertEqual(user_context(prepared)["observations"], json.loads(snapshot.observations_json))

    def test_native_and_json_have_identical_note_and_read_semantics(self):
        renderer = transport(real=True)
        args = {"context_search": {"filters": {"status": "failed", "engine": "flaml"}, "limit": 3}}
        commands = [renderer.parse_response(response(mode, arguments=args, note=note()), mode) for mode in ("native", "json")]
        self.assertEqual(commands[0], commands[1])
        self.assertEqual(json.loads(commands[0].arguments_json), args)
        for mode in ("native", "json"):
            payload = json.loads(renderer.render({}, mode))
            self.assertEqual(payload["max_tokens"], 32768)
        self.assertEqual(len(json.loads(renderer.render({}, "native"))["tools"]), 9)

    def test_transport_switch_preserves_delivery_and_original_source_provenance(self):
        oid = emit(self.store)
        compressor = ContextCompressor(self.store, transport(real=True))
        snapshot = compressor.snapshot()
        native = compressor.prepare(snapshot, "native")
        self.store.channel_error(native.request_id, "provider_context_limit", "Actual provider rejection differs from local byte guard")
        fallback = compressor.prepare(snapshot, "json", minimum_stage=1)
        self.assertEqual(user_context(native)["observations"], user_context(fallback)["observations"])
        compressor.accept_response(fallback.request_id, response(note=note()), allowed)
        self.assertEqual(self.store.observation_status(oid)["state"], "delivered")
        self.assertEqual(self.store.get_meta("controller_state")["policies_resources"]["counters"]["stagnation"], 0)

    def test_malformed_and_duplicate_key_commands_cannot_be_accepted(self):
        renderer = transport()
        cases = []
        bad = response("json")
        bad["choices"][0]["message"]["content"] = '{"operation_id":"x","operation_id":"y","tool":"finish","arguments":{"reason":"done"}}'
        cases.append(("json", bad))
        bad = response("json")
        bad["choices"][0]["message"]["content"] = '{"operation_id":"x","tool":"finish","arguments":{"reason":NaN}}'
        cases.append(("json", bad))
        for text in ("text before {}", "{} trailing", "```json\n{}\n```", "{} {}", "[]"):
            bad = response("json")
            bad["choices"][0]["message"]["content"] = text
            cases.append(("json", bad))
        for mode, bad in cases:
            with self.subTest(mode=mode, bad=str(bad)[:100]), self.assertRaises(ContextError):
                renderer.parse_response(bad, mode)

    def test_provider_parameters_cannot_override_protected_transport_fields(self):
        for key in ("messages", "model", "max_tokens", "stream", "tools", "tool_choice", "parallel_tool_calls"):
            with self.subTest(key=key), self.assertRaises(ContextError):
                transport(extra={key: "override"})

    def test_supplied_schemas_are_not_mutated_by_rendering(self):
        original = json.loads((Path(__file__).parent / "fixtures/controller_schemas.json").read_text())
        from context_compressor import Transport
        before = canonical(original)
        renderer = Transport(**original)
        for mode in ("native", "json", "native"):
            renderer.render({}, mode)
        self.assertEqual(canonical(original), before)

    def test_schema_titles_are_omitted_without_changing_fields_or_defaults(self):
        from context_compressor import Transport
        original = {"type": "object", "title": "Display name", "additionalProperties": False,
            "properties": {"title": {"type": "string", "title": "Field title", "minLength": 2,
                                      "description": "A required name"},
                "data": {"type": "object", "default": {"title": "actual data"}}}, "required": ["title"]}
        renderer = Transport(system="Test", tool_schemas={"analyze_experiments": original})
        compact = renderer.schemas["analyze_experiments"]["oneOf"][0]
        self.assertNotIn("title", compact)
        self.assertEqual(compact["properties"]["title"], {"type": "string", "minLength": 2, "description": "A required name"})
        self.assertEqual(compact["properties"]["data"]["default"], {"title": "actual data"})
        self.assertEqual(compact["required"], ["title"])
        self.assertFalse(compact["additionalProperties"])

    def test_serialized_schemas_validate_both_routes_and_all_note_transports(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("Optional schema validation requires the organizer's jsonschema dependency")
        renderer = transport(real=True)
        native = json.loads(renderer.render({}, "native"))
        schemas = {entry["function"]["name"]: entry["function"]["parameters"] for entry in native["tools"]}
        for name, schema in schemas.items():
            jsonschema.Draft202012Validator.check_schema(schema)
        for arguments in ({"context_search": {"filters": {"status": "failed"}}},
                          {"context_read": {"source_id": "id", "version": "v1", "scope": "reference"}}, {}):
            payload = {**arguments, "operation_id": "read-step", "research_note": note()}
            jsonschema.validate(payload, schemas["analyze_experiments"])
        user = json.loads(json.loads(renderer.render({}, "json"))["messages"][1]["content"])
        for name, schema in user["tool_argument_schemas"].items():
            jsonschema.Draft202012Validator.check_schema({**schema, "$defs": user["$defs"]})
        jsonschema.validate({"operation_id": "x", "tool": "finish", "arguments": {"reason": "done"}, "research_note": note()}, user["command_schema"])
