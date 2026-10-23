"""Append-only permitted sources, transactional outbox, and bounded discovery.

This database is deliberately separate from the developing agent. Integration
must route result persistence through complete_operation, rather than dual-write
an operation result and its delivery record in two independent transactions.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import base64
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
import time
from typing import Callable, Iterable

from .contracts import (AccessDenied, Command, ContextError, Emission, Record, SCOPES,
                        Source, SourceSlice, bounded, canonical, digest, identifier, read_ref, size, text_hash)


FILTERS = frozenset({"kind", "component", "engine", "status", "error_code", "protocol",
                     "parent_id", "environment", "tool", "evaluation_scope", "record_id"})


class ContextStore:
    def __init__(self, path: str | Path, *, allowed_scopes: frozenset[str] = SCOPES, tool_api_version=1):
        self.tool_api_version = tool_api_version
        if not allowed_scopes or not set(allowed_scopes) <= SCOPES:
            raise AccessDenied("Invalid scope allowlist")
        self.allowed_scopes = frozenset(allowed_scopes)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS cc_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cc_sources(
                source_id TEXT, version TEXT, section TEXT, scope TEXT NOT NULL,
                trust TEXT NOT NULL, origin TEXT NOT NULL, sha256 TEXT NOT NULL,
                byte_length INTEGER NOT NULL, content BLOB NOT NULL,
                PRIMARY KEY(source_id,version,section));
            CREATE TABLE IF NOT EXISTS cc_operations(
                operation_id TEXT, result_version TEXT, result_hash TEXT NOT NULL,
                emissions_hash TEXT NOT NULL, result_json TEXT NOT NULL,
                PRIMARY KEY(operation_id,result_version));
            CREATE TABLE IF NOT EXISTS cc_observations(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, observation_id TEXT UNIQUE NOT NULL,
                operation_id TEXT NOT NULL, result_version TEXT NOT NULL,
                source_id TEXT NOT NULL, source_version TEXT NOT NULL, section TEXT NOT NULL,
                scope TEXT NOT NULL, kind TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
                next_start INTEGER NOT NULL, delivered INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS cc_pending ON cc_observations(delivered,seq);
            CREATE TABLE IF NOT EXISTS cc_records(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT UNIQUE NOT NULL,
                kind TEXT, component TEXT, engine TEXT, status TEXT, error_code TEXT,
                protocol TEXT, parent_id TEXT, environment TEXT, tool TEXT,
                evaluation_scope TEXT, data_scope TEXT, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS cc_record_filters ON cc_records(
                kind,component,engine,status,error_code,protocol,parent_id,seq);
            CREATE INDEX IF NOT EXISTS cc_record_parent ON cc_records(parent_id,seq);
            CREATE TABLE IF NOT EXISTS cc_snapshots(
                snapshot_id TEXT PRIMARY KEY, max_record_seq INTEGER NOT NULL, manifest_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cc_requests(
                request_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, transport TEXT NOT NULL,
                request_hash TEXT NOT NULL, wire_json TEXT NOT NULL, audit_json TEXT NOT NULL,
                created_at REAL NOT NULL, FOREIGN KEY(snapshot_id) REFERENCES cc_snapshots(snapshot_id));
            CREATE TABLE IF NOT EXISTS cc_request_fragments(
                request_id TEXT, observation_id TEXT, start INTEGER, end INTEGER, envelope_json TEXT NOT NULL,
                PRIMARY KEY(request_id,observation_id),
                FOREIGN KEY(request_id) REFERENCES cc_requests(request_id));
            CREATE TABLE IF NOT EXISTS cc_commands(
                operation_id TEXT PRIMARY KEY, command_json TEXT NOT NULL,
                registered_request TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS cc_decisions(
                snapshot_id TEXT PRIMARY KEY, operation_id TEXT NOT NULL, request_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cc_receipts(request_id TEXT PRIMARY KEY, operation_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cc_notes(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, note_id TEXT UNIQUE NOT NULL,
                command_id TEXT NOT NULL, status TEXT NOT NULL, note_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cc_channel_errors(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, code TEXT, diagnostic TEXT, time REAL);
            CREATE TABLE IF NOT EXISTS cc_usage(request_id TEXT PRIMARY KEY, usage_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cc_profiles(cache_key TEXT PRIMARY KEY, reference_json TEXT NOT NULL);
        """)
        with self.transaction():
            if self.get_meta("schema_version") not in {None, 1}:
                raise ContextError("Unsupported context store schema")
            self._put_meta("schema_version", 1)
            if self.get_meta("cursor_secret") is None:
                self._put_meta("cursor_secret", secrets.token_hex(32))

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @contextmanager
    def transaction(self):
        if self.db.in_transaction:
            raise ContextError("Nested context-store transactions are not supported")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def get_meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM cc_meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _put_meta(self, key: str, value):
        self.db.execute("INSERT INTO cc_meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, canonical(value)))

    def _check_scope(self, scope):
        if scope not in self.allowed_scopes:
            raise AccessDenied(f"Source scope is not permitted: {scope}")

    def _scope_sql(self, column="data_scope"):
        values = tuple(sorted(self.allowed_scopes))
        return f"{column} IN ({','.join('?' for _ in values)})", values

    def route(self, value):
        if self.tool_api_version != 2:
            return value
        if isinstance(value, list):
            return [self.route(item) for item in value]
        if not isinstance(value, dict):
            return value
        if value == {"kind": "research_note"}:
            return {"kind": "note"}
        if value.get("tool") == "analyze_experiments" and isinstance(value.get("arguments"), dict):
            args = value["arguments"]
            if "context_read" in args:
                return {"tool": "read_run_records", "arguments": {"source": args["context_read"]}}
            if "context_search" in args:
                search = dict(args["context_search"])
                if search.get("filters", {}).get("kind") == "research_note":
                    search["filters"] = {**search["filters"], "kind": "note"}
                return {"tool": "list_run_records", "arguments": search}
        return {key: self.route(item) for key, item in value.items()}

    def _put_source(self, source: Source):
        self._check_scope(source.scope)
        blob = source.content.encode("utf-8")
        values = (source.source_id, source.version, source.section, source.scope, source.trust,
                  source.origin, text_hash(source.content), len(blob), blob)
        previous = self.db.execute("SELECT scope,trust,origin,sha256 FROM cc_sources WHERE source_id=? AND version=? AND section=?",
                                   values[:3]).fetchone()
        if previous and tuple(previous) != (source.scope, source.trust, source.origin, values[6]):
            raise ContextError("A source identity/version cannot be reused with changed content or provenance")
        self.db.execute("INSERT OR IGNORE INTO cc_sources VALUES(?,?,?,?,?,?,?,?,?)", values)
        if not source.section:
            catalog_id = "source:" + digest({"id": source.source_id, "version": source.version})
            self._put_record(Record(catalog_id, source.source_id, source.version, source.scope,
                                    kind="source", component=source.origin, engine="source", status="available",
                                    tool="read_run_records" if self.tool_api_version == 2 else "analyze_experiments"))

    def put_source(self, source: Source):
        """Controller-only publication; never expose this method as a model tool."""
        with self.transaction():
            self._put_source(source)
        return self.route(source.reference)

    def _source_row(self, source_id, version, section, scope):
        self._check_scope(scope)
        row = self.db.execute("SELECT source_id,version,section,scope,trust,origin,sha256,byte_length FROM cc_sources "
                              "WHERE source_id=? AND version=? AND section=?", (source_id, version, section)).fetchone()
        if not row or row["scope"] != scope:
            raise AccessDenied("Source is unavailable in the requested scope/version")
        return row

    def _piece(self, source_id, version, section, scope, start, end, limit=768):
        row = self._source_row(source_id, version, section, scope)
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= row["byte_length"]:
            raise ContextError("Invalid UTF-8 byte range")
        blob, first, following = self.db.execute("SELECT substr(content,?,?),substr(content,?,1),substr(content,?,1) "
            "FROM cc_sources WHERE source_id=? AND version=? AND section=?",
            (start + 1, min(limit, end - start), start + 1, end + 1, source_id, version, section)).fetchone()
        if any(part and part[0] & 0xC0 == 0x80 for part in (first, following)):
            raise ContextError("Requested byte range splits a UTF-8 character")
        blob = blob or b""
        # Keep a character straddling the end for the next page, never drop it.
        try:
            content = blob.decode("utf-8")
        except UnicodeDecodeError as error:
            if error.reason != "unexpected end of data" or error.start == 0:
                raise ContextError("Offset or limit splits a UTF-8 character") from error
            blob = blob[:error.start]
            content = blob.decode("utf-8")
        stop = start + len(blob)
        complete = stop == end
        return self.route({"source": {"source_id": source_id, "version": version, "section": section,
                           "sha256": row["sha256"], "scope": scope, "trust": row["trust"], "origin": row["origin"]},
                "range": {"unit": "utf8_bytes", "start": start, "end": stop},
                "content": content, "content_sha256": text_hash(content), "complete": complete,
                "document_complete": start == 0 and end == row["byte_length"] and complete,
                "requested_end": end, "source_bytes": row["byte_length"],
                "read": read_ref(source_id, version, scope, section, start),
                "continuation": None if stop == row["byte_length"] else read_ref(source_id, version, scope, section, stop)})

    def read_source(self, *, source_id, version, scope, section="", offset=0, limit=768):
        identifier(source_id, "source_id")
        bounded(version, 128, "source version")
        bounded(section, 128, "source section", empty=True)
        if type(limit) is not int or not 4 <= limit <= 768:
            raise ContextError("Read limit must be 4..768 UTF-8 bytes")
        row = self._source_row(source_id, version, section, scope)
        result = self._piece(source_id, version, section, scope, offset, row["byte_length"], limit)
        if size(result) > 8192:
            raise ContextError("Source envelope exceeds observation budget")
        return result

    def complete_operation(self, operation_id: str, result_version: str, result: dict,
                           emissions: Iterable[Emission | SourceSlice], *, records: Iterable[Record] = (),
                           commit: Callable[[sqlite3.Connection], None] | None = None) -> tuple[str, ...]:
        """Atomically save a full result and its outbox. Replaying this is idempotent.

        Emit a bounded training card, or the requested reference/code/diagnostic
        sources. Large requested ranges are acknowledged one delivered piece at a time.
        """
        identifier(operation_id, "operation_id")
        bounded(result_version, 128, "result_version")
        if not isinstance(result, dict):
            raise ContextError("Operation result must be an object")
        emissions = tuple(emissions)
        if not emissions:
            raise ContextError("A completed operation must publish an observation")
        ids = ((result["observation_id"],) if result.get("tool_api_version") == 2 and len(emissions) == 1
               else tuple(item.observation_id(operation_id, result_version) for item in emissions))
        if len(set(ids)) != len(ids):
            raise ContextError("Duplicate emissions")
        encoded = canonical(result)
        with self.transaction():
            normalized = []
            for item in emissions:
                if isinstance(item, Emission):
                    self._put_source(item.source)
                    start = item.start
                    end = len(item.source.content.encode("utf-8")) if item.end is None else item.end
                    source_id, version, section, scope = item.source.source_id, item.source.version, item.source.section, item.source.scope
                elif isinstance(item, SourceSlice):
                    source_id, version, section, scope = item.source_id, item.version, item.section, item.scope
                    start, end = item.start, item.end
                else:
                    raise ContextError("Expected an Emission or a persisted SourceSlice")
                meta = self._source_row(source_id, version, section, scope)
                if isinstance(item, SourceSlice):
                    piece = self._piece(source_id, version, section, scope, start, end)
                    if (meta["sha256"] != item.source_sha256 or piece["content_sha256"] != item.content_sha256
                            or piece["range"]["end"] != end):
                        raise ContextError("Read page no longer matches its persisted source")
                normalized.append({"source_id": source_id, "version": version, "section": section, "scope": scope,
                                   "start": start, "end": end, "kind": item.kind, "source_sha256": meta["sha256"]})
            signature = digest(normalized)
            previous = self.db.execute("SELECT result_hash,emissions_hash FROM cc_operations WHERE operation_id=? AND result_version=?",
                                       (operation_id, result_version)).fetchone()
            if previous and tuple(previous) != (digest(result), signature):
                raise ContextError("Operation result version was reused with different data")
            self.db.execute("INSERT OR IGNORE INTO cc_operations VALUES(?,?,?,?,?)",
                            (operation_id, result_version, digest(result), signature, encoded))
            for item, observation_id in zip(normalized, ids):
                self.db.execute("""INSERT OR IGNORE INTO cc_observations(
                    observation_id,operation_id,result_version,source_id,source_version,section,
                    scope,kind,start,end,next_start) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (observation_id, operation_id, result_version, item["source_id"], item["version"],
                     item["section"], item["scope"], item["kind"], item["start"], item["end"], item["start"]))
            self.db.execute("UPDATE cc_commands SET completed=1 WHERE operation_id=?", (operation_id,))
            for record in records:
                self._put_record(record)
            # The controller uses this same database transaction to persist its
            # terminal operation record. A failed commit rolls back the outbox too.
            if commit is not None:
                commit(self.db)
        return ids

    def operation_result(self, operation_id: str, result_version: str):
        row = self.db.execute("SELECT result_json FROM cc_operations WHERE operation_id=? AND result_version=?",
                              (operation_id, result_version)).fetchone()
        return json.loads(row[0]) if row else None

    def pending_count(self):
        clause, params = self._scope_sql("scope")
        return self.db.execute(f"SELECT count(*) FROM cc_observations WHERE delivered=0 AND {clause}", params).fetchone()[0]

    def pending_fragments(self, *, limit=4, byte_budget=8192):
        if type(limit) is not int or not 1 <= limit <= 4 or not 2048 <= byte_budget <= 8192:
            raise ContextError("Invalid observation batch limits")
        clause, params = self._scope_sql("scope")
        newest = self.db.execute(f"SELECT * FROM cc_observations WHERE delivered=0 AND {clause} ORDER BY seq DESC LIMIT 1", params).fetchone()
        rows = [] if newest is None else [newest]
        if newest is not None and limit > 1:
            rows.extend(self.db.execute(f"SELECT * FROM cc_observations WHERE delivered=0 AND {clause} AND seq!=? ORDER BY seq LIMIT ?", (*params, newest["seq"], limit - 1)).fetchall())
        result = []
        for row in rows:
            piece = self._observation_piece(row, row["next_start"])
            if size([*result, piece]) > byte_budget:
                if not result:
                    raise ContextError("Minimum observation does not fit the explicit batch budget")
                break
            result.append(piece)
        return result

    def _observation_piece(self, row, start):
        limit = 6144 if self.tool_api_version == 2 else 768
        while True:
            piece = self._piece(row["source_id"], row["source_version"], row["section"], row["scope"], start, row["end"], limit)
            value = {**piece, "observation_id": row["observation_id"], "operation_id": row["operation_id"],
                "result_version": row["result_version"], "kind": row["kind"],
                "requested_range": {"unit": "utf8_bytes", "start": row["start"], "end": row["end"]}}
            if size([value]) <= 8192 or limit <= 768:
                return value
            limit //= 2

    def observation_status(self, observation_id):
        row = self.db.execute("SELECT * FROM cc_observations WHERE observation_id=?", (observation_id,)).fetchone()
        if not row:
            return None
        self._check_scope(row["scope"])
        included = self.db.execute("SELECT 1 FROM cc_request_fragments WHERE observation_id=? LIMIT 1", (observation_id,)).fetchone()
        return {"state": "delivered" if row["delivered"] else "included" if included else "pending",
                "delivered_until": row["next_start"], "requested_end": row["end"]}

    def _put_record(self, record: Record):
        self._source_row(record.source_id, record.source_version, "", record.data_scope)
        payload = canonical(asdict(record))
        old = self.db.execute("SELECT payload FROM cc_records WHERE record_id=?", (record.record_id,)).fetchone()
        if old and old[0] != payload:
            raise ContextError("A historical record is immutable; use a new ID for a new result version")
        self.db.execute("""INSERT OR IGNORE INTO cc_records(record_id,kind,component,engine,status,error_code,
            protocol,parent_id,environment,tool,evaluation_scope,data_scope,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (record.record_id, record.kind, record.component, record.engine, record.status, record.error_code,
             record.protocol, record.parent_id, record.environment, record.tool, record.evaluation_scope, record.data_scope, payload))

    def publish_records(self, records: Iterable[Record]):
        with self.transaction():
            for record in records:
                self._put_record(record)

    def record(self, record_id):
        row = self.db.execute("SELECT seq,payload FROM cc_records WHERE record_id=?", (record_id,)).fetchone()
        if not row:
            return None
        value = json.loads(row["payload"])
        self._check_scope(value["data_scope"])
        return self.route({**value, "created_index": row["seq"],
                "read": read_ref(value["source_id"], value["source_version"], value["data_scope"])})

    def _cursor(self, payload):
        raw = canonical(payload).encode()
        signature = hmac.new(bytes.fromhex(self.get_meta("cursor_secret")), raw, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(signature + raw).decode()

    def _decode_cursor(self, token):
        if not isinstance(token, str) or len(token) > 4096:
            raise ContextError("Invalid cursor")
        try:
            blob = base64.b64decode(token, altchars=b"-_", validate=True)
            signature, raw = blob[:32], blob[32:]
            expected = hmac.new(bytes.fromhex(self.get_meta("cursor_secret")), raw, hashlib.sha256).digest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError("signature")
            return json.loads(raw)
        except (ValueError, TypeError) as error:
            raise ContextError("Invalid or foreign cursor") from error

    def search(self, filters: dict | None = None, *, cursor=None, limit=10):
        filters = {"kind": "experiment", **(filters or {})}
        if set(filters) - FILTERS or type(limit) is not int or not 1 <= limit <= 10:
            raise ContextError("Unsupported search filters or page size")
        for key, value in filters.items():
            bounded(value, 128, f"filter {key}")
        fingerprint = digest({"filters": filters, "scopes": sorted(self.allowed_scopes)})
        scope_clause, scope_params = self._scope_sql()
        maximum = self.db.execute(f"SELECT coalesce(max(seq),0) FROM cc_records WHERE {scope_clause}", scope_params).fetchone()[0]
        after = 0
        if cursor is not None:
            decoded = self._decode_cursor(cursor)
            if decoded.get("fingerprint") != fingerprint:
                raise ContextError("Cursor is bound to a different query or permissions")
            maximum, after = decoded["maximum"], decoded["after"]
        clause, params = self._scope_sql()
        where = [clause, "seq>?", "seq<=?", *[f"{key}=?" for key in sorted(filters)]]
        values = (*params, after, maximum, *[filters[key] for key in sorted(filters)], limit + 1)
        rows = self.db.execute(f"SELECT seq,record_id FROM cc_records WHERE {' AND '.join(where)} ORDER BY seq,record_id LIMIT ?", values).fetchall()
        cards, last = [], after
        for row in rows[:limit]:
            card = self.record(row["record_id"])
            if size({"records": [*cards, card]}) > 6144:
                break
            cards.append(card)
            last = row["seq"]
        if rows and not cards:
            raise ContextError("A search card cannot fit the observation budget")
        more = len(rows) > len(cards)
        next_cursor = self._cursor({"fingerprint": fingerprint, "maximum": maximum, "after": last}) if more else None
        result = {"records": cards, "cursor": next_cursor, "snapshot_max_seq": maximum,
                  "filters": filters, "ambiguous": len(cards) > 1 or more,
                  "source": "controller_index", "trust": "controller_fact"}
        if size(result) > 8192:
            raise ContextError("Search envelope exceeds observation budget")
        return result

    def failure_groups(self, limit=8):
        clause, params = self._scope_sql()
        # Unknown errors deliberately form distinct groups even if their text matches.
        rows = self.db.execute(f"""SELECT count(*) AS n,max(seq) AS example_seq FROM cc_records
            WHERE kind='experiment' AND status!='completed' AND {clause}
            GROUP BY error_code,CASE WHEN error_code IS NULL THEN record_id END,
                     component,engine,tool,environment,protocol,status,evaluation_scope,data_scope
            ORDER BY max(seq) DESC LIMIT ?""", (*params, limit)).fetchall()
        result = []
        for row in rows:
            example = self.db.execute("SELECT record_id FROM cc_records WHERE seq=?", (row["example_seq"],)).fetchone()[0]
            record = self.record(example)
            result.append({"count": row["n"], "example_id": example,
                           **{key: record[key] for key in ("component", "engine", "tool", "error_code", "status", "protocol", "environment", "evaluation_scope", "data_scope")},
                           "read": record["read"]})
        return result

    def active_note(self):
        return self.get_meta("active_note")

    def register_request(self, prepared, observations):
        """Bind the receipt to literal content in the exact serialized user message."""
        wire = json.loads(prepared.wire_json)
        if text_hash(prepared.wire_json) != prepared.request_hash or canonical(wire) != prepared.wire_json:
            raise ContextError("Prepared wire body or hash was changed")
        messages = wire.get("messages", [])
        if len(messages) != 2 or messages[1].get("role") != "user":
            raise ContextError("Unexpected transport framing")
        context = json.loads(messages[1]["content"])
        if context.get("observations") != observations:
            raise ContextError("Requested content, including provenance, must occur literally in the observation block")
        if context.get("compaction", {}).get("snapshot_id") != prepared.snapshot_id:
            raise ContextError("Request is bound to a different snapshot")
        snapshot = self.db.execute("SELECT manifest_json FROM cc_snapshots WHERE snapshot_id=?", (prepared.snapshot_id,)).fetchone()
        if not snapshot:
            raise ContextError("Unknown decision snapshot")
        manifest = json.loads(snapshot[0])
        checked_core = json.loads(canonical(context.get("core")))
        checked_core["policies_resources"].pop("remaining_seconds", None)
        if (digest(checked_core) != manifest["core_sha256"]
                or digest(observations) != manifest["observations_sha256"]
                or sorted(self.allowed_scopes) != manifest["allowed_scopes"]):
            raise ContextError("Protected snapshot content or access scope was changed")
        if not set(prepared.selected_ids) <= set(manifest["candidate_ids"]):
            raise ContextError("Request selected IDs outside the decision snapshot")
        from .contracts import SAFETY_MARGIN
        actual_bytes = len(prepared.wire_json.encode("utf-8"))
        if (actual_bytes != prepared.wire_bytes or wire.get("max_tokens") != prepared.response_reserve
                or prepared.conservative_size != actual_bytes + prepared.response_reserve
                or prepared.conservative_size + SAFETY_MARGIN > prepared.budget):
            raise ContextError("Wire size, reserve, or budget evidence is inconsistent")
        audit = {"builder_version": context["core"]["identity"]["builder_version"],
                 "selected_ids": prepared.selected_ids, "stage": prepared.stage, "attempts": prepared.attempts,
                 "wire_bytes": prepared.wire_bytes, "response_reserve": prepared.response_reserve,
                 "conservative_size": prepared.conservative_size, "budget": prepared.budget,
                 "build_seconds": prepared.build_seconds, "block_bytes": dict(prepared.block_bytes),
                 "fit_trace": [json.loads(item) for item in prepared.fit_trace],
                 "source_reads": manifest["source_reads"],
                 "exclusions": context["compaction"]["exclusions"]}
        with self.transaction():
            old = self.db.execute("SELECT request_hash FROM cc_requests WHERE request_id=?", (prepared.request_id,)).fetchone()
            if old:
                if old[0] != prepared.request_hash:
                    raise ContextError("request_id cannot be reused with a different wire body")
                return
            if self.db.execute("SELECT 1 FROM cc_decisions WHERE snapshot_id=?", (prepared.snapshot_id,)).fetchone():
                raise ContextError("This decision snapshot has already been consumed")
            checked = []
            for piece in observations:
                row = self.db.execute("SELECT * FROM cc_observations WHERE observation_id=?", (piece.get("observation_id"),)).fetchone()
                if not row or row["delivered"] or row["next_start"] != piece["range"]["start"]:
                    raise ContextError("Observation is stale or unavailable")
                expected = self._observation_piece(row, row["next_start"])
                if piece != expected:
                    raise ContextError("A pointer, shortened text, or changed provenance cannot acknowledge a source fragment")
                checked.append(piece)
            self.db.execute("INSERT INTO cc_requests VALUES(?,?,?,?,?,?,?)",
                            (prepared.request_id, prepared.snapshot_id, prepared.transport, prepared.request_hash,
                             prepared.wire_json, canonical(audit), time.time()))
            for piece in checked:
                self.db.execute("INSERT INTO cc_request_fragments VALUES(?,?,?,?,?)",
                                (prepared.request_id, piece["observation_id"], piece["range"]["start"],
                                 piece["range"]["end"], canonical(piece)))

    def request(self, request_id):
        row = self.db.execute("SELECT * FROM cc_requests WHERE request_id=?", (request_id,)).fetchone()
        if not row:
            raise ContextError("Unknown request_id")
        return dict(row)

    def accept_command(self, request_id: str, command: Command, validate: Callable[[Command], None]):
        """Validate schema AND controller admissibility before one atomic registration.

        The required callback must raise on rejection and have no execution side
        effects. This method only registers; execution belongs to the controller.
        """
        if not callable(validate):
            raise ContextError("A controller schema/admissibility validator is required")
        command = Command.parse(command.as_dict())
        encoded = canonical(command.as_dict())
        with self.transaction():
            request = self.request(request_id)
            old_receipt = self.db.execute("SELECT operation_id FROM cc_receipts WHERE request_id=?", (request_id,)).fetchone()
            previous = self.db.execute("SELECT command_json FROM cc_commands WHERE operation_id=?", (command.operation_id,)).fetchone()
            if old_receipt:
                if old_receipt[0] != command.operation_id or previous[0] != encoded:
                    raise ContextError("A request already accepted a different command")
                return command
            body = json.loads(json.loads(request["wire_json"])["messages"][1]["content"])
            state = self.get_meta("controller_state")
            if (state["identity"] != body["core"]["identity"]
                    or state["policies_resources"]["flags"] != body["core"]["policies_resources"]["flags"]):
                raise ContextError("Controller identity/policies changed since the request; revalidate from a new snapshot")
            if validate(command) is False:
                raise ContextError("Controller rejected command")
            decision = self.db.execute("SELECT operation_id FROM cc_decisions WHERE snapshot_id=?", (request["snapshot_id"],)).fetchone()
            if decision and decision[0] != command.operation_id:
                raise ContextError("A decision snapshot already accepted another command")
            if previous and previous[0] != encoded:
                raise ContextError("operation_id already names a different command")
            already_completed = self.db.execute("SELECT 1 FROM cc_operations WHERE operation_id=? LIMIT 1", (command.operation_id,)).fetchone()
            self.db.execute("INSERT OR IGNORE INTO cc_commands VALUES(?,?,?,?)",
                            (command.operation_id, encoded, request_id, bool(already_completed)))
            self.db.execute("INSERT OR IGNORE INTO cc_decisions VALUES(?,?,?)",
                            (request["snapshot_id"], command.operation_id, request_id))
            self.db.execute("INSERT INTO cc_receipts VALUES(?,?)", (request_id, command.operation_id))
            if command.note and not previous:
                note = command.note.persisted(command.operation_id)
                self.db.execute("INSERT INTO cc_notes(note_id,command_id,status,note_json) VALUES(?,?,?,?)",
                                (note["note_id"], command.operation_id, note["status"], canonical(note)))
                self._put_meta("active_note", None if note["status"] == "closed" else note)
                source = Source("note:" + note["note_id"], "1", canonical(note), "controller", "agent_assumption", origin="accepted_command")
                self._put_source(source)
                self._put_record(Record(source.source_id, source.source_id, "1", "controller", kind="research_note",
                                        component="research", status=note["status"], tool=command.tool))
            for row in self.db.execute("SELECT * FROM cc_request_fragments WHERE request_id=?", (request_id,)).fetchall():
                current = self.db.execute("SELECT next_start,end FROM cc_observations WHERE observation_id=?", (row["observation_id"],)).fetchone()
                if row["start"] > current["next_start"]:
                    raise ContextError("A delivery acknowledgement would skip an undelivered range")
                until = max(current["next_start"], row["end"])
                self.db.execute("UPDATE cc_observations SET next_start=?,delivered=? WHERE observation_id=?",
                                (until, until == current["end"], row["observation_id"]))
        return command

    def pending_commands(self, limit=10):
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ContextError("Invalid recovery page size")
        return [json.loads(row[0]) for row in self.db.execute("SELECT command_json FROM cc_commands WHERE completed=0 ORDER BY rowid LIMIT ?", (limit,))]

    def channel_error(self, request_id: str | None, code: str, diagnostic: str):
        bounded(code, 128, "channel error code")
        if request_id is not None:
            self.request(request_id)
        with self.transaction():
            self.db.execute("INSERT INTO cc_channel_errors(request_id,code,diagnostic,time) VALUES(?,?,?,?)",
                            (request_id, code, diagnostic, time.time()))

    def record_usage(self, request_id: str, *, prompt_tokens: int, completion_tokens: int, elapsed_seconds: float):
        self.request(request_id)
        if any(type(n) is not int or n < 0 for n in (prompt_tokens, completion_tokens)):
            raise ContextError("Server token usage must be nonnegative integers")
        from .contracts import finite
        finite(elapsed_seconds, "elapsed_seconds")
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens, "elapsed_seconds": elapsed_seconds,
                 "source": "server_usage"}
        with self.transaction():
            old = self.db.execute("SELECT usage_json FROM cc_usage WHERE request_id=?", (request_id,)).fetchone()
            if old and old[0] != canonical(usage):
                raise ContextError("Usage for a request is immutable; retries need distinct request IDs")
            self.db.execute("INSERT OR IGNORE INTO cc_usage VALUES(?,?)", (request_id, canonical(usage)))

    def excluded_ids(self, request_id, *, after_seq=0, limit=100):
        if type(after_seq) is not int or after_seq < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ContextError("Invalid exclusion page")
        request = self.request(request_id)
        audit = json.loads(request["audit_json"])
        manifest = audit["exclusions"]
        snapshot = json.loads(self.db.execute("SELECT manifest_json FROM cc_snapshots WHERE snapshot_id=?", (request["snapshot_id"],)).fetchone()[0])
        if snapshot["allowed_scopes"] != sorted(self.allowed_scopes):
            raise AccessDenied("Exclusion manifest belongs to different access permissions")
        selected = audit["selected_ids"]
        clause, params = self._scope_sql()
        not_in = f"AND record_id NOT IN ({','.join('?' for _ in selected)})" if selected else ""
        rows = self.db.execute(f"SELECT seq,record_id FROM cc_records WHERE seq>? AND seq<=? AND {clause} {not_in} ORDER BY seq LIMIT ?",
                               (after_seq, manifest["max_record_seq"], *params, *selected, limit + 1)).fetchall()
        return {"ids": [row["record_id"] for row in rows[:limit]],
                "next_after_seq": rows[limit - 1]["seq"] if len(rows) > limit else None}

    def handle_read(self, arguments: dict):
        """Adapter branch for analyze_experiments; cannot write state or execute code."""
        if set(arguments) == {"context_read"}:
            value = arguments["context_read"]
            if not isinstance(value, dict):
                raise ContextError("context_read must be an object")
            try:
                return self.read_source(**value)
            except TypeError as error:
                raise ContextError("Invalid context_read arguments") from error
        if set(arguments) == {"context_search"}:
            value = arguments["context_search"]
            if not isinstance(value, dict) or set(value) - {"filters", "cursor", "limit"}:
                raise ContextError("Invalid context_search arguments")
            return self.search(**value)
        raise ContextError("Expected exactly one context_read or context_search route")
