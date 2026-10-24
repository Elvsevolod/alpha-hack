"""Versioned reads and indexed discovery over controller-permitted run data."""
from __future__ import annotations

import json
import re

from context_compressor import Record, Source

from .knowledge_tools import CursorError, continuation, cursor, position
from .util import canonical, digest, file_hash


def words(text):
    return set(re.findall(r"\w+", text.casefold()))


class RunTools:
    def __init__(self, controller):
        self.c = controller
        self.store = controller.context_memory.store
        self.store.db.executescript("""
            CREATE TABLE IF NOT EXISTS api2_text(record_id TEXT PRIMARY KEY, terms TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS api2_pages(id TEXT PRIMARY KEY, content TEXT NOT NULL);
        """)

    def index(self):
        self.c.context_memory.index_experiments()
        rows = self.store.db.execute("""SELECT r.record_id,r.payload,s.content FROM cc_records r
            JOIN cc_sources s ON s.source_id=json_extract(r.payload,'$.source_id')
                AND s.version=json_extract(r.payload,'$.source_version') AND s.section=''
            WHERE NOT EXISTS (SELECT 1 FROM api2_text t WHERE t.record_id=r.record_id)""").fetchall()
        with self.store.transaction():
            for row in rows:
                # Only published permitted sources enter this index. Never inspect task files or worker artifacts.
                self.store._check_scope(json.loads(row["payload"])["data_scope"])
                self.store.db.execute("INSERT INTO api2_text VALUES (?,?)", (row["record_id"],
                    canonical(sorted(words(row["content"].decode("utf-8"))))))

    def discover(self, args, *, catalog=False):
        tool = "list_run_records" if catalog else "search_run_records"
        query = "" if catalog else args.query
        filters = args.filters.model_dump(exclude_none=True)
        if filters.get("kind") == "note":
            filters["kind"] = "research_note"
        if not catalog and not query.strip() and not filters:
            return {"status": "incompatible", "error_code": "query_or_filters_required",
                "recovery": {"tool": "list_run_records", "arguments": {}}}
        self.index()
        binding = {"tool": tool, "run": self.c.run_id, "request": digest({"query": query, "filters": args.filters.model_dump(exclude_none=True)}),
            "scopes": digest(sorted(self.store.allowed_scopes))}
        clauses, values = self.store._scope_sql()
        clauses, values = [clauses], list(values)
        for name, value in filters.items():
            clauses.append("r." + name + "=?")
            values.append(value)
        sought, identifiers = words(query), []
        for row in self.store.db.execute("SELECT r.record_id,t.terms FROM cc_records r JOIN api2_text t ON t.record_id=r.record_id WHERE "
                + " AND ".join(clauses) + " ORDER BY r.seq,r.record_id", values):
            if sought <= set(json.loads(row["terms"])):
                identifiers.append(row["record_id"])
        key = digest({**binding, "ids": identifiers})
        with self.store.transaction():
            self.store.db.execute("INSERT OR IGNORE INTO api2_pages VALUES (?,?)", (key, canonical(identifiers)))
        offset = 0
        result = {"status": "success" if identifiers else "no_matches", "records": [], "total": len(identifiers),
            "catalog_browse": catalog, "next_cursor": None, "continuation": None, "complete": True}
        for index, identifier in enumerate(identifiers[offset:offset + args.limit], offset):
            card = self.store.record(identifier)
            candidate = {**result, "records": result["records"] + [card]}
            token = self.store._cursor({"format": 2, "binding": binding, "snapshot": key, "position": index + 1}) if index + 1 < len(identifiers) else None
            candidate.update(next_cursor=token, complete=token is None, continuation=continuation(tool, args, token))
            if len(canonical(candidate)) > args.max_chars - 900:
                break
            result = candidate
        if identifiers[offset:] and not result["records"]:
            return {"status": "budget_too_small", "error_code": "record_card_exceeds_budget"}
        return result

    def search(self, args, *, catalog=False):
        # The signed run cursor includes an immutable snapshot, unlike corpus cursors.
        if args.cursor:
            obj = self.store._decode_cursor(args.cursor)
            tool = "list_run_records" if catalog else "search_run_records"
            binding = {"tool": tool, "run": self.c.run_id,
                "request": digest({"query": "" if catalog else args.query, "filters": args.filters.model_dump(exclude_none=True)}),
                "scopes": digest(sorted(self.store.allowed_scopes))}
            if obj.get("format") != 2 or obj.get("binding") != binding:
                raise CursorError("invalid_cursor: changed run query, filters or permissions")
            row = self.store.db.execute("SELECT content FROM api2_pages WHERE id=?", (obj.get("snapshot"),)).fetchone()
            if row is None:
                raise CursorError("stale_cursor: snapshot unavailable")
            ids = json.loads(row[0]); offset = obj.get("position")
            if type(offset) is not int or not 0 <= offset <= len(ids):
                raise CursorError("invalid_cursor: position")
            result = {"status": "success" if ids else "no_matches", "records": [], "total": len(ids),
                "catalog_browse": catalog, "next_cursor": None, "continuation": None, "complete": True}
            for index, identifier in enumerate(ids[offset:offset + args.limit], offset):
                card = self.store.record(identifier)
                token = self.store._cursor({**obj, "position": index + 1}) if index + 1 < len(ids) else None
                candidate = {**result, "records": result["records"] + [card], "next_cursor": token,
                    "continuation": continuation(tool, args, token), "complete": token is None}
                if len(canonical(candidate)) > args.max_chars - 900:
                    break
                result = candidate
            if ids[offset:] and not result["records"]:
                return {"status": "budget_too_small", "error_code": "record_card_exceeds_budget"}
            return result
        return self.discover(args, catalog=catalog)

    def page(self, tool, args, content, scope, identity):
        source = Source("page:" + digest(identity), digest(content), content, scope, origin=tool)
        self.store.put_source(source)
        binding = {"tool": tool, "run": self.c.run_id, "source": source.source_id, "version": source.version,
            "request": digest(args.model_dump(exclude={"cursor", "max_chars", "knowledge_gap"}, mode="json"))}
        offset = position(args.cursor, binding)
        if offset > len(content):
            raise CursorError("invalid_cursor: source position")
        end = min(len(content), offset + max(1, args.max_chars - 2400))
        token = cursor(binding, end) if end < len(content) else None
        return {"status": "success", "content": content[offset:end], "complete": token is None,
            "source": {"id": source.source_id, "version": source.version, "scope": scope, "sha256": digest(content)},
            "range": {"unit": "unicode_characters", "start": offset, "end": end},
            "next_cursor": token, "continuation": continuation(tool, args, token)}

    def dataset(self, args):
        value, identity = self.c.eda.report(args, self.c.plan)
        analysis_status = value.get("status", "available")
        status = analysis_status if analysis_status in {"technical_failure", "budget_too_small", "incompatible"} else "success"
        if len(canonical(value)) <= args.max_chars - 900 and not args.cursor:
            return {"status": status, "analysis_status": analysis_status, "delivery_status": "complete",
                "report": value, "scope": args.scope, "complete": True}
        result = self.page("get_dataset_report", args, canonical(value), args.scope, identity)
        result.update(status=status, analysis_status=analysis_status,
            delivery_status="complete" if result["complete"] else "partial")
        return result

    def read(self, args):
        self.index()
        if args.source:
            value = self.store.read_source(**args.source.model_dump())
            return {"status": "success", **value}
        records = []
        for identifier in args.ids:
            record = self.store.record(identifier)
            if not record:
                return {"status": "unknown_id", "error_code": "unknown_run_record", "id": identifier}
            self.store._check_scope(record["data_scope"])
            row = self.store.db.execute("SELECT content FROM cc_sources WHERE source_id=? AND version=? AND section='' AND scope=?",
                (record["source_id"], record["source_version"], record["data_scope"])).fetchone()
            if row is None:
                return {"status": "missing_version", "id": identifier}
            records.append({"id": identifier, "record": record, "content": row[0].decode()})
        if args.include_code:
            identifier = args.ids[0]
            sha = self.c.journal.get("proposed_code_sources", {}).get(identifier.removeprefix("operation:"))
            if not sha:
                try:
                    sha = self.c.lookup(identifier)[0].pipeline.custom_hash
                except ValueError:
                    pass
            if not sha:
                return {"status": "unknown_id", "error_code": "saved_code_unavailable"}
            path = self.c.workdir / "code" / (sha + ".py")
            if not path.is_file() or file_hash(path) != sha:
                return {"status": "technical_failure", "error_code": "code_checksum_mismatch"}
            return self.page("read_run_records", args, path.read_text(), "code", {"code_sha256": sha})
        content = canonical(records)
        if len(content) <= args.max_chars - 900 and not args.cursor:
            return {"status": "success", "records": records, "complete": True}
        return self.page("read_run_records", args, content, "controller", {"records": args.ids, "content": digest(content)})


def revision_result(c, operation_id, previous, error=None):
    revision = c.journal.get("protocol_revision")
    if not revision or revision["operation_id"] != operation_id:
        return {"status": "incompatible", "decision": "rejected", "revision_status": "rejected",
            "previous_protocol_hash": previous, "proposed_protocol_hash": None, "active_protocol_hash": c.plan.hash,
            "reevaluated_ids": [], "pending_ids": [], "outcomes": [], "reason": error}
    from .contracts import ValidationPlan
    proposed = ValidationPlan.model_validate(revision["proposed"]).hash
    states = {"pending": "reevaluating", "committed": "committed", "completed": "completed", "aborted": "interrupted_before_commit"}
    unknown = any(i["status"] == "outcome_unknown" for i in revision["candidates"])
    status = "outcome_unknown" if unknown else states[revision["status"]]
    return {"status": "success" if status == "completed" else "outcome_unknown" if unknown else "interrupted",
        "decision": "admitted", "revision_status": status, "previous_protocol_hash": ValidationPlan.model_validate(revision["old"]).hash,
        "proposed_protocol_hash": proposed, "active_protocol_hash": c.plan.hash,
        "reevaluated_ids": [i["spec"]["experiment_id"] for i in revision["candidates"] if i["status"] == "completed"],
        "pending_ids": [i["spec"]["experiment_id"] for i in revision["candidates"] if i["status"] != "completed"],
        "outcomes": [{"id": i["spec"]["experiment_id"], "status": i["status"]} for i in revision["candidates"]],
        "reason": error or revision.get("reason")}
