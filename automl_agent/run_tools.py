"""Versioned reads and indexed discovery over controller-permitted run data."""
from __future__ import annotations

import json
import re

import pandas as pd

from context_compressor import Record, Source

from .contracts import InspectArgs
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
        from .contracts import request_arguments
        request = {key: value for key, value in request_arguments(args).items() if key not in {"cursor", "max_chars", "knowledge_gap"}}
        binding = {"tool": tool, "run": self.c.run_id, "source": source.source_id, "version": source.version,
            "request": digest(request)}
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
        raw = args.model_dump(exclude={"cursor", "max_chars", "knowledge_gap", "budget_seconds", "correlation_method"})
        if args.correlation_method == "supported":
            if args.report != "correlations":
                raise ValueError("correlation_method=supported requires report=correlations")
            value = self.correlations(args)
            if value.get("status") in {"budget_too_small", "technical_failure", "outcome_unknown"}:
                return value
        elif args.report in {"transfer", "domain"}:
            value = self.transfer(args)
        else:
            value = self.c.dataset.inspect(InspectArgs(**raw), self.c.plan)
        if len(canonical(value)) <= args.max_chars - 900 and not args.cursor:
            return {"status": "success", "report": value, "scope": args.scope, "complete": True}
        return self.page("get_dataset_report", args, canonical(value), args.scope,
            {"data": self.c.dataset.source_hash, "protocol": self.c.plan.hash, "args": raw})

    def correlations(self, args):
        import time
        import numpy as np
        import pandas as pd
        from .executor import stage_frame
        from .util import file_hash, read_json
        from .profiling import MAX_CORRELATION_COLUMNS, MAX_CORRELATION_ROWS
        c = self.c
        if args.group_by:
            raise ValueError("Bounded correlations do not accept group_by")
        frame, columns = c.dataset.analysis_frame(args, c.plan)
        numeric = [name for name in columns if pd.api.types.is_numeric_dtype(frame[name])]
        if not numeric or len(numeric) > MAX_CORRELATION_COLUMNS:
            raise ValueError("Select between 1 and 30 numeric columns for bounded correlations")
        selected = frame[numeric].replace([np.inf, -np.inf], np.nan)
        if len(selected) > MAX_CORRELATION_ROWS:
            selected = selected.sample(n=MAX_CORRELATION_ROWS, random_state=c.training_seed)
        identity = digest({"analyzer": "correlations_v1", "data": c.dataset.source_hash, "task": c.task.hash, "protocol": c.plan.hash,
            "scope": args.scope, "columns": columns, "rows": list(selected.index), "seed": c.training_seed})
        directory = c.workdir / "feature_diagnostics" / ("correlations_" + identity)
        manifest = directory / "worker_result.json"
        if manifest.is_file():
            raw = read_json(manifest)
            if raw.get("status") != "completed":
                return {"status": "technical_failure", "relationship_status": "unknown", "error_code": raw.get("status"), "cached": True}
            if file_hash(directory / "correlations.json") != raw["hashes"]["correlations.json"]:
                raise ValueError("Correlation artifact checksum mismatch")
        else:
            if (directory / "request.json").exists():
                return {"status": "outcome_unknown", "relationship_status": "unknown", "error_code": "correlation_request_unconfirmed"}
            available = min(20., args.budget_seconds, c.budget.remaining - c.budget.final_reserve(c.finalist_costs()))
            if available < 5:
                return {"status": "budget_too_small", "relationship_status": "unknown"}
            started = time.monotonic()
            directory.mkdir(parents=True, exist_ok=True)
            stage_frame(directory / "correlation_frame.json", selected, deadline=min(c.budget.deadline, time.time() + available), memory_mb=c.executor.memory_mb)
            request = {"mode": "correlations", "excluded_columns": {name: "nonnumeric" for name in columns if name not in numeric}}
            grant = available - (time.monotonic() - started)
            if grant < .5:
                return {"status": "budget_too_small", "relationship_status": "unknown"}
            raw = c.executor.run(directory, request, grant)
            c.journal.event("correlations_worker", {"identity": identity, "seconds": time.monotonic() - started, "status": raw["status"]})
            if raw["status"] != "completed":
                return {"status": "technical_failure", "relationship_status": "unknown", "error_code": raw["status"]}
        return {**read_json(directory / "correlations.json"), "scope": args.scope, "rows": len(selected),
            "original_rows": len(frame), "sampled_row_hash": digest(list(selected.index)), "seed": c.training_seed,
            "data_version": c.dataset.source_hash, "protocol_hash": c.plan.hash}

    def transfer(self, args):
        from .research_evaluation import transfer_profile
        from .executor import stage_frame
        from .util import file_hash, read_json
        c = self.c
        if args.columns or args.predicate or args.group_by:
            raise ValueError("Transfer reports use the confirmed feature set, without adaptive row filters")
        if args.include_target and (args.scope != "fit" or args.report == "domain"):
            raise ValueError("Only fit-label conflict aggregates are available; domain diagnostics never receive task labels")
        fit = c.dataset.features(c.plan.fit)
        train, test = c.dataset.features(list(range(len(c.dataset.train)))), c.dataset.features(test=True)
        if args.report == "transfer":
            labels = c.dataset.train.iloc[c.plan.fit][c.task.targets] if args.include_target else pd.DataFrame(index=fit.index)
            return transfer_profile(fit, labels, c.dataset.features(c.plan.validation), train, test, c.task)
        identity = digest({"mode": "domain_v1", "data": c.dataset.source_hash, "features": c.task.feature_columns,
            "seed": c.training_seed, "budget": args.budget_seconds})
        directory = c.workdir / "feature_diagnostics" / identity
        completed = directory / "worker_result.json"
        if completed.is_file():
            raw = read_json(completed)
            if raw.get("status") != "completed":
                return {"status": "technical_failure", "error_code": raw.get("status"), "cached": True}
            if file_hash(directory / "domain_report.json") != raw["hashes"]["domain_report.json"]:
                raise ValueError("Domain diagnostic checksum mismatch")
            return read_json(directory / "domain_report.json")
        if (directory / "request.json").is_file():
            return {"status": "outcome_unknown", "error_code": "domain_diagnostic_unconfirmed"}
        available = c.budget.remaining - c.budget.final_reserve(c.finalist_costs())
        if available < 5:
            return {"status": "budget_too_small"}
        directory.mkdir(parents=True, exist_ok=True)
        stage_frame(directory / "domain_train.json", train, deadline=c.budget.deadline, memory_mb=c.executor.memory_mb)
        stage_frame(directory / "domain_test.json", test, deadline=c.budget.deadline, memory_mb=c.executor.memory_mb)
        request = {"mode": "domain_diagnostic", "cpu": c.executor.cpu, "seed": c.training_seed}
        measured = c.executor.run(directory, request, min(args.budget_seconds, available))
        if measured["status"] != "completed":
            return {"status": "technical_failure", "error_code": measured["status"]}
        report = read_json(directory / "domain_report.json")
        c.journal.event("feature_only_domain_diagnostic", {"id": identity, "seconds": measured.get("wall_seconds"), "task_labels_used": False})
        return report

    def read(self, args):
        self.index()
        if args.view == "pipeline":
            identifier = args.ids[0]
            spec, _ = self.c.lookup(identifier)
            record = self.store.record(identifier)
            result = {"status": "success", "experiment_id": identifier, "pipeline": spec.pipeline.research_recipe(),
                "recipe_hash": digest(spec.pipeline.research_recipe()),
                "evidence_ref": {"kind": "record", "id": identifier, "version": record["source_version"]},
                "complete": True, "learned_state": "relearned inside fit; not copied from the parent"}
            if spec.pipeline.engine == 'ensemble':
                result['ensemble'] = {'members': spec.pipeline.members, 'strategy': spec.pipeline.ensemble_strategy}
            if len(canonical(result)) <= args.max_chars - 900 and not args.cursor:
                return result
            return self.page("read_run_records", args, canonical(result), "registered_feedback",
                {"pipeline": spec.pipeline.hash, "experiment": identifier})
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
                return {"status": "unknown_id", "error_code": "saved_code_unavailable",
                    "reason": "Declarative models have a pipeline recipe, not submitted custom source code",
                    "recovery": {"tool": "read_run_records", "arguments": {"ids": [identifier], "view": "pipeline"}}}
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
