"""API v2 discovery/read actions over the existing immutable corpus."""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import hmac
from itertools import chain
import json
import re
import sqlite3
from types import SimpleNamespace
import zlib

from ml_knowledge.lexical import ALIASES, tokens
from ml_knowledge.runtime import ReferenceArgs, applicability_for

from .util import canonical, digest
from .tool_results import envelope

KINDS = {"documents": "documentation", "methods": "method", "recipes": "recipe", "search_spaces": "search_space"}


class CursorError(ValueError):
    pass


def cursor(binding, position):
    data = canonical({"format": 2, "binding": binding, "position": position}).encode()
    signature = hmac.new(b"alpha-tool-api-v2", data, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(signature + data).decode()


def position(token, binding):
    if token is None:
        return 0
    try:
        if len(token) > 4096:
            raise ValueError()
        raw = base64.b64decode(token, altchars=b"-_", validate=True)
        if not hmac.compare_digest(raw[:32], hmac.new(b"alpha-tool-api-v2", raw[32:], hashlib.sha256).digest()):
            raise ValueError()
        obj = json.loads(raw[32:])
        if obj["format"] != 2 or obj["binding"] != binding:
            raise ValueError()
        if type(obj["position"]) is not int or obj["position"] < 0:
            raise ValueError()
        return obj["position"]
    except (ValueError, TypeError, KeyError):
        raise CursorError("invalid_or_stale_cursor: query, filters, source, snapshot or version changed") from None


def continuation(tool, args, token):
    arguments = args.model_dump(mode="json")
    if tool == "read_knowledge" and arguments.get("view") == "document":
        arguments.pop("view")
        arguments.pop("parameter", None)
    return {"tool": tool, "arguments": {**arguments, "cursor": token}} if token else None


class KnowledgeTools:
    def __init__(self, controller):
        self.c = controller
        self.store = controller.references
        self.cache = {}

    def where(self, args):
        clauses, values = ["d.kind=?"], [KINDS[args.collection]]
        for key, value in args.filters.model_dump(exclude_none=True).items():
            clauses.append("EXISTS (SELECT 1 FROM libraries v WHERE v.doc=d.id AND v." + key + "=?)")
            values.append(value)
        # Library and version must belong to the SAME supported library tuple.
        if args.filters.library and args.filters.version:
            clauses.append("EXISTS (SELECT 1 FROM libraries v WHERE v.doc=d.id AND v.library=? AND v.version=?)")
            values.extend([args.filters.library, args.filters.version])
        return " AND ".join(clauses), values

    def binding(self, tool, args, extra=None):
        request = args.model_dump(exclude={"cursor", "max_chars", "limit", "knowledge_gap"}, mode="json")
        if request.get("view") == "document":
            request.pop("view")
            request.pop("parameter", None)
        return {"tool": tool, "corpus": self.store.version, "request": digest(request), **(extra or {})}

    def exact(self, query, where, values):
        sql = "SELECT s.symbol,d.id FROM symbols s JOIN docs d ON d.id=s.doc WHERE " + where
        rows = self.store.db.execute(sql + " AND s.symbol=? ORDER BY s.symbol,d.id", [*values, query.strip()]).fetchall()
        if not rows:
            rows = self.store.db.execute(sql + " AND s.short=? ORDER BY s.symbol,d.id", [*values, query.strip().casefold()]).fetchall()
        symbols = sorted({row["symbol"] for row in rows})
        return symbols, [{"doc": identifier, "score": 1, "matched_symbol": symbols[0] if len(symbols) == 1 else None}
            for identifier in sorted({row["id"] for row in rows})]

    def lexical(self, query, where, values):
        normalized = query.casefold()
        for key, value in ALIASES.items():
            if key in normalized:
                normalized += " " + value
        words = set(tokens(normalized))
        scores = Counter()
        for word in sorted(words):
            row = self.store.db.execute("SELECT ids FROM postings WHERE term=?", (word,)).fetchone()
            if row:
                scores.update(json.loads(zlib.decompress(row[0])))
        hits = {}
        ids = sorted(scores)
        for start in range(0, len(ids), 400):
            batch = ids[start:start + 400]
            rows = self.store.db.execute("SELECT f.id,f.doc,f.title FROM fragments f JOIN docs d ON d.id=f.doc WHERE "
                + where + " AND f.id IN (" + ",".join("?" for _ in batch) + ")", [*values, *batch])
            for row in rows:
                score = scores[row["id"]] + 3 * len(words & tokens(row["title"]))
                if score > hits.get(row["doc"], 0):
                    hits[row["doc"]] = score
        return [{"doc": identifier, "score": score} for identifier, score in sorted(hits.items(), key=lambda x: (-x[1], x[0]))]

    def fts(self, query, where, values):
        if not self.store.use_fts:
            raise sqlite3.OperationalError("FTS5 disabled")
        normalized = query.casefold()
        for key, value in ALIASES.items():
            if key in normalized:
                normalized += " " + value
        words = sorted(tokens(normalized))
        if not words:
            return []
        expression = " OR ".join('"' + word.replace('"', '""') + '"' for word in words)
        # Retain the existing corpus's symbol/term precedence before BM25.
        # Expose both scores; neither is a probability or semantic answer verdict.
        exact, family = set(), set()
        for name in re.findall(r"[A-Za-z_]\w*(?:\.\w+)+|\b[A-Z][A-Za-z0-9_]+\b", query):
            for row in self.store.db.execute("SELECT s.symbol,d.id FROM symbols s JOIN docs d ON d.id=s.doc WHERE "
                    + where + " AND (s.symbol=? OR s.short=?)", [*values, name, name.casefold()]):
                exact.add(row["id"])
                fit = self.store.db.execute("SELECT doc FROM symbols WHERE symbol=?", (row["symbol"] + ".fit",)).fetchone()
                if fit:
                    family.add(fit[0])
        rows = self.store.db.execute("SELECT f.doc,f.title,f.text,bm25(fts,8,1) AS score FROM fts JOIN fragments f ON f.id=fts.rowid "
            "JOIN docs d ON d.id=f.doc WHERE fts MATCH ? AND " + where + " ORDER BY score,d.id,f.start", [expression, *values])
        # Public aliases may resolve to a canonical signature whose text contains
        # none of the alias's terms (for example a shared fit implementation).
        # Preserve the original corpus's explicit symbol/family recall path.
        linked = (row for identifier in sorted(exact | family)
            for row in self.store.db.execute("SELECT f.doc,f.title,f.text,NULL AS score FROM fragments f "
                "JOIN docs d ON d.id=f.doc WHERE " + where + " AND f.doc=? ORDER BY f.start", [*values, identifier]))
        hits = {}
        terms = set(words)
        for row in chain(rows, linked):
            coverage = (1000 * (row["doc"] in exact or query.casefold().strip() == row["title"].casefold())
                + 600 * (row["doc"] in family) + 3 * len(terms & tokens(row["title"])) + 4 * len(terms & tokens(row["text"])))
            hit = {"doc": row["doc"], "score": row["score"], "match_strength": coverage}
            if row["doc"] not in hits or (-coverage, row["score"] or 0) < (-hits[row["doc"]]["match_strength"], hits[row["doc"]]["score"] or 0):
                hits[row["doc"]] = hit
        return sorted(hits.values(), key=lambda h: (-h["match_strength"], h["score"] or 0, h["doc"]))

    def card(self, hit, collection, rank, method):
        row = self.store.db.execute("SELECT meta,title FROM docs WHERE id=?", (hit["doc"],)).fetchone()
        meta = json.loads(row["meta"])
        fragment = self.store.db.execute("SELECT substr(text,1,220) FROM fragments WHERE doc=? ORDER BY start LIMIT 1", (hit["doc"],)).fetchone()
        applicability, reasons = applicability_for(meta, self.store.context)
        ref = {"kind": "knowledge", "collection": collection, "id": hit["doc"], "version": self.store.version}
        return {"id": hit["doc"], "reference": ref, "title": row["title"], "library": meta["library"],
            "version": meta["version"], "doc_sha256": meta["doc_sha256"], "rank": rank,
            "score": hit.get("score"), "score_direction": "lower_is_better" if method == "fts" else "higher_is_better" if method != "catalog" else None,
            "match": method, "matched_symbol": hit.get("matched_symbol"), "snippet": fragment[0] if fragment else "",
            "match_strength": hit.get("match_strength"),
            "ranking": "existing_symbol_and_term_precedence_then_bm25" if method == "fts" else method,
            "applicability": applicability, "applicability_reasons": reasons,
            "documentation_status": meta.get("content_status", "partial"), "answer_confirmed": False,
            "read": {"tool": "read_knowledge", "arguments": {"collection": collection, "id": hit["doc"], "corpus_version": self.store.version}}}

    def discover(self, args, *, catalog=False):
        if args.collection == "actions":
            from .action_cards import discover
            return discover(self.c, args, catalog)
        tool = "list_knowledge" if catalog else "search_knowledge"
        if self.store.disabled_reason:
            return {"status": "technical_failure", "error_code": "corpus_unavailable"}
        if not catalog and args.collection != "documents" and args.strategy in {"fts", "exact_symbol"}:
            return {"status": "incompatible", "error_code": "unsupported_strategy_for_collection"}
        if self.store.excluded_by_scope(" ".join(filter(None, (getattr(args, "query", ""), args.filters.library)))):
            return {"status": "no_matches", "collection": args.collection, "entries": [], "total": 0,
                "scope_reason": "outside_classical_ml_scope", "answer_confirmed": False, "complete": True}
        where, values = self.where(args)
        binding = self.binding(tool, args, {"fts_enabled": self.store.use_fts})
        offset = position(args.cursor, binding)
        if args.filters.version and not self.store.db.execute("SELECT 1 FROM docs d WHERE " + where + " LIMIT 1", values).fetchone():
            return {"status": "missing_version", "total": 0, "entries": [], "corpus": self.store.version}
        key = digest(binding)
        if key not in self.cache:
            fallback, symbols, hits = None, [], []
            method = "catalog" if catalog else args.strategy
            if catalog:
                hits = [{"doc": row[0]} for row in self.store.db.execute("SELECT d.id FROM docs d WHERE " + where + " ORDER BY d.id", values)]
            else:
                if args.collection == "documents" and args.strategy in {"auto", "exact_symbol"}:
                    symbols, hits = self.exact(args.query, where, values)
                    if hits or args.strategy == "exact_symbol":
                        method = "exact_symbol"
                if not hits and args.strategy in {"fts", "auto"} and args.collection == "documents":
                    try:
                        hits = self.fts(args.query, where, values)
                        method = "fts"
                        if not hits:
                            fallback = "fts_no_matches"
                    except sqlite3.OperationalError as error:
                        if "fts" not in str(error).lower() and "module" not in str(error).lower():
                            raise
                        if args.strategy == "fts":
                            return {"status": "technical_failure", "error_code": "fts_unavailable", "method": "fts"}
                        fallback = "fts_unavailable"
                if args.strategy == "lexical" or not hits and args.strategy == "auto":
                    hits, method = self.lexical(args.query, where, values), "lexical"
                    if args.collection != "documents":
                        fallback = "catalog_uses_lexical"
            if len(self.cache) >= 8:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = (hits, method, fallback, symbols)
        hits, method, fallback, symbols = self.cache[key]
        if offset > len(hits):
            raise CursorError("invalid_cursor: position outside results")
        status = "ambiguous" if len(symbols) > 1 else "success" if hits else "no_matches"
        result = {"status": status, "collection": args.collection, "corpus": self.store.version,
            "method": method, "fallback_reason": fallback, "catalog_browse": catalog,
            "answer_confirmed": False, "total": len(hits), "entries": [], "next_cursor": None,
            "complete": True, "continuation": None}
        # Cover the direct ToolCommand limit as well as the narrower wire ID.
        operation = SimpleNamespace(tool=tool, operation_id="0" * 100)
        for index, hit in enumerate(hits[offset:offset + args.limit], offset):
            card = self.card(hit, args.collection, index + 1, method)
            candidate = {**result, "entries": result["entries"] + [card]}
            token = cursor(binding, index + 1) if index + 1 < len(hits) else None
            candidate.update(next_cursor=token, complete=token is None, continuation=continuation(tool, args, token))
            if len(canonical(envelope(self.c, operation, candidate))) > args.max_chars:
                break
            result = candidate
        if hits[offset:] and not result["entries"]:
            return {"status": "budget_too_small", "error_code": "card_exceeds_budget", "minimum_max_chars": min(12000, args.max_chars * 2)}
        self.c.journal.event("knowledge_discovery", {"tool": tool, "query": getattr(args, "query", None),
            "filters": args.filters.model_dump(), "requested_strategy": getattr(args, "strategy", None),
            "method": method, "fallback_reason": fallback, "page_start": offset,
            "found_ids": [e["id"] for e in result["entries"]], "corpus": self.store.version})
        return result

    def read(self, args):
        if args.view == "parameters" and args.collection != "documents":
            return {"status": "incompatible", "error_code": "parameters_require_documentation"}
        if args.collection == "actions":
            from .action_cards import read
            return read(self.c, args)
        if args.corpus_version != self.store.version:
            return {"status": "missing_version", "error_code": "corpus_version_unavailable"}
        record = self.store.document(args.id)
        if not record:
            return {"status": "unknown_id", "error_code": "unknown_knowledge_id"}
        if record["type"] != KINDS[args.collection]:
            return {"status": "incompatible", "error_code": "collection_mismatch"}
        if args.view == "parameters":
            return self.read_parameters(args, record)
        binding = self.binding("read_knowledge", args, {"doc_sha256": record["doc_sha256"]})
        offset = position(args.cursor, binding)
        # Reuse verified fragments and protected warning spans of the existing reader.
        old_args = ReferenceArgs(mode="read", reference_id=args.id, max_chars=max(100, args.max_chars - 1600))
        old_binding = self.store._binding(old_args, self.store.context)
        result = self.store._read(record["id"], old_args, offset, record["doc_sha256"], old_binding, self.store.context)
        if result.get("entry"):
            entry = result["entry"]
            end = entry["fragment"]["end"]
            token = cursor(binding, end) if end < len(record["text"]) else None
            result.update(status="success", complete=token is None and record.get("content_status") == "complete",
                next_cursor=token, continuation=continuation("read_knowledge", args, token),
                reference={"kind": "knowledge", "collection": args.collection, "id": record["id"], "version": self.store.version},
                range={"unit": "unicode_characters", "start": offset, "end": end})
            self.c.journal.event("knowledge_fragment_read", {"id": record["id"], "corpus": self.store.version,
                "range": result["range"], "fragment_sha256": entry["fragment"]["sha256"]})
        return result

    def read_parameters(self, args, record):
        from .parameter_docs import VERSION, parameter_section, adapter_contract
        from ml_knowledge.core import digest as corpus_digest
        if args.collection != "documents":
            return {"status":"incompatible", "error_code":"parameters_require_documentation",
                "recovery":"Search documents by exact library API symbol, then read its parameters view."}
        section = parameter_section(record,args.parameter)
        if section is None:
            return {"status":"no_matches", "error_code":"parameter_section_unavailable",
                "source_status":record.get("content_status"), "recovery":"Read the full versioned document; no parameter schema was inferred."}
        if section["status"] != "available":
            return {"status":"no_matches", "error_code":"unknown_parameter",
                "parameter":args.parameter, "recovery":"Browse view=parameters without a parameter filter."}
        contract = adapter_contract(record,section["available_names"])
        binding = {"parameters":digest(self.binding("read_knowledge", args, {"doc_sha256":record["doc_sha256"],
            "parameter_view_version":VERSION,"adapter_hash":digest(contract)}))}
        offset = position(args.cursor,binding) if args.cursor else section["start"]
        if not section["start"] <= offset <= section["end"]:
            raise CursorError("invalid_cursor: position outside the parameter section")
        def result_at(end):
            content = record["text"][offset:end]
            entry = self.store.entry({"doc":record["id"],"text":content,"start":offset,"end":end,
                "sha":corpus_digest(content)},context=self.store.context,record=record)
            entry["parameter_view"] = {"version":VERSION,"parameter":args.parameter,
                "signature":section["signature"],"adapter":contract,
                "detail_status":section["detail_status"],
                **({"declarations":section["declarations"],"forwards_kwargs":section["forwards_kwargs"],
                    "further_documentation":{"tool":"search_knowledge","arguments":{
                        "collection":"documents","query":record["title"]+" "+(args.parameter or "parameters"),
                        "filters":{"library":record["library"],"version":record["version"]}}}}
                    if section["detail_status"]=="signature_only" else {}),
                "semantics":"Exact frozen documentation. Defaults and constraints are in the declarations and descriptions; values are not executable code."}
            if section["detail_status"] == "signature_only":
                entry["answer_usable"] = False
            token = cursor(binding,end) if end < section["end"] else None
            return {"status":"success","entry":entry,"view":"parameters",
                "complete":token is None and record.get("content_status")=="complete",
                "answer_confirmed":False,"next_cursor":token,"continuation":continuation("read_knowledge",args,token),
                "reference":{"kind":"knowledge","collection":"documents","id":record["id"],"version":self.store.version},
                "range":{"unit":"unicode_characters","start":offset,"end":end}}
        # Fit the actual public envelope; page boundaries remain original-document offsets.
        operation = SimpleNamespace(tool="read_knowledge",operation_id="0"*100)
        low, high = offset, min(section["end"],offset+args.max_chars)
        best = None
        while low <= high:
            middle = (low+high)//2
            candidate = result_at(middle)
            if len(canonical(envelope(self.c,operation,candidate))) <= args.max_chars:
                best = candidate
                low = middle+1
            else:
                high = middle-1
        if best is None or best["range"]["end"] == offset and offset < section["end"]:
            return {"status":"budget_too_small","error_code":"parameter_metadata_exceeds_budget",
                "recovery":"Use a larger max_chars or request one exact parameter."}
        self.c.journal.event("knowledge_parameters_read",{"id":record["id"],"corpus":self.store.version,
            "range":best["range"],"parameter":args.parameter,"view_version":VERSION})
        return best

