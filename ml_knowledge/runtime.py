"""Offline reference tool with bounded Unicode responses and bound cursors."""
from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote
import zlib

from .core import RANKING_VERSION, canonical, digest, file_hash, read_json
from .lexical import ALIASES, tokens


@dataclass(frozen=True)
class ReferenceArgs:
    mode: str = "search"
    query: str = ""
    reference_id: str | None = None
    library: str | None = None
    version: str | None = None
    cursor: str | int | None = None
    limit: int = 5
    max_chars: int = 10000
    knowledge_gap: str | None = None

    def __post_init__(self):
        if self.mode not in {"search", "api", "read", "methods", "recipes", "search_space"}:
            raise ValueError("Unknown get_reference mode")
        if isinstance(self.limit, bool) or not 1 <= self.limit <= 5 or not isinstance(self.limit, int):
            raise ValueError("limit must be 1..5")
        if isinstance(self.max_chars, bool) or not isinstance(self.max_chars, int) or not 100 <= self.max_chars <= 20000:
            raise ValueError("max_chars must be 100..20000 Unicode characters")
        if not isinstance(self.query, str) or any(x is not None and not isinstance(x, str) for x in (self.reference_id, self.library, self.version)):
            raise ValueError("Query and filters must be strings")


REFERENCE_SCHEMA = {"type": "object", "properties": {
    "mode": {"enum": ["search", "api", "read", "methods", "recipes", "search_space"]},
    "query": {"type": "string"}, "reference_id": {"type": ["string", "null"]},
    "library": {"type": ["string", "null"]}, "version": {"type": ["string", "null"]},
    "cursor": {"anyOf": [{"type": "string"}, {"const": 0}, {"type": "null"}]},
    "limit": {"type": "integer", "minimum": 1, "maximum": 5},
    "max_chars": {"type": "integer", "minimum": 100, "maximum": 20000},
    "knowledge_gap": {"type": ["string", "null"]}}, "additionalProperties": False}


class IntegrityError(ValueError):
    pass


class ReferenceStore:
    def __init__(self, root, *, use_fts=True, verify=True):
        self.root = Path(root)
        self.use_fts = use_fts
        self.lock = {}
        self.version = "unavailable"
        self.disabled_reason = None
        self.db = None
        try:
            self.lock = read_json(self.root / "knowledge.lock.json")
            self.version = self.lock["corpus"]
            if verify:
                self.verify()
            self.db = sqlite3.connect("file:" + quote(str((self.root / "knowledge.sqlite").resolve())) + "?mode=ro&immutable=1", uri=True)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA cache_size=-16384")
            self.db.execute("PRAGMA mmap_size=0")
            if self.db.execute("SELECT value FROM meta WHERE key='corpus'").fetchone()[0] != self.version:
                raise IntegrityError("Corpus identity mismatch")
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
            self.disabled_reason = type(exc).__name__

    def close(self):
        if self.db:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def verify(self, requirements=None):
        if requirements and file_hash(requirements) != self.lock["requirements_sha256"]:
            raise IntegrityError("Requirements mismatch")
        for name, expected in self.lock["files"].items():
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root.resolve()) or not path.is_file() or file_hash(path) != expected:
                raise IntegrityError("Corpus file checksum mismatch")
        return {"status": "verified", "corpus": self.version}

    def document(self, identifier):
        if self.disabled_reason:
            raise IntegrityError("Corpus is disabled")
        alias = self.db.execute("SELECT doc FROM aliases WHERE alias=?", (identifier,)).fetchone()
        if alias:
            identifier = alias[0]
        row = self.db.execute("SELECT d.meta,t.text FROM docs d JOIN texts t ON d.sha=t.sha WHERE d.id=?", (identifier,)).fetchone()
        if row is None:
            return None
        record = json.loads(row["meta"])
        text = row["text"]
        if digest(text) != record["doc_sha256"]:
            raise IntegrityError("Document checksum mismatch")
        return {**record, "text": text}

    def _binding(self, args, context):
        return {"corpus": self.version, "ranking": RANKING_VERSION, "context": digest(context or {}),
                "request": digest({k: getattr(args, k) for k in ("mode", "query", "reference_id", "library", "version")})}

    def _cursor(self, binding, position, doc_sha=None):
        payload = canonical({**binding, "position": position, "doc": doc_sha}).encode()
        key = hashlib.sha256(("mlkb-cursor:" + self.version).encode()).digest()
        signature = hmac.new(key, payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(payload + signature).decode().rstrip("=")

    def _position(self, cursor, binding):
        if cursor is None or type(cursor) is int and cursor == 0:
            return 0, None
        if not isinstance(cursor, str) or len(cursor) > 4096:
            return None, "invalid_cursor"
        try:
            raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
            data, signature = raw[:-32], raw[-32:]
            obj = json.loads(data)
            # Check the embedded corpus using its own signature first, so a
            # valid old token is stale, whereas a corrupted token is invalid.
            key = hashlib.sha256(("mlkb-cursor:" + obj["corpus"]).encode()).digest()
            if not hmac.compare_digest(signature, hmac.new(key, data, hashlib.sha256).digest()):
                return None, "invalid_cursor"
            if obj["request"] != binding["request"]:
                return None, "invalid_cursor"
            if any(obj[x] != binding[x] for x in ("corpus", "ranking", "context")):
                return None, "stale_cursor"
            if type(obj["position"]) is not int or obj["position"] < 0:
                return None, "invalid_cursor"
            return obj["position"], obj.get("doc")
        except (ValueError, KeyError, TypeError):
            return None, "invalid_cursor"

    @staticmethod
    def _bounded(value, budget):
        if len(canonical(value)) <= budget:
            return value
        return {"status": "budget_too_small", "min_chars": len(canonical(value))}

    def query(self, args=None, *, context=None, **kwargs):
        if args is None:
            args = ReferenceArgs(**kwargs)
        elif isinstance(args, dict):
            args = ReferenceArgs(**args)
        elif not isinstance(args, ReferenceArgs):
            args = ReferenceArgs(**{name: getattr(args, name) for name in ReferenceArgs.__dataclass_fields__ if hasattr(args, name)})
        if self.disabled_reason:
            return {"status": "integrity_error", "reason": self.disabled_reason}
        try:
            result = self._query(args, context or {})
        except (IntegrityError, sqlite3.DatabaseError):
            result = {"status": "integrity_error"}
        assert len(canonical(result)) <= args.max_chars
        return result

    get_reference = query

    def _filters(self, args):
        clauses, values = [], []
        versions = []
        if args.library:
            versions.append("v.library=?"); values.append(args.library)
        if args.version:
            versions.append("v.version=?"); values.append(args.version)
        if versions:
            clauses.append("EXISTS (SELECT 1 FROM libraries v WHERE v.doc=d.id AND " + " AND ".join(versions) + ")")
        kind = {"methods": "method", "recipes": "recipe", "search_space": "search_space"}.get(args.mode)
        if kind:
            clauses.append("d.kind=?"); values.append(kind)
        return " AND ".join(clauses) or "1", values

    def _query(self, args, context):
        binding = self._binding(args, context)
        position, token_info = self._position(args.cursor, binding)
        if position is None:
            return {"status": token_info}
        where, values = self._filters(args)
        if args.version and not self.db.execute("SELECT 1 FROM docs d WHERE " + where + " LIMIT 1", values).fetchone():
            return {"status": "missing_version"}
        if args.reference_id:
            identifier = args.reference_id
            alias = self.db.execute("SELECT doc FROM aliases WHERE alias=?", (identifier,)).fetchone()
            if alias:
                identifier = alias[0]
            prefix = {"methods": "method:", "recipes": "recipe:", "search_space": "search_space:"}.get(args.mode)
            if prefix and not identifier.startswith(prefix):
                identifier = prefix + identifier
            # Legacy method IDs are unambiguous in read mode too.
            if args.mode == "read" and not self.db.execute("SELECT 1 FROM docs WHERE id=?", (identifier,)).fetchone():
                identifier = "method:" + identifier
            row = self.db.execute("SELECT d.id FROM docs d WHERE " + where + " AND d.id=?", [*values, identifier]).fetchone()
            if not row:
                return {"status": "not_found"}
            return self._read(row[0], args, position, token_info, binding, context)
        if args.mode == "read":
            return {"status": "not_found", "reason": "reference_id_required"}
        if args.mode == "api":
            query = args.query.strip()
            rows = self.db.execute("SELECT s.symbol,d.id FROM symbols s JOIN docs d ON d.id=s.doc WHERE " + where + " AND s.symbol=? ORDER BY d.id", [*values, query]).fetchall()
            if not rows:
                rows = self.db.execute("SELECT s.symbol,d.id FROM symbols s JOIN docs d ON d.id=s.doc WHERE " + where + " AND s.short=? ORDER BY s.symbol,d.id", [*values, query.casefold()]).fetchall()
            options = sorted({r["symbol"] for r in rows})
            if len(options) > 1:
                return self._bounded({"status": "ambiguous_symbol", "symbols": options}, args.max_chars)
            if not rows:
                return {"status": "not_found"}
            return self._read(rows[0]["id"], args, position, token_info, binding, context)
        hits = self.search(args)
        if not hits:
            return {"status": "not_found"}
        if position >= len(hits):
            return {"status": "invalid_cursor"}
        entries = []
        # Results remain full coherent fragments; callers can read an oversized
        # result with a larger budget or use a smaller character-page read.
        base = {"status": "available", "corpus": self.version, "entries": entries,
                "next_cursor": None, "answer_usable": True}
        minimum = None
        for hit in hits[position:position + args.limit]:
            entry = self.entry(hit, context=context)
            next_pos = position + len(entries) + 1
            candidate = {**base, "entries": entries + [entry],
                         "next_cursor": self._cursor(binding, next_pos) if next_pos < len(hits) else None}
            candidate["answer_usable"] = all(e["answer_usable"] for e in candidate["entries"])
            if len(canonical(candidate)) > args.max_chars:
                minimum = len(canonical(candidate))
                break
            entries.append(entry)
            base = candidate
        if not entries:
            # A bounded locator is a partial response, never a ready answer.
            hit = hits[position]
            locator = {"status": "partial", "answer_usable": False, "reason": "fragment_exceeds_budget",
                       "required_read": {"reference_id": hit["doc"], "mode": "read"}, "min_chars": minimum}
            return self._bounded(locator, args.max_chars)
        base["entries"] = entries
        if base["next_cursor"] or not base["answer_usable"]:
            base["status"] = "partial"
        return base

    def _read(self, identifier, args, position, token_sha, binding, context):
        record = self.document(identifier)
        text = record["text"]
        if token_sha and token_sha != record["doc_sha256"]:
            return {"status": "stale_cursor"}
        if position >= len(text) and position != 0:
            return {"status": "invalid_cursor"}
        def response(end):
            hit = {"doc": identifier, "start": position, "end": end, "text": text[position:end], "sha": digest(text[position:end])}
            entry = self.entry(hit, context=context, record=record)
            return {"status": "partial" if end < len(text) or record.get("content_status") != "complete" else "complete",
                    "corpus": self.version, "entry": entry, "answer_usable": entry["answer_usable"],
                    "next_cursor": self._cursor(binding, end, record["doc_sha256"]) if end < len(text) else None}
        # At least one Unicode character must fit or no progress is made.
        smallest = response(min(position + 1, len(text)))
        if len(canonical(smallest)) > args.max_chars:
            return {"status": "budget_too_small", "min_chars": len(canonical(smallest))}
        lo, hi = position + 1, len(text)
        if len(canonical(response(hi))) <= args.max_chars:
            return response(hi)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(canonical(response(mid))) <= args.max_chars:
                lo = mid
            else:
                hi = mid - 1
        # Never break a known warning and its condition into apparent permission.
        for span in record.get("required_spans", []):
            if span["start"] < lo < span["end"]:
                if span["start"] > position:
                    lo = span["start"]
                else:
                    needed = len(canonical(response(span["end"])))
                    return {"status": "budget_too_small", "min_chars": needed}
        return response(lo)

    def search(self, args, *, maximum=100):
        if self.disabled_reason:
            raise IntegrityError("Corpus is disabled")
        where, values = self._filters(args)
        query = args.query.casefold()
        for key, value in ALIASES.items():
            if key in query:
                query += " " + value
        terms = sorted(tokens(query))
        if not terms:
            if args.mode in {"methods", "recipes", "search_space"} and not args.query.strip():
                return [dict(row) for row in self.db.execute("SELECT f.* FROM fragments f JOIN docs d ON d.id=f.doc WHERE " + where + " ORDER BY d.id,f.start LIMIT ?", [*values, maximum])]
            return []
        candidates = {}
        exact_docs = set()
        family_docs = set()
        for symbol in re.findall(r"[A-Za-z_]\w*(?:\.\w+)+", args.query):
            for row in self.db.execute("SELECT d.id FROM symbols s JOIN docs d ON d.id=s.doc WHERE " + where + " AND s.symbol=?", [*values,symbol]):
                exact_docs.add(row[0])
        # Preserve public aliases even when many re-exports share a canonical
        # stored docstring. Short class names also expose the fitting signature.
        for name in re.findall(r'\b[A-Z][A-Za-z0-9_]+\b',args.query):
            for row in self.db.execute("SELECT s.symbol,d.id FROM symbols s JOIN docs d ON d.id=s.doc WHERE " + where + " AND s.short=?",[*values,name.casefold()]):
                exact_docs.add(row['id'])
                fit=self.db.execute('SELECT doc FROM symbols WHERE symbol=?',(row['symbol']+'.fit',)).fetchone()
                if fit:family_docs.add(fit[0])
        if self.use_fts:
            expression = " OR ".join('"' + word.replace('"', '""') + '"' for word in terms)
            try:
                sql = "SELECT f.*,bm25(fts,8,1) AS score FROM fts JOIN fragments f ON f.id=fts.rowid JOIN docs d ON d.id=f.doc WHERE fts MATCH ? AND " + where + " ORDER BY score,f.id LIMIT ?"
                for row in self.db.execute(sql, [expression, *values, maximum * 3]):
                    candidates[row["id"]] = dict(row)
            except sqlite3.OperationalError as exc:
                if "fts" not in str(exc).lower() and "module" not in str(exc).lower():
                    raise
        for identifier in sorted(exact_docs | family_docs):
            for row in self.db.execute("SELECT f.* FROM fragments f JOIN docs d ON d.id=f.doc WHERE " + where + " AND f.doc=?", [*values,identifier]):
                candidates[row["id"]] = dict(row)
        if not candidates:
            scores = Counter()
            for term in terms:
                row = self.db.execute("SELECT ids FROM postings WHERE term=?", (term,)).fetchone()
                if row:
                    scores.update(json.loads(zlib.decompress(row[0])))
            # Filter every candidate BEFORE ranking/truncation, including fallback.
            ids = sorted(scores)
            for start in range(0, len(ids), 400):
                batch = ids[start:start+400]
                sql = "SELECT f.* FROM fragments f JOIN docs d ON d.id=f.doc WHERE " + where + " AND f.id IN (" + ",".join("?" for _ in batch) + ")"
                for row in self.db.execute(sql, [*values, *batch]):
                    candidates[row["id"]] = dict(row)
        def ranking(hit):
            title_terms = tokens(hit["title"])
            content_terms = tokens(hit["text"])
            exact = args.query.casefold().strip() == hit["title"].casefold()
            score = 1000 * (exact or hit["doc"] in exact_docs) + 600 * (hit['doc'] in family_docs) + 3 * len(set(terms) & title_terms) + 4 * len(set(terms) & content_terms)
            return (-score, hit.get("score", 0), hit["doc"], hit["start"])
        selected, seen = [], set()
        for hit in sorted(candidates.values(), key=ranking):
            if hit["doc"] not in seen:
                selected.append(hit);seen.add(hit["doc"])
            if len(selected) >= maximum:
                break
        return selected

    def entry(self, hit, *, context=None, record=None):
        record = record or self.document(hit["doc"])
        fragment = hit["text"]
        if digest(fragment) != hit["sha"] or record["text"][hit["start"]:hit["end"]] != fragment:
            raise IntegrityError("Fragment checksum/range mismatch")
        conditions = record.get("conditions", [])
        warnings = record.get("warnings", [])
        # Metadata retains full known conditions even when the content is paged.
        usable = record.get("content_status") == "complete"
        missing = [span for span in record.get("required_spans", []) if not (hit["start"] <= span["start"] and span["end"] <= hit["end"])]
        usable = usable and not missing
        applicability, reasons = applicability_for(record, context or {})
        return {"id": record["id"], "type": record["type"], "library": record["library"], "version": record["version"],
                "library_versions": record.get("library_versions", [{"library":record["library"],"version":record["version"]}]),
                "source": record["source_url"], "doc_sha256": record["doc_sha256"],
                "fragment": {"id": record["id"] + ":" + str(hit["start"]) + ":" + str(hit["end"]),
                             "start": hit["start"], "end": hit["end"], "sha256": hit["sha"]},
                "content": fragment, "conditions": conditions, "warnings": warnings,
                "required_reading": missing,
                "applicability": applicability, "applicability_reasons": reasons,
                "content_status": record.get("content_status", "partial"), "answer_usable": usable}


def applicability_for(record, context):
    rules = record.get("payload", {}).get("applicability_rules", [])
    unknown, reasons = False, []
    for rule in rules:
        key, expected = rule["field"], rule["value"]
        actual = context.get(key)
        if actual is None or rule["op"] not in {"eq", "not_in"}:
            unknown = True
        elif rule["op"] == "eq" and actual != expected or rule["op"] == "not_in" and actual in expected:
            reasons.append(rule.get("reason", key + " incompatible"))
    if reasons:
        return "inapplicable", reasons
    # Reading official docs never enables an adapter. Only a host-verified
    # capability and complete machine-checkable conditions establish application.
    # Passing a negative guard does not prove the card's prose prerequisites.
    if not rules or unknown or record.get("conditions"):
        return "unknown", []
    return "applicable", []
