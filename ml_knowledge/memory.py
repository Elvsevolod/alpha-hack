"""Append-only evidence history and context-dependent active knowledge."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from .core import canonical, digest
from .runtime import applicability_for

DEPENDENCIES = ("data_version", "task_hash", "feature_roles", "validation_hash", "environment", "pipeline_hash")
TYPES = {"documentation_fact", "project_recommendation", "agent_hypothesis", "experiment_result"}


class KnowledgeMemory:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, record_id TEXT, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS progress (fingerprint TEXT PRIMARY KEY);
        """)

    def close(self):
        self.db.close()

    def _event(self, kind, record_id, body):
        self.db.execute("INSERT INTO events(kind,record_id,body) VALUES (?,?,?)", (kind, record_id, canonical(body)))

    def _save(self, record):
        if record["type"] not in TYPES or not record.get("claim"):
            raise ValueError("A typed substantive claim is required")
        if len(record["claim"]) > 2400:
            raise ValueError("Active excerpts are limited to 2400 characters; keep full sources in the corpus")
        identifier = record["id"]
        previous = self.db.execute("SELECT body FROM records WHERE id=?", (identifier,)).fetchone()
        if previous:
            if previous[0] == canonical(record):
                return identifier
            raise ValueError("Evidence IDs are immutable; use a new ID and explicitly link the contradiction")
        with self.db:
            for (raw,) in self.db.execute("SELECT body FROM records"):
                other = json.loads(raw)
                if record.get("topic") and other.get("topic") == record["topic"] and other["claim"] != record["claim"]:
                    self._event("possible_contradiction", identifier, {"other_id": other["id"], "required_check": "Compare source, scope and conditions; neither claim is silently replaced"})
            self.db.execute("INSERT INTO records VALUES (?,?)", (identifier, canonical(record)))
            self._event("recorded", identifier, record)
            fingerprint = digest({k: record[k] for k in ("type", "claim", "source", "dependencies")})
            self.db.execute("INSERT OR IGNORE INTO progress VALUES (?)", (fingerprint,))
        return identifier

    def fact(self, store, reference_id, start, end, *, conditions=(), contraindications=(), topic=None):
        doc = store.document(reference_id)
        if not doc or not 0 <= start < end <= len(doc["text"]):
            raise ValueError("Valid source document and range required")
        if doc["type"] != "documentation":
            raise ValueError("Project recommendations and hypotheses cannot be promoted to documentation facts")
        claim = doc["text"][start:end]
        source = {"reference_id": doc["id"], "doc_sha256": doc["doc_sha256"], "start": start, "end": end,
                  "excerpt_sha256": digest(claim), "library": doc["library"], "version": doc["version"], "corpus": store.version}
        record = {"type": "documentation_fact", "claim": claim, "source": source, "dependencies": {},
                  "conditions": list(dict.fromkeys([*doc.get("conditions", []), *conditions])),
                  "contraindications": list(dict.fromkeys([*doc.get("warnings", []), *contraindications])),
                  "verification_scope": "Exact claim only; added conditions are host policy, not automatically verified source text",
                  "verification": "source_verified", "applicability": "unknown", "topic": topic}
        record["id"] = digest(record)
        return self._save(record)

    def hypothesis(self, claim, *, observation, expected_effect, test, context, topic=None):
        if not all((observation, expected_effect, test)):
            raise ValueError("Hypotheses need an observation, expected effect and a test")
        record = {"type": "agent_hypothesis", "claim": claim, "source": {"kind": "agent", "observation": observation,
                  "expected_effect": expected_effect, "test": test}, "dependencies": self._dependencies(context),
                  "conditions": [], "contraindications": [], "verification": "unchecked", "applicability": "unknown", "topic": topic}
        record["id"] = digest(record)
        return self._save(record)

    def recommendation(self, store, reference_id, *, context):
        doc = store.document(reference_id if reference_id.startswith("method:") else "method:" + reference_id)
        if not doc or doc["type"] != "method":
            raise ValueError("Registered method card required")
        payload = doc["payload"]
        references = payload.get("references", [])
        supporting = [store.document(reference) for reference in references]
        if any(source is None for source in supporting):
            raise ValueError("Recommendation has an unresolved source")
        status, reasons = applicability_for(doc, context)
        record = {"type": "project_recommendation", "claim": payload["hypothesis"],
                  "source": {"reference_id": doc["id"], "doc_sha256": doc["doc_sha256"], "version": doc["version"],
                             "corpus": store.version, "references": references},
                  "dependencies": self._dependencies(context), "conditions": doc.get("conditions", []),
                  "contraindications": doc.get("warnings", []), "verification": "source_verified",
                  "applicability": status, "applicability_reasons": reasons, "topic": doc["id"]}
        record["source"]["supporting_sources"] = [{"reference_id": source["id"], "doc_sha256": source["doc_sha256"],
                                                    "library": source["library"], "version": source["version"]}
                                                   for source in supporting]
        record["id"] = digest(record)
        return self._save(record)

    def experiment(self, experiment_id, *, controller_lookup, context):
        # The host supplies a lookup of executed, controller-verified results;
        # a model's claim or reference ID is not an execution attestation.
        result = controller_lookup(experiment_id)
        if not result or result.get("experiment_id") != experiment_id or result.get("status") not in {"completed", "failed", "timeout", "cancelled"}:
            raise ValueError("Recorded controller outcome required")
        source = {key: result[key] for key in ("experiment_id", "status", "metrics", "artifacts", "configuration_sha256") if key in result}
        record = {"type": "experiment_result", "claim": "Observed experiment outcome: " + canonical(source),
                  "source": source, "dependencies": self._dependencies(context), "conditions": [],
                  "contraindications": ["A timeout does not establish low predictive quality."] if source["status"] == "timeout" else [],
                  "verification": "source_verified", "applicability": "unknown", "topic": experiment_id}
        record["id"] = digest(record)
        return self._save(record)

    @staticmethod
    def _dependencies(context):
        missing = set(DEPENDENCIES) - set(context)
        if missing:
            raise ValueError("Explicit dependency fingerprints required: " + ", ".join(sorted(missing)))
        return {key: digest(context[key]) for key in DEPENDENCIES}

    def current(self, store, context):
        results = []
        conflicts = {}
        for record_id, raw in self.db.execute("SELECT record_id,body FROM events WHERE kind='possible_contradiction'"):
            conflict = json.loads(raw)
            other_id = conflict["other_id"]
            conflicts.setdefault(record_id, set()).add(other_id)
            conflicts.setdefault(other_id, set()).add(record_id)
        for (raw,) in self.db.execute("SELECT body FROM records ORDER BY id"):
            record = json.loads(raw)
            source = record["source"]
            reasons = []
            required_reading = []
            content_complete = True
            if "reference_id" in source:
                doc = store.document(source["reference_id"])
                if (not doc or doc["doc_sha256"] != source["doc_sha256"] or doc["version"] != source["version"]
                        or "library" in source and doc["library"] != source["library"]):
                    record["verification"] = "stale_source"
                    reasons.append("source_or_version_changed")
                elif record["type"] == "documentation_fact" and (doc["text"][source["start"]:source["end"]] != record["claim"] or digest(record["claim"]) != source["excerpt_sha256"]):
                    record["verification"] = "stale_source"
                    reasons.append("source_excerpt_changed")
                else:
                    record["verification"] = "source_verified"
                if doc and record["type"] == "documentation_fact":
                    record["conditions"] = list(dict.fromkeys([*doc.get("conditions", []), *record["conditions"]]))
                    record["contraindications"] = list(dict.fromkeys([*doc.get("warnings", []), *record["contraindications"]]))
                    content_complete = doc.get("content_status") == "complete"
                    required_reading = [{"id": doc["id"], "doc_sha256": doc["doc_sha256"], **span}
                                        for span in doc.get("required_spans", [])
                                        if not source["start"] <= span["start"] < span["end"] <= source["end"]]
                    if not content_complete:
                        required_reading.append({"id": doc["id"], "reason": "Source has unresolved sections"})
                if doc and record["type"] == "project_recommendation":
                    record["applicability"], record["applicability_reasons"] = applicability_for(doc, context)
                    if "supporting_sources" not in source:
                        record["verification"] = "stale_source"
                        reasons.append("supporting_source_fingerprints_missing")
                    for evidence in source.get("supporting_sources", []):
                        support = store.document(evidence["reference_id"])
                        if not support or any(support[key] != evidence[key] for key in ("doc_sha256", "library", "version")):
                            record["verification"] = "stale_source"
                            reasons.append("supporting_source_changed:" + evidence["reference_id"])
            changed = [key for key, value in record["dependencies"].items() if key not in context or digest(context[key]) != value]
            if changed:
                record["applicability"] = "stale_context"
                reasons.extend("changed:" + key for key in changed)
            record["stale_reasons"] = reasons
            record["required_reading"] = required_reading
            record["conflicts"] = sorted(conflicts.get(record["id"], ()))
            record["answer_usable"] = (record["verification"] == "source_verified" and content_complete
                                       and not required_reading and not record["conflicts"]
                                       and record["applicability"] not in {"stale_context", "inapplicable"})
            results.append(record)
            # Revalidation history is idempotent and does not count as progress.
            state = {"verification": record["verification"], "applicability": record["applicability"], "reasons": reasons,
                     "answer_usable": record["answer_usable"], "required_reading": required_reading,
                     "conflicts": record["conflicts"]}
            previous = self.db.execute("SELECT body FROM events WHERE record_id=? AND kind='revalidated' ORDER BY seq DESC LIMIT 1", (record["id"],)).fetchone()
            if not previous or previous[0] != canonical(state):
                with self.db:
                    self._event("revalidated", record["id"], state)
        return results

    def active(self, store, context):
        return [r for r in self.current(store, context) if r["verification"] != "stale_source" and r["applicability"] not in {"stale_context", "inapplicable"}]

    def provided(self, record_id, *, packet_sha256):
        with self.db:
            self._event("provided", record_id, {"packet_sha256": packet_sha256})

    def referenced(self, record_id, *, operation_id):
        with self.db:
            self._event("referenced", record_id, {"operation_id": operation_id})

    def outcome(self, record_id, *, decision, evidence, controller_verify):
        if decision not in {"applied", "rejected"}:
            raise ValueError("Decision must be applied or rejected")
        if not controller_verify(decision, evidence):
            raise ValueError("Observable execution/configuration or justified rejection evidence required")
        with self.db:
            self._event(decision, record_id, evidence)

    @property
    def progress_count(self):
        return self.db.execute("SELECT COUNT(*) FROM progress").fetchone()[0]

    def history(self):
        return [{"seq": seq, "kind": kind, "record_id": identifier, "body": json.loads(body)} for seq, kind, identifier, body in self.db.execute("SELECT * FROM events ORDER BY seq")]

