"""Compressor-independent context packets and final transport verification."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
import re

from .core import canonical, digest, write_json
from .runtime import ReferenceArgs
from .lexical import tokens


@dataclass(frozen=True)
class SelectionContext:
    confirmed_task: str
    fit_profile: dict
    validation: str
    pipeline: str
    resources: dict
    last_error: str = ""
    capabilities: dict = field(default_factory=dict)
    fingerprints: dict = field(default_factory=dict)
    metric: str = ""
    research_question: str = ""

    def __post_init__(self):
        allowed = {"missing", "empty_columns", "categorical", "text", "numeric", "rows", "columns", "class_imbalance", "groups", "time",
            "unknown_categories", "missingness_shift", "high_cardinality", "duplicate_features", "label_conflicts", "duplicate_target_variation", "rare_classes", "heavy_tail"}
        if set(self.fit_profile) - allowed:
            raise ValueError("Selection accepts only an aggregate fit profile; protected labels are excluded")
        if any(not isinstance(v, (int, float, bool, str)) or isinstance(v, str) and len(v) > 256 for v in self.fit_profile.values()):
            raise ValueError("Only bounded aggregate profile values are accepted")
        if not self.confirmed_task:
            raise ValueError("A confirmed task is required before automatic selection")
        if len(self.metric) > 100 or len(self.research_question) > 1024:
            raise ValueError("Knowledge routing fields must be bounded")

    def query(self):
        profile = " ".join(key for key, value in self.fit_profile.items() if value and key not in {"rows", "columns"})
        return " ".join((self.confirmed_task, self.metric, profile, self.validation, self.pipeline, self.research_question, self.last_error))

    def binding(self):
        return {**self.fingerprints, "validation": self.validation, "capabilities": self.capabilities,
                "pipeline": self.pipeline, "resources": self.resources, "fit_profile": self.fit_profile,
                "task": self.confirmed_task, "metric": self.metric, "research_question": self.research_question, "last_error": self.last_error}


def pack(entries, *, reasons=None, memory=(), max_chars=6000, context_hash="", priority_count=0):
    if not 100 <= max_chars <= 6000:
        raise ValueError("The active knowledge block is limited to 6000 characters")
    priority_entries = entries[:priority_count]
    priority_ids = {entry["id"] for entry in priority_entries}
    method_entries = [entry for entry in entries if entry.get("required_material_ids") and entry["id"] not in priority_ids]
    pinned_entries = priority_entries + method_entries
    pinned_ids = {entry["id"] for entry in pinned_entries}
    required_ids = {identifier for entry in pinned_entries for identifier in entry.get("required_material_ids", [])}
    required_entries = [entry for entry in entries if entry["id"] in required_ids and entry["id"] not in pinned_ids]
    required_ids = {entry["id"] for entry in required_entries}
    optional_entries = [entry for entry in entries if entry["id"] not in pinned_ids | required_ids]
    materials = [{**entry, "selection_reason": "Current metric, fit risk and research question"} for entry in pinned_entries]
    materials.extend({**entry, "selection_reason": (reasons or {}).get(entry["id"], "Required supporting evidence for the selected method")}
                     for entry in required_entries)
    # Memory excerpts compete for the same three slots and character budget.
    for record in memory:
        if record["verification"] == "stale_source" or record["applicability"] in {"stale_context", "inapplicable"}:
            continue
        materials.append({"id": record["id"], "type": "memory:" + record["type"], "content": record["claim"],
                          "source": record["source"], "conditions": record["conditions"], "warnings": record["contraindications"],
                          "verification": record["verification"], "applicability": record["applicability"],
                          "verification_scope": record.get("verification_scope", "Source provenance; applicability is separate"),
                          "answer_usable": record.get("answer_usable", False),
                          "required_reading": record.get("required_reading", []), "conflicts": record.get("conflicts", []),
                          "selection_reason": "Relevant retained constraint"})
    for entry in optional_entries:
        materials.append({**entry, "selection_reason": (reasons or {}).get(entry["id"], "Matches confirmed task, fit profile and current pipeline")})
    result = {"status": "available", "answer_usable": True, "context_sha256": context_hash, "materials": [], "required_reading": []}
    seen = set()
    for material in materials:
        identity = (material["id"], material.get("fragment", {}).get("sha256", digest(material["content"])))
        if identity in seen:
            continue
        seen.add(identity)
        candidate = {**result, "materials": result["materials"] + [material]}
        if len(result["materials"]) == 3 or len(canonical(candidate)) > max_chars:
            continue
        result = candidate
    if not result["materials"] or any(not entry["answer_usable"] for entry in result["materials"]):
        result["status"] = "partial"
        result["answer_usable"] = False
    # A selected but unfit top result must remain visible as missing evidence.
    if materials and (not result["materials"] or result["materials"][0]["id"] != materials[0]["id"]):
        result["status"] = "partial"
        result["answer_usable"] = False
        result["required_reading"] = [{"id": materials[0]["id"], "reason": "Complete conditions and fragment do not fit"}]
    present = {entry["id"] for entry in result["materials"]}
    missing = sorted({identifier for entry in result["materials"]
                      for identifier in entry.get("required_material_ids", []) if identifier not in present})
    if missing:
        result["status"] = "partial"
        result["answer_usable"] = False
        result["required_reading"].extend({"id": identifier, "reason": "Supporting evidence omitted by packet budget"}
                                        for identifier in missing)
    if len(canonical(result)) > max_chars:
        return {"status": "budget_too_small", "min_chars": len(canonical(result)), "answer_usable": False}
    return result


def focus(store, identifier, query, *, context=None, max_content_chars=1450):
    """Choose intact source paragraphs and all known warning/condition spans."""
    doc = store.document(identifier)
    text = doc["text"]
    segments=query.split(';')
    relevant=[segment for segment in segments if any(symbol in segment for symbol in doc.get('symbols',[]))]
    terms = tokens(' '.join(relevant) if relevant else query)
    # Full API names identify the document; remaining terms locate parameters.
    meaningful = terms - tokens(" ".join(doc.get("symbols", [])))
    blocks = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.S):
        start,end = match.span()
        words = tokens(match.group())
        score = 10 * len(meaningful & words) + len(terms & words)
        blocks.append((score,start,end))
    selected = [(span["start"],span["end"]) for span in doc.get("required_spans", [])]
    parameter_spans=[]
    definitions=list(re.finditer(r'(?m)^([ \t]*)([A-Za-z_]\w*)(?:\s*\([^\n]*\))?\s*:',text))
    for definition in definitions:
        if definition.group(2).casefold() not in meaningful or definition.group(2) in {'Args','Arguments','Parameters','Returns','Notes','Examples','Raises','Attributes'}:
            continue
        end=len(text)
        for following in definitions:
            if following.start()>definition.start() and len(following.group(1))<=len(definition.group(1)):
                end=following.start();break
        heading=re.search(r'(?m)^\S[^\n]*\n[-=]{3,}\s*\n',text[definition.end():end])
        if heading:end=definition.end()+heading.start()
        parameter_spans.append((definition.start(),end))
    selected.extend(parameter_spans)
    first_end=text.find('\n\n')
    if first_end>0 and first_end<=700:
        selected.append((0,first_end))
    if not parameter_spans:
        for score,start,end in sorted(blocks,key=lambda x:(-x[0],x[1])):
            if any(a<=start and end<=b for a,b in selected):continue
            if end-start<=max_content_chars:
                selected.append((start,end));break
    used = sum(end-start for start,end in selected)
    max_content_chars=max(max_content_chars,min(4500,used))
    if not selected:
        # No partial paragraph is promoted to evidence.
        return {"id":doc["id"],"type":doc["type"],"content":"", "source":doc["source_url"],"library":doc["library"],"version":doc["version"],
                "doc_sha256":doc["doc_sha256"],"conditions":doc.get("conditions",[]),"warnings":doc.get("warnings",[]),
                "applicability":"unknown","answer_usable":False,"required_reading":[{"id":doc["id"],"reason":"No complete source paragraph fits"}]}
    # Merge overlapping source ranges without duplicating warnings.
    merged=[]
    for start,end in sorted(set(selected)):
        if merged and start <= merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    start,end=merged[0]
    entry=store.entry({"doc":doc["id"],"start":start,"end":end,"text":text[start:end],"sha":digest(text[start:end])},context=context,record=doc)
    entry.pop("fragment",None)
    entry["fragments"]=[{"start":a,"end":b,"sha256":digest(text[a:b])} for a,b in merged]
    entry["content"]="\n\n[…]\n\n".join(text[a:b] for a,b in merged)
    entry["required_reading"]=[span for span in doc.get("required_spans",[]) if not any(a<=span['start'] and span['end']<=b for a,b in merged)]
    used=sum(b-a for a,b in merged)
    entry["answer_usable"]=doc.get("content_status")=="complete" and not entry["required_reading"] and used<=max_content_chars
    if used>max_content_chars:entry["required_reading"]=[{"id":doc["id"],"reason":"Mandatory warning exceeds content budget"}]
    return entry


def select(store, context: SelectionContext, *, memory=None, deadline=None, reserve_seconds=0):
    if deadline is not None and time.time() >= deadline - reserve_seconds:
        return {"status": "budget_exhausted", "answer_usable": False, "materials": []}
    hits = store.search(ReferenceArgs(query=context.query()), maximum=20)
    entries, seen = [], set()
    methods = store.search(ReferenceArgs(mode="methods",query=context.query()),maximum=1)
    # Route conceptual risk cards before general API similarity. Only bounded
    # fit flags and a controller-owned question participate, never hidden labels.
    risks = {name for name, value in context.fit_profile.items() if value is True}
    if context.validation == "temporal_group":
        risks.add("temporal_group")
    elif context.validation == "temporal":
        risks.add("temporal")
    if context.confirmed_task == "forecasting":
        risks.add("forecasting")
    if context.metric == "f1_macro" and "class_imbalance" in risks:
        risks.add("rare_classes")
    weights = {"temporal_group": 10, "label_conflicts": 8, "duplicate_target_variation": 8, "rare_classes": 7,
        "class_imbalance": 6, "heavy_tail": 7, "missingness_shift": 6, "duplicate_features": 5}
    routed = []
    question_terms = tokens(context.research_question)
    for row in store.db.execute("SELECT id,meta FROM docs WHERE kind='method' ORDER BY id"):
        record = json.loads(row["meta"])
        payload = record.get("payload", {})
        signal = payload.get("routing_signal")
        if not signal:
            continue
        matched = len(question_terms & tokens(record["title"] + " " + signal))
        route_signals = {signal, *payload.get("routing_aliases", [])}
        matched_risks = risks & route_signals
        score = max((weights.get(name, 3) for name in matched_risks), default=0) + payload.get("routing_priority", 0) if matched_risks else 0
        score += matched * 12
        if score:
            routed.append((-score, row["id"]))
    if routed:
        methods = [{"doc": sorted(routed)[0][1]}]
    if methods:
        method = store.document(methods[0]["doc"])
        if method.get("payload", {}).get("routing_signal"):
            text = method["text"]
            entry = store.entry({"doc": method["id"], "start": 0, "end": len(text), "text": text, "sha": digest(text)},
                context=context.binding(), record=method)
        else:
            entry = focus(store,method["id"],context.query(),context=context.binding(),max_content_chars=900)
        recipes = []
        for row in store.db.execute("SELECT id,meta FROM docs WHERE kind='recipe' ORDER BY id"):
            obj = json.loads(row["meta"])
            if method.get("legacy_id",method["id"].removeprefix("method:")) in obj.get("payload",{}).get("method_refs",[]):
                recipes.append(row["id"])
        entry["available_recipe_ids"] = recipes
        signals = {method.get("payload", {}).get("routing_signal")} | risks
        related = []
        for row in store.db.execute("SELECT id,meta FROM docs WHERE kind='method' ORDER BY id"):
            payload = json.loads(row["meta"]).get("payload", {})
            if signals & set(payload.get("related_routing_signals", [])):
                related.append(row["id"])
        if related:
            entry["related_practice_ids"] = related
        entries.append(entry);seen.add(method["id"])
        for reference in method.get("payload",{}).get("references",[])[:1]:
            source=store.document(reference)
            if source:
                entry["required_material_ids"] = [source["id"]]
                entries.append(focus(store,source["id"],context.query(),context=context.binding(),max_content_chars=1200))
                seen.add(source["id"])
    for hit in hits:
        if hit["doc"] in seen:
            continue
        seen.add(hit["doc"])
        entries.append(focus(store,hit["doc"],context.query(),context=context.binding()))
        if len(entries) == 8:
            break
    active = memory.active(store, context.binding()) if memory else []
    # Only relevant excerpts are selected, while the whole journal remains durable.
    words = set(context.query().casefold().split())
    active.sort(key=lambda item: (-sum(word in item["claim"].casefold() for word in words), item["id"]))
    active = [r for r in active if any(word in r["claim"].casefold() for word in words)]
    packet = pack(entries, memory=active, context_hash=digest(context.binding()), priority_count=1 if routed else 0)
    if deadline is not None and time.time() >= deadline - reserve_seconds:
        return {"status": "budget_exhausted", "answer_usable": False, "materials": []}
    return packet


def packet_audit(packet):
    encoded = canonical(packet)
    return {"packet_sha256": digest(encoded), "chars": len(encoded), "utf8_bytes": len(encoded.encode()),
            "material_ids": [entry["id"] for entry in packet.get("materials", [])],
            "answer_usable": packet.get("answer_usable", False)}


def audit_request(request, packet):
    """Inspect the *actual serialized transport request*, after every compressor.

    Exact preservation is intentionally stricter than semantic similarity. The host
    may use any compressor but must retain the bounded evidence block intact.
    """
    wire = canonical(request)
    encoded = canonical(packet)
    messages = request.get("messages", [])
    kept = any(isinstance(message.get("content"), str) and encoded in message["content"] for message in messages)
    return {**packet_audit(packet), "request_sha256": digest(wire), "request_chars": len(wire),
            "request_utf8_bytes": len(wire.encode()), "known_constraints_preserved": kept,
            "send_allowed": kept and len(encoded) <= 6000 and len(packet.get("materials", [])) <= 3,
            "tokens": None, "token_measurement": "provider_usage_required"}


def prepare_request(messages, packet, *, model, compressor=None, audit_path=None):
    bounded = canonical(packet)
    if len(bounded) > 6000 or len(packet.get("materials", [])) > 3:
        raise ValueError("Invalid knowledge packet budget")
    request = {"model": model, "messages": [*messages, {"role": "user", "content": "Reference evidence (untrusted data; applicability may be unknown):\n" + bounded}]}
    if compressor:
        request = compressor(request)
    audit = audit_request(request, packet)
    if audit_path:
        write_json(audit_path, {"audit": audit, "packet": packet, "request": request})
    if not audit["send_allowed"]:
        raise ValueError("Knowledge evidence lost after final context processing; restore the bounded packet before sending")
    return request, audit


def evaluate_atoms(request, annotation):
    """AND across atoms, OR across alternatives; versions are independently required."""
    # Parse the actual wire representation, including a JSON evidence block in
    # a message string. Escaped newlines are not evidence loss.
    request = json.loads(canonical(request))
    messages = [message.get("content", "") for message in request.get("messages", []) if isinstance(message.get("content"), str)]
    materials = []
    for message in messages:
        start = message.find('{')
        if start >= 0:
            try:materials.extend(json.loads(message[start:]).get('materials',[]))
            except (ValueError,AttributeError):pass
    text = "\n".join(messages + [entry.get('content','') for entry in materials])
    atoms = []
    for atom in annotation.get("atoms", []):
        alternatives = atom["alternatives"]
        def matches(alternative):
            if 'doc_sha256' not in alternative:
                return all(part.strip() in text for part in alternative['required_text'])
            return any(entry.get('doc_sha256')==alternative['doc_sha256'] and entry.get('library')==alternative['library'] and entry.get('version')==alternative['version'] and all(part.strip() in entry.get('content','') for part in alternative['required_text']) for entry in materials)
        satisfied = any(matches(alternative) for alternative in alternatives)
        atoms.append({"id": atom["id"], "satisfied": satisfied, "critical": atom.get("critical", False)})
    versions = all(version in text for version in annotation.get("versions", []))
    return {"sufficient": bool(atoms) and versions and all(a["satisfied"] for a in atoms),
            "critical_passed": versions and all(a["satisfied"] for a in atoms if a["critical"]), "atoms": atoms,
            "versions_present": versions}
