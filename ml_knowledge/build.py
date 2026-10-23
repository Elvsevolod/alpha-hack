"""Deterministic offline corpus builder. No network calls or agent imports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import tempfile
import zipfile
import zlib

from .core import (FORMAT_VERSION, NORMALIZER_VERSION, CHUNKER_VERSION, canonical,
                   digest, file_hash, read_json, validate_manifest, write_json)
from .normalize import Resolver, normalize, python_docs, required_spans

from .lexical import ALIASES, tokens


def chunks(text, maximum=2400):
    """Lossless offsets, preferring paragraph boundaries; oversize paragraphs stay whole.

    Reads can page through arbitrary characters. For retrieval an oversized code,
    warning or table stays a single addressable fragment rather than being cut.
    """
    start = 0
    ends = [m.end() for m in re.finditer(r"\n\s*\n", text)] + [len(text)]
    last = 0
    for end in ends:
        if end - start > maximum and last > start:
            yield start, last
            start = last
        last = end
    if start < len(text):
        yield start, len(text)


def base_record(identifier, text, entry, snapshot, source, kind, symbols=()):
    return {"id": identifier, "type": kind, "title": symbols[0] if symbols else Path(source).stem,
            "library": entry["package"], "version": entry["version"], "symbols": list(symbols),
            "source_url": (entry["repository"] + "/blob/" + snapshot["commit"] + "/" + source) if snapshot["commit"] else snapshot["source_url"],
            "source_commit": snapshot["commit"], "license": entry["license"],
            "version_evidence": {"tag": entry["tag"], "commit": snapshot["commit"], "kind": "exact_release_repository"},
            "text": normalize(text), "conditions": [], "warnings": [], "applicability": "unknown",
            "content_status": "complete", "retrieved_at": snapshot["retrieved_at"]}


def freeze(root):
    """Inventory first, then normalise exactly those frozen source bytes."""
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest, root / "requirements.txt")
    input_files, inventory, api_by_package = {}, [], {}
    builder_hashes = {name: file_hash(root/name) for name in ("build.py","normalize.py","lexical.py","manifest.json")}
    for entry in manifest["sources"]:
        package = entry["package"]
        snapshot_path = root / "sources" / package / "snapshot.json"
        if not snapshot_path.exists():
            inventory.append({"library": package, "id": "repository", "status": "unavailable", "reason": "exact_repository_snapshot_missing"})
            continue
        snap = read_json(snapshot_path)
        if digest(entry) != snap["manifest_entry_sha256"]:
            raise ValueError("Source manifest changed after acquisition: " + package)
        input_files[str(snapshot_path.relative_to(root))] = file_hash(snapshot_path)
        for item in snap["files"]:
            path = root / "sources" / package / "tree" / item["path"]
            if file_hash(path) != item["sha256"]:
                raise ValueError("Source integrity mismatch: " + str(path))
            input_files[str(path.relative_to(root))] = item["sha256"]
            inventory.append({"library": package, "id": item["path"], "status": "included", "source_sha256": item["sha256"]})
        for item in snap["excluded"]:
            # Binary/source-only paths are inventoried as scope exclusions.
            inventory.append({"library": package, **item, "id": item["path"]})
        api_path = root / "sources" / "installed" / (package + ".jsonl")
        api = {}
        if api_path.exists():
            input_files[str(api_path.relative_to(root))] = file_hash(api_path)
            for line in api_path.read_text().splitlines():
                obj = json.loads(line)
                if digest(obj["text"]) != obj["sha256"]:
                    raise ValueError("Installed docstring hash mismatch")
                api[obj["symbol"]] = obj
                inventory.append({"library": package, "id": "api:" + obj["symbol"], "status": "included", "source_sha256": obj["sha256"]})
            report = api_path.with_suffix(".report.json")
            input_files[str(report.relative_to(root))] = file_hash(report)
            for error in read_json(report)["errors"]:
                inventory.append({"library": package, "id": "module:" + error["module"], "status": "unavailable", "reason": error["reason"]})
        else:
            inventory.append({"library": package, "id": "installed_public_api", "status": "unavailable", "reason": "exact_package_export_missing"})
        api_by_package[package] = api
    expected = {"manifest_sha256": digest(manifest), "items": sorted(inventory, key=lambda x: (x["library"], x["id"]))}
    frozen = root / "frozen"
    frozen.mkdir(exist_ok=True)
    write_json(frozen / "expected.json", expected)
    records, expansion = {}, []
    notices = []
    for entry in manifest["sources"]:
        package = entry["package"]
        path = root / "sources" / package / "snapshot.json"
        if not path.exists():
            continue
        snap = read_json(path)
        tree = root / "sources" / package / "tree"
        api = api_by_package.get(package, {})
        substitutions = {}
        preset = tree / "catboost/docs/en/presets.yaml"
        if preset.exists():
            import yaml  # allowed build dependency; never used by the runtime
            substitutions = yaml.safe_load(preset.read_text()).get("default", {})
        for source in snap["files"]:
            if any(word in Path(source["path"]).name.upper() for word in ("LICENSE", "COPYING", "NOTICE")):
                notices.append(package + " " + entry["version"] + " — " + source["path"] + "\nSHA256: " + source["sha256"] + "\n\n" + (tree/source["path"]).read_text(errors="strict"))
        # Re-exports share one text/fragment set; every old API ID remains an alias.
        api_groups = defaultdict(list)
        for symbol, obj in sorted(api.items()):
            api_groups[digest(obj["text"][len(symbol):])].append(symbol)
        for group in sorted(api_groups.values(), key=lambda names: min(names)):
            symbol = min(group, key=lambda name: (sum(part.startswith("_") for part in name.split(".")), name.count("."), len(name), name))
            obj = api[symbol]
            identifier = "api_" + symbol.replace(".", "_")
            record = base_record(identifier, obj["text"], entry, snap, "", "documentation", group)
            record["aliases"] = ["api_" + name.replace(".", "_") for name in group if name != symbol]
            record.update(title=symbol, source_url=(entry["repository"] + "/tree/" + snap["commit"]) if snap["commit"] else snap["source_url"],
                          source_kind="installed_exact_release_docstring", original_sha256=obj["sha256"],
                          version_evidence={"kind": "installed_exact_release_docstring", "version": entry["version"],
                                            "export_sha256": input_files["sources/installed/" + package + ".jsonl"]})
            records[identifier] = record
        for item in snap["files"]:
            relative = item["path"]
            source = tree / relative
            if source.suffix in {".md", ".mdx", ".rst", ".txt", ".ipynb", ".rstinc", ".html"}:
                resolver = Resolver(tree, api, substitutions)
                text = resolver.read(relative)
                identifier = "doc_" + package + "_" + digest(relative)[:16]
                record = base_record(identifier, text, entry, snap, relative, "documentation")
                record.update(original_sha256=item["sha256"], source_kind="exact_release_document",
                              includes=sorted(resolver.origins), unresolved=resolver.gaps,
                              content_status="partial" if resolver.gaps else "complete")
                records[identifier] = record
                if source.suffix == ".html":
                    record["source_url"] = entry["documentation_root"] + relative.removeprefix("docs/")
                    record["version_evidence"] = {"kind": "official_archive_explicit_patch_title", "archives": snap["archives"], "version": entry["version"]}
                for gap in resolver.gaps:
                    expansion.append({"library": package, "document": identifier, **gap})
            elif source.suffix == ".py":
                for prefix, module in entry["modules"].items():
                    if relative.startswith(prefix):
                        suffix = relative[len(prefix):].removesuffix(".py").replace("/", ".")
                        if suffix.endswith(".__init__"):
                            suffix = suffix[:-9]
                        if suffix == "__init__":
                            suffix = ""
                        module += ("." + suffix) if suffix else ""
                        for obj in python_docs(source.read_text(encoding="utf-8-sig"), module):
                            identifier = "api_" + obj["symbol"].replace(".", "_")
                            if identifier in records:
                                continue
                            record = base_record(identifier, obj["text"], entry, snap, relative, "documentation", [obj["symbol"]])
                            record.update(original_sha256=item["sha256"], source_kind="exact_release_source_docstring",
                                          source_lines=[obj["line"], obj["end_line"]])
                            records[identifier] = record
                        break
                else:
                    if any(relative.startswith(prefix) for prefix in entry["docs"]):
                        identifier = "doc_" + package + "_" + digest(relative)[:16]
                        records[identifier] = base_record(identifier, "```python\n" + source.read_text(encoding="utf-8-sig") + "\n```", entry, snap, relative, "documentation")
                        records[identifier].update(original_sha256=item["sha256"], source_kind="official_example_source")
    # Preserve public IDs used by existing callers; exact newly exported APIs win.
    baseline_path = root / "baseline" / "before.zip"
    input_files["baseline/before.zip"] = file_hash(baseline_path)
    allowed = {e["package"]: e["version"] for e in manifest["sources"]}
    with zipfile.ZipFile(baseline_path) as z:
        legacy = z.read("knowledge_runtime/references/records.jsonl").decode()
        for line in legacy.splitlines():
            old = json.loads(line)
            if old["doc_id"] in records or allowed.get(old["library"]) != old["doc_version"] or old["version_match"] != "exact_release":
                continue
            text = z.read("knowledge_runtime/" + old["path"]).decode()
            if digest(text.encode()) != old["normalized_sha256"]:
                raise ValueError("Baseline document hash mismatch")
            records[old["doc_id"]] = {"id": old["doc_id"], "type": "documentation", "title": old["section_path"],
                "library": old["library"], "version": old["doc_version"], "source_url": old["source_url"],
                "source_commit": old.get("source_commit"), "symbols": old.get("api_symbols", []), "text": normalize(text),
                "license": old["license"], "source_kind": old["source_kind"], "original_sha256": old["original_sha256"],
                "version_evidence": {"kind": "frozen_baseline_exact_record", "snapshot_sha256": input_files["baseline/before.zip"]},
                "conditions": [], "warnings": [], "applicability": "unknown", "content_status": old["content_status"]}
    aliases = {alias for record in records.values() for alias in record.get("aliases", [])}
    for alias in aliases:
        # Newly exported aliases take precedence over redundant baseline copies.
        records.pop(alias, None)
    for name, kind in (("methods", "method"), ("recipes", "recipe"), ("search_spaces", "search_space")):
        for path in sorted((root / "catalogs" / name).glob("*.json")):
            obj = read_json(path)
            input_files[str(path.relative_to(root))] = file_hash(path)
            identifier = obj["id"]
            # Recipe and search-space IDs overlap; mode disambiguates them.
            key = kind + ":" + identifier
            record = {"id": key, "legacy_id": identifier, "type": kind, "title": obj.get("title", identifier),
                      "library": "project", "version": str(obj["version"]), "symbols": [], "text": canonical(obj),
                      "source_url": "project:" + name + "/" + identifier, "source_kind": "project_recommendation",
                      "license": "project", "conditions": obj.get("conditions", []), "warnings": [obj["constraints"]] if obj.get("constraints") else [],
                      "applicability": "unknown", "content_status": "complete", "payload": obj}
            records[key] = record
    contract = "Resource contract: all auxiliary fits, search, inference, control and frozen refit share one absolute deadline. Reserve finalization time. A timeout is evidence of resource cost, not low predictive quality. Respect the host-provided CPU/memory limits. Never reset the deadline on restart."
    records["resource_contract"] = {"id": "resource_contract", "type": "contract", "title": "Resource contract", "library": "project", "version": "1", "text": contract, "symbols": [], "source_url": "project:resource_contract", "license": "project", "conditions": [], "warnings": ["A timeout is evidence of resource cost, not low predictive quality."], "content_status": "complete", "applicability": "unknown"}
    alias_lookup = {alias: record for record in records.values() for alias in record.get("aliases", [])}
    for record in records.values():
        if record["type"] != "method":
            continue
        supports = []
        for reference in record["payload"]["references"]:
            source = records.get(reference) or alias_lookup.get(reference)
            if source is None:
                raise ValueError("Method has an unresolved reference: " + reference)
            body = source["text"]
            end = body.find("\n\n")
            end = end if end > 0 else len(body)
            supports.append({"reference_id": reference, "resolved_id": source["id"], "version": source["version"],
                             "start": 0, "end": end, "excerpt": body[:end], "excerpt_sha256": digest(body[:end]),
                             "doc_sha256": digest(normalize(body)), "semantic_scope": "Documented API contract; the proposed experiment remains a project hypothesis"})
        record["payload"]["supporting_sections"] = supports
        payload = record["payload"]
        record["text"] = "\n\n".join([payload["title"], "Project hypothesis: " + payload["hypothesis"],
            "Conditions: " + " ".join(payload["conditions"]), "Contraindications: " + payload["constraints"],
            "Cost: " + payload["cost"], "Minimum experiment: " + payload["minimum_experiment"],
            "Evidence: " + ", ".join(payload["references"])])
        if payload.get("routing_signal"):
            record["text"] = "\n\n".join([payload["title"],
                "Observation: " + payload["observations"], "Diagnostic: " + payload["diagnostic"],
                "Minimum experiment: " + payload["minimum_experiment"],
                "Acceptance: " + payload["acceptance"], "Contraindications: " + payload["constraints"],
                "Execution: " + payload["implementation_path"] + "; unverified until concrete capability check.",
                "Cost: " + payload["cost"], "Evidence status: project hypothesis, not measured improvement.",
                "Research sources: " + ", ".join(payload.get("research_sources", []))])
    for record in records.values():
        versions = {(record["library"], record["version"])}
        payload = record.get("payload", {})
        references = payload.get("references", [])
        for method in payload.get("method_refs", []):
            references = references + records.get("method:"+method, {}).get("payload", {}).get("references", [])
        if record["type"] == "search_space":
            recipe = records.get("recipe:"+payload.get("id", ""), {}).get("payload", {})
            for method in recipe.get("method_refs", []):
                references = references + records.get("method:"+method, {}).get("payload", {}).get("references", [])
        for reference in references:
            source = records.get(reference) or alias_lookup.get(reference)
            if source:versions.add((source["library"],source["version"]))
        record["library_versions"] = [{"library":library,"version":version} for library,version in sorted(versions)]
    output = frozen / "documents.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for identifier, record in sorted(records.items()):
            record["text"] = normalize(record["text"])
            record["required_spans"] = required_spans(record["text"])
            record["doc_sha256"] = digest(record["text"])
            stream.write(canonical(record) + "\n")
    counts = {}
    failed_paths = {(gap["library"],gap["path"]) for gap in expansion}
    coverage_items = []
    for item in inventory:
        result = dict(item)
        if (item["library"],item["id"]) in failed_paths and item["status"] == "included":
            result.update(status="error", reason="unresolved_document_expansion; partial text retained")
        coverage_items.append(result)
    for package in allowed:
        items = [x for x in coverage_items if x["library"] == package]
        state = Counter(x["status"] for x in items)
        unresolved = [x for x in expansion if x["library"] == package]
        scoped = len(items) - state["out_of_scope"]
        counts[package] = {**dict(state), "in_scope": scoped, "included_fraction": state["included"] / scoped if scoped else 0,
                           "unresolved_expansions": len(unresolved), "complete": not unresolved and not state["unavailable"] and not state["error"]}
    write_json(frozen / "coverage.json", {"libraries": counts, "unresolved": expansion,
               "complete": all(x["complete"] for x in counts.values()), "inventory_sha256": digest(expected),
               "errored_items": [item for item in coverage_items if item["status"] in {"error","unavailable"}]})
    (frozen / "THIRD_PARTY_NOTICES.txt").write_text("\n\n".join(notices) + "\n", encoding="utf-8")
    write_json(frozen / "inputs.lock.json", {"files": input_files, "builder_sha256": builder_hashes, "manifest_sha256": digest(manifest),
               "documents_sha256": file_hash(output), "expected_sha256": file_hash(frozen / "expected.json"),
               "coverage_sha256": file_hash(frozen / "coverage.json"), "notices_sha256": file_hash(frozen / "THIRD_PARTY_NOTICES.txt"), "requirements_sha256": file_hash(root / "requirements.txt")})
    print(canonical({"documents": len(records), "unresolved": len(expansion), "bytes": output.stat().st_size}))


SCHEMA = """
PRAGMA page_size=4096;
CREATE TABLE docs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, library TEXT NOT NULL, version TEXT NOT NULL, title TEXT NOT NULL, sha TEXT NOT NULL, meta TEXT NOT NULL);
CREATE INDEX docs_filter ON docs(library,version,kind);
CREATE TABLE libraries (doc TEXT NOT NULL, library TEXT NOT NULL, version TEXT NOT NULL, PRIMARY KEY(doc,library,version));
CREATE INDEX library_filter ON libraries(library,version,doc);
CREATE TABLE texts (sha TEXT PRIMARY KEY, text TEXT NOT NULL);
CREATE TABLE symbols (symbol TEXT NOT NULL, short TEXT NOT NULL, doc TEXT NOT NULL, PRIMARY KEY(symbol,doc));
CREATE INDEX symbol_short ON symbols(short);
CREATE TABLE aliases (alias TEXT PRIMARY KEY, doc TEXT NOT NULL);
CREATE TABLE fragments (id INTEGER PRIMARY KEY, doc TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL, sha TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL);
CREATE INDEX fragment_doc ON fragments(doc,start);
CREATE TABLE postings (term TEXT PRIMARY KEY, ids BLOB NOT NULL);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def build(frozen, destination, *, pilot=False):
    if destination.exists():
        raise FileExistsError("Build destination must be a fresh directory: " + str(destination))
    lock = read_json(frozen / "inputs.lock.json")
    for name, key in (("documents.jsonl", "documents_sha256"), ("expected.json", "expected_sha256"), ("coverage.json", "coverage_sha256")):
        if file_hash(frozen / name) != lock[key]:
            raise ValueError("Frozen input checksum mismatch: " + name)
    if "notices_sha256" in lock and file_hash(frozen / "THIRD_PARTY_NOTICES.txt") != lock["notices_sha256"]:
        raise ValueError("License notices checksum mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mlkb-build-", dir=destination.parent))
    try:
        db = sqlite3.connect(staging / "knowledge.sqlite")
        db.executescript(SCHEMA)
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA cache_size=-32768")
        postings = defaultdict(list)
        count, fragment_id = 0, 0
        if pilot:
            largest = {}
            with (frozen / "documents.jsonl").open() as stream:
                for line in stream:
                    obj = json.loads(line)
                    if obj["library"] not in largest or len(obj["text"]) > len(largest[obj["library"]]["text"]):
                        largest[obj["library"]] = obj
            chosen = {obj["id"] for obj in largest.values()}
        with (frozen / "documents.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if pilot and record["id"] not in chosen:
                    continue
                text = record.pop("text")
                if digest(text) != record["doc_sha256"]:
                    raise ValueError("Normalized document checksum mismatch")
                identifier = record["id"]
                db.execute("INSERT INTO texts VALUES (?,?) ON CONFLICT DO NOTHING", (record["doc_sha256"], text))
                db.execute("INSERT INTO docs VALUES (?,?,?,?,?,?,?)", (identifier, record["type"], record["library"], record["version"], record["title"], record["doc_sha256"], canonical(record)))
                for version in record.get("library_versions", [{"library":record["library"],"version":record["version"]}]):
                    db.execute("INSERT INTO libraries VALUES (?,?,?)", (identifier, version["library"], version["version"]))
                for symbol in sorted(set(record.get("symbols", []))):
                    db.execute("INSERT INTO symbols VALUES (?,?,?)", (symbol, symbol.rsplit(".", 1)[-1].casefold(), identifier))
                for alias in sorted(set(record.get("aliases", []))):
                    db.execute("INSERT INTO aliases VALUES (?,?)", (alias, identifier))
                for start, end in chunks(text):
                    fragment_id += 1
                    part = text[start:end]
                    db.execute("INSERT INTO fragments VALUES (?,?,?,?,?,?,?)", (fragment_id, identifier, start, end, digest(part), record["title"], part))
                    for term in sorted(tokens(part + " " + record["title"])):
                        postings[term].append(fragment_id)
                count += 1
        for term, ids in sorted(postings.items()):
            db.execute("INSERT INTO postings VALUES (?,?)", (term, zlib.compress(canonical(ids).encode(), 9)))
        del postings
        db.execute("CREATE VIRTUAL TABLE fts USING fts5(title,text,content='fragments',content_rowid='id',tokenize='unicode61 remove_diacritics 2')")
        db.execute("INSERT INTO fts(fts) VALUES ('rebuild')")
        db.execute("INSERT INTO fts(fts) VALUES ('optimize')")
        corpus = digest({"documents": lock["documents_sha256"], "format": FORMAT_VERSION, "pilot": pilot})
        db.execute("INSERT INTO meta VALUES ('corpus',?)", (corpus,))
        db.commit()
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        db.execute("VACUUM")
        db.close()
        shutil.copyfile(frozen / "coverage.json", staging / "coverage.json")
        if "notices_sha256" in lock:
            shutil.copyfile(frozen / "THIRD_PARTY_NOTICES.txt", staging / "THIRD_PARTY_NOTICES.txt")
        write_json(staging / "knowledge.lock.json", {"format": FORMAT_VERSION, "normalizer": NORMALIZER_VERSION,
                   "chunker": CHUNKER_VERSION, "corpus": corpus, "requirements_sha256": lock["requirements_sha256"],
                   "documents_sha256": lock["documents_sha256"], "inputs_sha256": file_hash(frozen / "inputs.lock.json"),
                   "inventory_sha256": lock["expected_sha256"], "python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
                   "builder_sha256": lock.get("builder_sha256", {}),
                   "documents": count, "fragments": fragment_id, "pilot": pilot,
                   "files": {p.name: file_hash(p) for p in sorted(staging.iterdir())}})
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return read_json(destination / "knowledge.lock.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["freeze", "build"])
    p.add_argument("--root", type=Path, default=Path(__file__).parent)
    p.add_argument("--destination", type=Path)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    if args.action == "freeze":
        freeze(args.root)
    else:
        print(canonical(build(args.root / "frozen", args.destination, pilot=args.pilot)))
