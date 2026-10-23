from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import get_args

from .contracts import Kind, PipelineSpec
from .compatibility import incompatibilities, registry_fingerprint
from .util import file_hash, read_json


def validate_catalogs(root):
    """Validate authored cards, executable recipes and typed search spaces."""
    catalogs = {name: {p.stem: read_json(p) for p in sorted((root / name).glob("*.json"))}
        for name in ("methods", "recipes", "search_spaces")}
    known = {json.loads(line)["doc_id"] for line in (root / "references/records.jsonl").read_text().splitlines() if line}
    for group, entries in catalogs.items():
        for identifier, entry in entries.items():
            if entry.get("id") != identifier or not isinstance(entry.get("version"), int) or entry["version"] < 1:
                raise ValueError("Invalid catalog identity/version: " + identifier)
            if group == "methods":
                for field in ("title", "observations", "hypothesis", "constraints", "minimum_experiment", "acceptance"):
                    if not isinstance(entry.get(field), str) or not entry[field]:
                        raise ValueError("Invalid method field: " + identifier + "." + field)
                refs = entry.get("references", [])
                if not isinstance(refs, list) or set(refs) - known != set(entry.get("missing_references", [])):
                    raise ValueError("Unregistered method references: " + identifier)
            elif group == "recipes":
                PipelineSpec.model_validate(entry["pipeline"])
                if not entry.get("task_kinds") or not set(entry["task_kinds"]) <= set(get_args(Kind)):
                    raise ValueError("Invalid recipe task families: " + identifier)
                if not set(entry.get("method_refs", [])) <= set(catalogs["methods"]):
                    raise ValueError("Unknown recipe method: " + identifier)
                if entry.get("search_space_ref") and entry["search_space_ref"] not in catalogs["search_spaces"]:
                    raise ValueError("Unknown recipe parameter space: " + identifier)
                if entry.get("refit_policy", {}).get("new_search") is not False:
                    raise ValueError("Recipe refit must forbid new search: " + identifier)
            else:
                if entry.get("owner") != "optuna" or not entry.get("parameters"):
                    raise ValueError("Invalid parameter-space owner: " + identifier)
                pending = dict(entry["parameters"])
                resolved = set()
                while pending:
                    ready = [name for name, rule in pending.items() if not rule.get("when") or rule["when"].get("parameter") in resolved]
                    if not ready:
                        raise ValueError("Cyclic or unknown parameter-space condition: " + identifier)
                    for name in ready:
                        rule = pending.pop(name)
                        if rule.get("type") == "categorical":
                            if not isinstance(rule.get("choices"), list) or not rule["choices"]:
                                raise ValueError("Empty categorical search space")
                        elif rule.get("type") in {"int", "float"}:
                            if not isinstance(rule.get("low"), (int, float)) or not isinstance(rule.get("high"), (int, float)) or rule["low"] > rule["high"] or rule.get("log") and rule["low"] <= 0:
                                raise ValueError("Invalid numeric search bounds")
                            if rule.get("step", 1) <= 0:
                                raise ValueError("Invalid search step")
                        else:
                            raise ValueError("Unknown parameter rule type")
                        resolved.add(name)
    return {name: len(entries) for name, entries in catalogs.items()}

INTERFACE = """Candidate interface v2 (project-owned):
from automl_agent.api import prediction_bundle, TrainingSummary
def build_pipeline(context): return Candidate()
Candidate.fit(X, y, context) -> TrainingSummary(rows=len(X), seconds=...)
Candidate.predict(X, context) -> prediction_bundle({target: array}, X, context, prediction_type='values')
X is a pandas DataFrame indexed by immutable row keys. y is a DataFrame with all targets.
Keep the original X.index when transforming or sorting; transfer keys with the rows.
Use context['task'] for targets/classes and context['cpu'], context['memory_mb'] for limits.
The selected PipelineSpec and training seed are context['pipeline'] and context['seed'].
In pinned scikit-learn 1.5.2, IterativeImputer(keep_empty_features=True) skips
the iterative process. The ready FeatureMatrix preserves empty columns outside
that imputer; check actual_iterations and auxiliary_fits before claiming model imputation.
For a reusable transformer define build_transformer(context), fit(X,y=None), transform(X).
The transform must return a DataFrame with unchanged index. Ready preprocessing and model
are selected in the accompanying PipelineSpec; use custom_component='transformer'.
Write exactly the entry point selected by custom_component: 'transformer' requires
build_transformer, 'pipeline' requires build_pipeline, 'analysis' requires analyze.
A transformer does not train or wrap the ready model and must not define build_pipeline.
Minimal transformer structure (replace transform with your feature hypothesis):
class Features:
    def fit(self, X, y=None): return self
    def transform(self, X): return X.copy()
def build_transformer(context): return Features()
For fit-only analysis define analyze(X,y,context) and custom_component='analysis'.
Return {'statistics':[{'name':'statistic_id','aggregation':'correlation',
 'columns':['known_feature','known_target'],'value':0.2,'support_rows':len(X)}]}.
Use scalar aggregate statistics; do not return input rows, target arrays or examples.
Aggregates below 30 supporting fit rows are suppressed before API transmission.
Only allowlisted libraries are available. No filesystem, network, subprocess, pretrained
models, installation, or protected labels. The controller calculates the metric.
For a full custom pipeline, the exact public imports are:
from automl_agent.pipelines import DirectPipeline
from automl_agent.contracts import PipelineSpec, TaskSpec
The constructor is DirectPipeline(spec, task), where spec is a PipelineSpec and task is
TaskSpec.model_validate(context['task']). There is no top-level 'contracts' module.
Do not guess internal attributes or constructor arguments; use the transformer interface
to attach new features to a ready model. Never fit on validation/test.
"""


class ReferenceStore:
    def __init__(self, root: Path, capabilities=None):
        self.root, self.capabilities = root, capabilities
        self.lock = read_json(root / "knowledge.lock.json") if (root / "knowledge.lock.json").is_file() else None
        self.version = self.lock["bundle_version"] if self.lock else "minimal-interface-v1"
        self.cache = {}
        self.records = []
        if (root / "references" / "records.jsonl").is_file():
            self.records = [json.loads(line) for line in (root / "references" / "records.jsonl").read_text().splitlines() if line]

    def verify(self, requirements: Path):
        if self.lock is None:
            return {"status": "partial", "reason": "Extended knowledge corpus has not been built"}
        if self.lock["requirements_sha256"] != file_hash(requirements):
            raise ValueError("Knowledge bundle requirements hash mismatch")
        for relative, expected in self.lock["files"].items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root.resolve()) or not path.is_file() or file_hash(path) != expected:
                raise ValueError("Knowledge bundle integrity verification failed")
        validate_catalogs(self.root)
        return {"status": "verified", "bundle_version": self.version, "coverage": self.lock.get("coverage", {})}

    def _catalog(self, directory):
        return [read_json(path) for path in sorted((self.root / directory).glob("*.json"))] if (self.root / directory).is_dir() else []

    def query(self, args, task=None, plan=None):
        if getattr(self, "disabled_reason", None) and args.reference_id != "candidate_interface":
            return {"status": "disabled_integrity_error", "reason": self.disabled_reason}
        key = (self.version, registry_fingerprint(self.capabilities), args.hash, task.hash if task else "", plan.hash if plan else "")
        if key in self.cache:
            return self.cache[key]
        result = self._query(args, task, plan)
        result["knowledge_bundle_version"] = self.version
        self.cache[key] = result
        return result

    def _query(self, args, task, plan):
        if args.reference_id == "candidate_interface" or args.mode == "api" and args.query in {"", "candidate_interface", "build_pipeline"}:
            return {"status": "available", "doc_id": "candidate_interface", "source": "project_owned_contract", "version_match": "exact_release", "content": INTERFACE}
        if args.mode in {"methods", "recipes", "search_space"}:
            directory = {"methods": "methods", "recipes": "recipes", "search_space": "search_spaces"}[args.mode]
            entries = self._catalog(directory)
            if args.reference_id:
                entries = [entry for entry in entries if entry.get("id") == args.reference_id]
            elif args.query:
                words = set(re.findall(r"[\w.]+", args.query.lower()))
                entries.sort(key=lambda entry: -sum(w in json.dumps(entry, ensure_ascii=False).lower() for w in words))
            if args.mode == "recipes" and task:
                for entry in entries:
                    reasons = []
                    if task.kind not in entry.get("task_kinds", []):
                        reasons.append("unsupported_task_family")
                    pipeline = PipelineSpec.model_validate(entry["pipeline"])
                    reasons.extend(incompatibilities(pipeline, task, plan.strategy if plan else None, self.capabilities))
                    entry["runtime_status"] = "verified" if not reasons and self.capabilities and plan and self.capabilities.verified(pipeline, task, plan.strategy) else "unverified"
                    entry["exclusion_reasons"] = reasons
                    entry["applicable"] = not reasons
            elif args.mode == "search_space" and task:
                recipes = {entry["id"]: entry for entry in self._catalog("recipes")}
                for entry in entries:
                    recipe = recipes.get(entry["id"])
                    reasons = []
                    if recipe is None:
                        reasons.append("parameter_space_has_no_registered_executable_recipe")
                    else:
                        pipeline = PipelineSpec.model_validate({**recipe["pipeline"], "engine": "optuna",
                            "search_owner": "optuna", "search_space": entry["parameters"]})
                        if task.kind not in recipe["task_kinds"]:
                            reasons.append("unsupported_task_family")
                        reasons.extend(incompatibilities(pipeline, task, plan.strategy if plan else None, self.capabilities))
                    entry.update(applicable=not reasons, exclusion_reasons=reasons, runtime_status="unverified")
            elif args.mode == "methods":
                # A conceptual card does not establish executable compatibility.
                for entry in entries:
                    entry.update(applicable=None, applicability_status="requires_concrete_pipeline",
                        runtime_status="unverified", exclusion_reasons=[])
            return {"status": "available" if entries else "not_found", "entries": entries[args.cursor:args.cursor + args.limit],
                "next_cursor": args.cursor + args.limit if len(entries) > args.cursor + args.limit else None}
        candidates = self.records
        if args.library:
            candidates = [r for r in candidates if r["library"] == args.library]
        if args.version:
            candidates = [r for r in candidates if r["required_version"] == args.version and r["version_match"] == "exact_release"]
            if not candidates:
                return {"status": "missing_version", "library": args.library, "version": args.version}
        resolution = None
        if args.reference_id:
            candidates = [r for r in candidates if r["doc_id"] == args.reference_id]
        elif args.mode == "api":
            symbols = {symbol: record for record in candidates for symbol in record.get("api_symbols", [])}
            symbol = args.query.strip()
            if symbol not in symbols:
                aliases = sorted(name for name in symbols if name.rsplit(".", 1)[-1].casefold() == symbol.casefold())
                if len(aliases) == 1:
                    symbol = aliases[0]
                    resolution = {"requested": args.query, "resolved_symbol": symbol, "rule": "unique_short_name_in_filtered_corpus"}
                else:
                    words = set(re.findall(r"\w+", symbol.casefold()))
                    scored = [(sum(4 if word == name.rsplit(".", 1)[-1].casefold() else 1
                        for word in words if word in name.casefold()), name) for name in (aliases or symbols)]
                    suggestions = []
                    for score, name in sorted(scored, key=lambda item: (-item[0], len(item[1]), item[1]))[:args.limit]:
                        if not score:
                            continue
                        record = symbols[name]
                        suggestions.append({"symbol": name, "doc_id": record["doc_id"],
                            "library": record["library"], "required_version": record["required_version"]})
                    return {"status": "not_found", "query": args.query,
                        "reason": "Ambiguous short name" if len(aliases) > 1 else "Use an exact symbol or document ID from suggestions; no API was inferred",
                        "suggestions": suggestions}
            exact = [r for r in candidates if symbol in r.get("api_symbols", [])]
            candidates = exact
        else:
            query = args.query.lower()
            aliases = read_json(self.root / "aliases_ru_en.json") if (self.root / "aliases_ru_en.json").is_file() else {}
            for source, replacement in aliases.items():
                if source in query:
                    query += " " + replacement
            words = re.findall(r"[\w.]+", query)
            matching = []
            if (self.root / "search.sqlite").is_file() and words:
                try:
                    connection = sqlite3.connect(f"file:{self.root / 'search.sqlite'}?mode=ro", uri=True)
                    expression = " OR ".join('"' + word.replace('"', '""') + '"' for word in words)
                    matching = [row[0] for row in connection.execute("SELECT doc_id FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT 30", (expression,))]
                    connection.close()
                except sqlite3.OperationalError:
                    matching = []
            if matching:
                candidates = sorted([r for r in candidates if r["doc_id"] in matching], key=lambda r: matching.index(r["doc_id"]))
            else:
                scored = [(sum(word in (r.get("search_text", "") + " " + " ".join(r.get("api_symbols", []))).lower() for word in words), r) for r in candidates]
                candidates = [r for score, r in sorted(scored, key=lambda pair: -pair[0]) if score > 0]
        if not candidates:
            return {"status": "not_found", "query": args.query, "reason": "No matching verified-version reference; do not infer an API"}
        if args.mode == "read" or args.reference_id:
            record = candidates[0]
            content = (self.root / record["path"]).read_text(encoding="utf-8")
            fragment = content[args.cursor:args.cursor + args.max_chars]
            return {"status": record["content_status"], "record": record, "content": fragment,
                "next_cursor": args.cursor + len(fragment) if args.cursor + len(fragment) < len(content) else None}
        entries, left = [], args.max_chars
        for record in candidates[:args.limit]:
            content = (self.root / record["path"]).read_text(encoding="utf-8")
            length = min(left, max(300, args.max_chars // args.limit))
            entries.append({"record": record, "content": content[:length], "next_cursor": length if length < len(content) else None})
            left -= length
            if left <= 0:
                break
        result = {"status": "available", "entries": entries}
        if resolution:
            result["query_resolution"] = resolution
        return result
