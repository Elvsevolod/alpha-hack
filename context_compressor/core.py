"""Confirmed compact projections and one immutable snapshot per decision."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import time
import uuid
from typing import Callable

from .contracts import (AccessDenied, BUILDER_VERSION, CORE_LIMITS, ContextError,
                        Snapshot, Source, bounded, canonical, digest, finite,
                        identifier, read_ref, size, image_descriptor, validate_image)
from .store import ContextStore


KINDS = {"regression", "classification", "multioutput_regression", "multioutput_classification",
         "multilabel", "forecasting", "ranking"}
METRICS = {"rmse", "mse", "mae", "mape", "r2", "f1", "accuracy", "balanced_accuracy", "roc_auc", "log_loss", "average_precision", "ndcg", "map"}
COLLECTIONS = ("targets", "id_columns", "feature_columns", "categorical_columns", "text_columns",
               "series_columns", "classes", "feature_constraints", "evidence")
REQUIRED_FLAGS = {"worker_network_allowed": False, "credential_environment_allowed": False,
                  "control_labels_accessible": False, "arbitrary_external_slices": False,
                  "full_contract_enforced": True}
WORK_IDS = {"baseline_id", "leader_id", "latest_id", "active_id", "active_operation_id", "parent_id", "related_failure_id", "suggested_new_operation_id"}


def validate_core(core):
    if set(core) != set(CORE_LIMITS):
        raise ContextError("The protected core has missing or unknown sections")
    for name, maximum in CORE_LIMITS.items():
        if size(core[name]) > maximum:
            raise ContextError(f"Protected core section {name} exceeds {maximum} UTF-8 bytes; use a lossless reference or reject support")
    if size(core) > 16384:
        raise ContextError("Protected core exceeds 16384 UTF-8 bytes")


def _descriptor(value):
    result = {"type": type(value).__name__, "sha256": digest(value)}
    if isinstance(value, (list, dict)):
        result["count"] = len(value)
    return result


def _task_projection(task, task_hash):
    if task.get("kind") not in KINDS or not isinstance(task.get("metric"), dict):
        raise ContextError("A confirmed supported TaskSpec is required")
    metric = dict(task["metric"])
    if metric.get("name") not in METRICS:
        raise ContextError("Unsupported metric; do not silently shorten its meaning")
    maximize = metric["name"] not in {"rmse", "mse", "mae", "mape", "log_loss"}
    if "maximize" in metric and metric["maximize"] is not maximize:
        raise ContextError("Metric direction contradicts its confirmed name")
    for key, value in list(metric.items()):
        bounded(key, 128, "metric parameter")
        if isinstance(value, (list, dict)) or isinstance(value, str) and len(value.encode()) > 128:
            metric[key] = {**_descriptor(value), "section": "metric", "field": key}
    projection = {"kind": task["kind"], "metric": metric, "maximize": maximize,
                  "prediction_type": task.get("output", {}).get("prediction_type"),
                  "collections": {key: _descriptor(task[key]) for key in COLLECTIONS if key in task},
                  "output": {key: _descriptor(value) for key, value in task.get("output", {}).items() if key != "prediction_type"},
                  "temporal_group_roles": {key: _descriptor(task[key]) for key in ("time_column", "group_column", "horizon", "explicit_validation") if task.get(key) is not None},
                  "contract": read_ref("task", task_hash, "contract"),
                  "sections": sorted(task)}
    if projection["prediction_type"] not in {"values", "labels", "probabilities", "scores"}:
        raise ContextError("A confirmed prediction type is required")
    # Keep ordinary contracts readable. Wide collections retain an explicit
    # descriptor and the lossless contract route instead of hiding every value.
    for key in ("targets", "id_columns"):
        value = task.get(key, [])
        candidate = {**projection, key: value}
        projection[key] = value if size(value) <= 512 and size(candidate) <= CORE_LIMITS["task"] else _descriptor(value)
    for key, value in task.get("output", {}).items():
        candidate = {**projection, "output": {**projection["output"], key: value}}
        if key != "prediction_type" and size(value) <= 512 and size(candidate) <= CORE_LIMITS["task"]:
            projection["output"][key] = value
    for key in projection["temporal_group_roles"]:
        candidate = {**projection, "temporal_group_roles": {**projection["temporal_group_roles"], key: task[key]}}
        if size(task[key]) <= 256 and size(candidate) <= CORE_LIMITS["task"]:
            projection["temporal_group_roles"][key] = task[key]
    for key in ("classes", "categorical_columns", "text_columns", "series_columns"):
        value = task.get(key)
        if value is not None and size(value) <= 384:
            candidate = {**projection, key: value}
            if size(candidate) <= CORE_LIMITS["task"]:
                projection = candidate
    return projection


def configure(store: ContextStore, *, run_id: str, task: dict, policies: dict, policy_flags: dict,
              data_hash: str, protocol_hash: str, environment_hash: str, deadline_epoch: float,
              reserve_seconds: float, cpu: int, memory_mib: int, counters: dict[str, int],
              work: dict | None = None, profile: dict | None = None):
    """Controller-only state publication. Never call with model-supplied facts.

    Full TaskSpec validation and policy enforcement remain the executor's job.
    This method checks whether their compact representation is supported.
    """
    identifier(run_id, "run_id")
    for name, value in (("data_hash", data_hash), ("protocol_hash", protocol_hash), ("environment_hash", environment_hash)):
        identifier(value, name)
    finite(deadline_epoch, "deadline_epoch", 1)
    finite(reserve_seconds, "reserve_seconds")
    if type(cpu) is not int or cpu < 1 or type(memory_mib) is not int or memory_mib < 1:
        raise ContextError("Positive CPU and RAM limits are required")
    if any(policy_flags.get(key) is not value for key, value in REQUIRED_FLAGS.items()):
        raise ContextError("Missing or unsafe enforcement/access flags")
    for key, value in policy_flags.items():
        bounded(key, 128, "policy code")
        if isinstance(value, str):
            bounded(value, 128, "policy value")
        elif not isinstance(value, (bool, int)):
            raise ContextError("Mandatory policy conditions need bounded codes and an enforced full-policy reference")
    for key, value in counters.items():
        bounded(key, 128, "counter")
        if type(value) is not int or value < 0:
            raise ContextError("Counters must be nonnegative integers")
    work = dict(work or {})
    routes = {"leader_pipeline", "research_options", "unexecuted_intents", "local_guides"}
    if set(work) - (WORK_IDS | {"related_ids", "dependencies", "diagnostic", "hypothesis", "rejected_operations", "decision_frame", "research_memory"} | routes):
        raise ContextError("Unsupported current-work fields")
    for name in routes & set(work):
        route = work[name]
        if (not isinstance(route, dict) or route.get("tool") != "read_run_records"
                or not isinstance(route.get("arguments"), dict) or size(route) > 768):
            raise ContextError("Current-work routes must be bounded public reads")
    for key in WORK_IDS:
        if work.get(key) is not None:
            identifier(work[key], key)
    for key in ("related_ids", "dependencies"):
        if key in work:
            if not isinstance(work[key], (list, tuple)) or len(work[key]) > 8:
                raise ContextError(f"{key} supports at most eight IDs")
            for item in work[key]:
                identifier(item, key)
    if "diagnostic" in work:
        bounded(work["diagnostic"], 512, "current diagnostic", empty=True)
    if size(work.get("hypothesis")) > 2048 or size(work.get("rejected_operations")) > 1024:
        raise ContextError("Current hypothesis or rejection summaries exceed their bounded projection")
    frame = work.get("decision_frame")
    if frame is not None:
        if not isinstance(frame, dict) or type(frame.get("version")) is not int or frame.get("version") != 1 or size(frame) > 768:
            raise ContextError("Unsupported or oversized decision frame")
        allowed_frame = {"version", "read", "facts_source", "last_operation", "analysis_status", "last_request_status", "limitation"}
        if set(frame) - allowed_frame:
            raise ContextError("Unknown decision frame fields")
    memory = work.get("research_memory")
    if memory is not None:
        if (not isinstance(memory, dict) or type(memory.get("version")) is not int
                or memory["version"] != 1 or size(memory) > 1536
                or set(memory) - {"version", "read", "current", "previous", "selection_source", "limitation"}
                or len(memory.get("previous", [])) > 2):
            raise ContextError("Unsupported or oversized research memory")
    task_hash, policy_hash = digest(task), digest(policies)
    identity = {"run_id": run_id, "task_hash": task_hash, "data_hash": data_hash,
                "protocol_hash": protocol_hash, "environment_hash": environment_hash,
                "policy_hash": policy_hash, "builder_version": BUILDER_VERSION}
    resources = {"flags": policy_flags, "policy": read_ref("policies", policy_hash, "policy"),
                 "cpu": cpu, "memory_mib": memory_mib, "deadline_epoch": deadline_epoch,
                 "reserve_seconds": reserve_seconds, "counters": dict(counters)}
    for key in ("validation_strategy", "selection", "feedback", "budget_mode", "training_seed", "model_response_limit_seconds"):
        if key in policies:
            value = policies[key]
            if size(value) <= 1024:
                resources[key] = value
            elif key == "feedback":
                resources[key] = {k: v for k, v in value.items() if k != "report_ids"}
                resources[key]["report_ids"] = _descriptor(value.get("report_ids", []))
            else:
                resources[key] = _descriptor(value)
    state = {"identity": identity, "task": _task_projection(task, task_hash),
             "policies_resources": resources, "current_work": work, "profile": profile}
    with store.transaction():
        old = store.get_meta("controller_state")
        if old:
            if old["identity"]["run_id"] != run_id or old["policies_resources"]["deadline_epoch"] != deadline_epoch:
                raise ContextError("Reconfiguration cannot reset the run identity or original deadline")
            if any(counters.get(key, -1) < value for key, value in old["policies_resources"]["counters"].items()):
                raise ContextError("Reconfiguration cannot reset or remove counters")
        # The projection is confirmed controller state; text inside the full
        # document (labels, file names, quotations) never becomes an instruction.
        for source in (Source("task", task_hash, canonical(task), "contract", "external_data", origin="confirmed_contract"),
                       Source("policies", policy_hash, canonical(policies), "policy", "controller_fact", origin="controller_policy")):
            store._put_source(source)
        for key, value in task.items():
            bounded(key, 128, "contract section")
            store._put_source(Source("task", task_hash, canonical(value), "contract", "external_data", key, "confirmed_contract"))
        if profile:
            cached = store.db.execute("SELECT reference_json FROM cc_profiles WHERE cache_key=?", (profile.get("cache_key"),)).fetchone()
            if not cached or json.loads(cached[0]) != profile:
                raise ContextError("Profile reference must come from ProfileCache")
        store._put_meta("controller_state", state)
        core = store.route(_core(store, state))
        validate_core(core)
    return state


@dataclass(frozen=True)
class ProfileKey:
    data_version: str
    fit_hash: str
    scope: str = "fit"
    format_version: str = "2"

    def __post_init__(self):
        identifier(self.data_version, "profile data_version")
        identifier(self.fit_hash, "fit_hash")
        bounded(self.format_version, 128, "profile format_version")
        if self.scope not in {"fit", "validation_features", "test_features"}:
            raise AccessDenied("A data profile must have an explicit permitted feature/fit scope")

    @property
    def hash(self):
        return digest(self.__dict__)


class ProfileCache:
    def __init__(self, store: ContextStore):
        self.store = store

    def get_or_create(self, key: ProfileKey, calculate: Callable[[], dict]):
        self.store._check_scope(key.scope)
        cached = self.store.db.execute("SELECT reference_json FROM cc_profiles WHERE cache_key=?", (key.hash,)).fetchone()
        if cached:
            return json.loads(cached[0])
        document = calculate()
        if not isinstance(document, dict) or not isinstance(document.get("columns"), dict):
            raise ContextError("A profile calculator must return a document with a columns mapping")
        columns = document["columns"]
        counts = Counter(str(value.get("dtype", value.get("type", "unknown"))) for value in columns.values())
        summary = {"rows": document.get("rows", {}), "feature_count": len(columns), "type_counts": dict(counts),
                   "columns_with_missing": sum(bool(value.get("missing", value.get("missing_count", 0))) for value in columns.values()),
                   "columns_with_new_categories": sum(bool(value.get("new_categories", value.get("unseen_categories", value.get("new_test_category_rows", 0)))) for value in columns.values()),
                   "scope": key.scope, "data_version": key.data_version, "fit_hash": key.fit_hash,
                   "format_version": key.format_version, "source": "controller_profile"}
        allowed_risks = {"empty_columns", "unknown_categories", "missingness_shift", "high_cardinality",
            "duplicate_features", "label_conflicts", "duplicate_target_variation", "class_imbalance", "rare_classes", "heavy_tail"}
        summary["risk_flags"] = {name: value for item in document.get("observations", []) if item.get("type") == "fit_risks"
            for name, value in item.get("flags", {}).items() if name in allowed_risks and type(value) is bool}
        summary["trust"] = "external_data"
        source_id = "profile:" + key.hash
        reference = {"cache_key": key.hash, "summary": summary,
                     "read": read_ref(source_id, key.format_version, key.scope),
                     "column_section_rule": "column.<sha256 of canonical JSON column name>"}
        if key.format_version == "6":
            summary.update(status=document.get("status", "available"), findings=[])
            if document.get("error"):
                summary["error"] = str(document["error"])[:128]
            for item in document.get("findings", [])[:3]:
                if item.get("scope") != key.scope or item.get("kind") not in allowed_risks | {"constant_features"}:
                    continue
                finding = {name: str(item.get(name, ""))[:160] for name in ("kind", "observation")}
                summary["findings"].append(finding)
                if size(reference) > 3072:
                    summary["findings"].pop()
                    break
        if size(reference) > 3072:
            raise ContextError("Profile summary exceeds 3072 bytes; calculator must provide a bounded summary")
        with self.store.transaction():
            self.store._put_source(Source(source_id, key.format_version, canonical(document), key.scope,
                                          "external_data", origin="controller_profile"))
            for name, value in columns.items():
                self.store._put_source(Source(source_id, key.format_version, canonical({"column": name, "profile": value}), key.scope,
                                              "external_data", "column." + digest(name), "controller_profile"))
            # All compact columns when they fit; prioritize missingness and
            # unseen categories on wide datasets. Omitted columns remain readable.
            selected = {}
            ranked = sorted(columns, key=lambda name: (-bool(columns[name].get("new_test_category_rows")),
                -bool(columns[name].get("missing", columns[name].get("missing_count"))), name))
            fields = ("dtype", "missing", "missing_fraction", "test_missing_fraction", "unique", "median", "new_test_category_rows")
            for name in ranked:
                value = {k: columns[name][k] for k in fields if k in columns[name]}
                if size({**selected, name: value}) <= 10240:
                    selected[name] = value
            compact = {"columns": selected, "omitted_columns": len(columns) - len(selected),
                       "scope": key.scope, "trust": "external_data"}
            self.store._put_source(Source(source_id, key.format_version, canonical(compact), key.scope,
                                         "external_data", "compact_columns", "controller_profile"))
            self.store.db.execute("INSERT OR IGNORE INTO cc_profiles VALUES(?,?)", (key.hash, canonical(reference)))
        return reference


def project_work(store, current_work):
    """One projection for sizing and final rendering of protected work."""
    work = dict(current_work)
    for key in ("baseline_id", "leader_id", "latest_id"):
        if work.get(key):
            record = store.record(work[key])
            if record:
                work[key.removesuffix("_id") + "_result"] = {name: record[name] for name in
                    ("record_id", "status", "score", "metric", "evaluation_scope", "protocol")}
    work["observation_queue"] = {"pending": store.pending_count(), "acknowledgement": "validated_durable_command"}
    return work


def _core(store, state):
    work = project_work(store, state["current_work"])
    reading = {"contract": state["task"]["contract"],
               "search": {"tool": "analyze_experiments", "arguments": {"context_search": {"filters": {}, "limit": 10}}},
               "notes_filter": {"kind": "research_note"}, "source_catalog": {"kind": "source"}}
    if state.get("profile"):
        reading["profile"] = state["profile"]["read"]
    return {"identity": state["identity"], "task": state["task"], "policies_resources": state["policies_resources"],
            "current_work": work, "research_note": store.active_note(), "reading_routes": reading}


def _select_records(store, work, note, maximum):
    candidates = {}

    def add(record_id, priority):
        if not record_id:
            return
        try:
            record = store.record(record_id)
        except AccessDenied:
            return
        if record and record["created_index"] <= maximum:
            previous = candidates.get(record_id)
            if previous is None or priority < previous[0]:
                candidates[record_id] = (priority, record)

    for value in [work.get("latest_id"), work.get("active_id"), *(note or {}).get("related_ids", [])]:
        add(value, 0)
    for key, priority in (("baseline_id", 1), ("leader_id", 1), ("parent_id", 2), ("related_failure_id", 3)):
        add(work.get(key), priority)
    for value in [*work.get("related_ids", []), *work.get("dependencies", [])]:
        add(value, 4)
    # Bounded dependency expansion; omitted edges remain in the exact source.
    for _, record in list(candidates.values())[:16]:
        for dependency in record["dependencies"]:
            add(dependency, 4)
    clause, params = store._scope_sql()
    for row in store.db.execute(f"SELECT record_id FROM cc_records WHERE kind='experiment' AND seq<=? AND {clause} ORDER BY seq DESC LIMIT 12", (maximum, *params)):
        add(row[0], 5)
    for row in store.db.execute(f"SELECT record_id FROM cc_records WHERE seq IN (SELECT max(seq) FROM cc_records WHERE kind='experiment' AND seq<=? AND {clause} GROUP BY component,engine ORDER BY max(seq) DESC LIMIT 8)", (maximum, *params)):
        add(row[0], 6)
    selected = sorted(candidates.values(), key=lambda pair: (pair[0], pair[1]["created_index"], pair[1]["record_id"]))[:32]
    cards = []
    for priority, record in selected:
        card = {**record, "selection_priority": priority}
        try:
            parent = store.record(record["parent_id"]) if record.get("parent_id") else None
        except AccessDenied:
            parent = None
        research_role = lambda role: 'full' if role in {'baseline', 'full'} else role
        comparable = bool(parent and record["status"] == parent["status"] == "completed"
                          and record["score"] is not None and parent["score"] is not None
                          and record["metric"] is not None and record["maximize"] is not None
                          and all(record[key] == parent[key] for key in
                                  ("protocol", "environment", "metric", "maximize", "data_scope"))
                          and research_role(record['evaluation_scope']) == research_role(parent['evaluation_scope'])
                          and record['evaluation_scope'] not in {'pilot', 'unknown', None})
        card["parent_comparison"] = {"comparable": comparable}
        if comparable:
            delta = record["score"] - parent["score"]
            card["parent_comparison"].update({"score_delta": delta, "gain": delta if record["maximize"] else -delta,
                                               "interpretation": "observed_difference; not_causal_evidence"})
        cards.append(card)
    return cards


def make_snapshot(store: ContextStore) -> Snapshot:
    started = time.perf_counter()
    snapshot_id = uuid.uuid4().hex
    with store.transaction():
        state = store.get_meta("controller_state")
        if state is None:
            raise ContextError("Configure confirmed controller state before snapshotting")
        core = store.route(_core(store, state))
        validate_core(core)
        clause, params = store._scope_sql()
        maximum, count = store.db.execute(f"SELECT coalesce(max(seq),0),count(*) FROM cc_records WHERE {clause}", params).fetchone()
        cards = _select_records(store, state["current_work"], core["research_note"], maximum)
        observations = store.pending_fragments()
        images = []
        for image in (store.get_meta("diagnostic_images") or [])[:1]:
            if image.get("operation_id") not in {item["operation_id"] for item in observations}:
                continue
            validate_image(image)
            if image["data_version"] != state["identity"]["data_hash"] or image["protocol_hash"] != state["identity"]["protocol_hash"]:
                continue
            store._check_scope(image["scope"])
            source = store.db.execute("SELECT scope,origin FROM cc_sources WHERE source_id=? AND version=? AND section=''",
                (image["source_id"], image["source_version"])).fetchone()
            if not source or source["scope"] != image["scope"] or source["origin"] != "get_diagnostic_report":
                raise AccessDenied("Image has no permitted published diagnostic source")
            images.append(image)
        # Whole experiment JSONs dominated the prompt and could end mid-field.
        # Only controller-created, bounded summaries are inlined automatically.
        details, sources = [], []
        for card in cards[:8]:
            row = store.db.execute("""SELECT content FROM cc_sources
                WHERE source_id=? AND version=? AND section='summary' AND scope=? AND byte_length<=1536""",
                (card["source_id"], card["source_version"], card["data_scope"])).fetchone()
            if row:
                details.append({"record_id": card["record_id"], "summary": json.loads(row[0]),
                                "read": card["read"], "trust": "external_data"})
                sources.append({"source_id": card["source_id"], "version": card["source_version"], "section": "summary"})
        keep = ("record_id", "engine", "status", "score", "metric", "maximize", "evaluation_scope", "seconds",
                "peak_memory_mib", "parent_id", "dependencies", "checked_change", "parent_comparison", "read", "selection_priority")
        cards = [{k: card[k] for k in keep} for card in cards]
        failures = store.failure_groups()
        profile = state.get("profile")
        if profile:
            reference = profile["read"]["arguments"]["context_read"]
            row = store.db.execute("SELECT content FROM cc_sources WHERE source_id=? AND version=? AND section='compact_columns' AND scope=?",
                (reference["source_id"], reference["version"], reference["scope"])).fetchone()
            if row:
                profile = {**profile, **json.loads(row[0])}
        manifest = {"core_sha256": digest(core), "observation_ids": [x["observation_id"] for x in observations],
                    "images": [image_descriptor(image) for image in images],
                    "observations_sha256": digest(observations),
                    "candidate_ids": [x["record_id"] for x in cards],
                    "sources": sources, "history_count": count,
                    "source_reads": len(details) + len(observations),
                    "allowed_scopes": sorted(store.allowed_scopes), "created_at": time.time()}
        store.db.execute("INSERT INTO cc_snapshots VALUES(?,?,?)", (snapshot_id, maximum, canonical(manifest)))
    return Snapshot(snapshot_id, canonical(core), canonical(observations), tuple(canonical(x) for x in cards),
                    tuple(canonical(store.route(x)) for x in details), canonical(store.route(failures)), canonical(store.route(profile)), maximum,
                    count, core["current_work"]["observation_queue"]["pending"], manifest["created_at"],
                    time.perf_counter() - started, len(details) + len(observations), canonical(images))
