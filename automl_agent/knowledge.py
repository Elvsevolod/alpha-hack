"""Bind the standalone ML reference corpus and evidence memory to one task run."""
from __future__ import annotations

import json
from pathlib import Path

from ml_knowledge.context import SelectionContext, audit_request, select
from ml_knowledge.memory import KnowledgeMemory, TYPES
from ml_knowledge.runtime import ReferenceStore

from .compatibility import incompatibilities, registry_fingerprint
from .contracts import PipelineSpec
from .references import INTERFACE
from .util import atomic_json, canonical, digest


class IndexedKnowledgeMemory(KnowledgeMemory):
    """Preserve standalone memory semantics without scanning unrelated topics."""
    def __init__(self, path):
        super().__init__(path)
        self.db.executescript("""
            CREATE INDEX IF NOT EXISTS host_topic ON records(json_extract(body,'$.topic'));
            CREATE INDEX IF NOT EXISTS host_event_record_kind ON events(record_id,kind,seq);
        """)

    def _save(self, record):
        if record["type"] not in TYPES or not record.get("claim"):
            raise ValueError("A typed substantive claim is required")
        if len(record["claim"]) > 2400:
            raise ValueError("Active excerpts are limited to 2400 characters; keep full sources in the corpus")
        identifier, encoded = record["id"], canonical(record)
        previous = self.db.execute("SELECT body FROM records WHERE id=?", (identifier,)).fetchone()
        if previous:
            if previous[0] == encoded:
                return identifier
            raise ValueError("Evidence IDs are immutable; use a new ID and explicitly link the contradiction")
        with self.db:
            if record.get("topic"):
                for (raw,) in self.db.execute("SELECT body FROM records WHERE json_extract(body,'$.topic')=?", (record["topic"],)):
                    other = json.loads(raw)
                    if other["claim"] != record["claim"]:
                        self._event("possible_contradiction", identifier, {"other_id": other["id"],
                            "required_check": "Compare source, scope and conditions; neither claim is silently replaced"})
            self.db.execute("INSERT INTO records VALUES (?,?)", (identifier, encoded))
            self._event("recorded", identifier, record)
            fingerprint = digest({key: record[key] for key in ("type", "claim", "source", "dependencies")})
            self.db.execute("INSERT OR IGNORE INTO progress VALUES (?)", (fingerprint,))
        return identifier


class AgentReferences(ReferenceStore):
    def __init__(self, root, capabilities):
        corpus = Path(root) / "ml_knowledge" / "corpus"
        if not corpus.is_dir():
            # Only the delivered flat bundle supplies this import. No network,
            # legacy corpus or shared task directory is used as a fallback.
            import ml_knowledge_bundle
            corpus = ml_knowledge_bundle.restore(Path(ml_knowledge_bundle.__file__).parent / ".ml_knowledge_runtime")
        self.capabilities = capabilities
        self.context = {}
        super().__init__(corpus, verify=False)

    def verify(self, requirements=None):
        if self.disabled_reason:
            raise ValueError("Reference corpus is disabled: " + self.disabled_reason)
        return super().verify(requirements)

    def query(self, args, task=None, plan=None):
        if args.reference_id == "candidate_interface" or args.mode == "api" and args.query in {"", "candidate_interface", "build_pipeline"}:
            return self._bounded({"status": "complete", "doc_id": "candidate_interface",
                "source": "project_owned_contract", "content": INTERFACE}, args.max_chars)
        context = {**self.context, "capabilities": self.capabilities.summary(),
            "capability_registry": registry_fingerprint(self.capabilities), "environment": digest(self.capabilities.versions)}
        if task:
            context.update(task=task.kind, task_hash=task.hash,
                feature_roles=digest({key: getattr(task, key) for key in
                    ("feature_columns", "categorical_columns", "text_columns", "time_column", "group_column")}))
        if plan:
            context.update(validation=plan.strategy, validation_hash=plan.hash)
        result = super().query(args.model_dump(mode="json"), context=context)
        if result.get("status") in {"not_found", "missing_version", "ambiguous_symbol"}:
            result = self._bounded({**result, "query": args.query}, args.max_chars)
        return result

    def catalog_entry(self, directory, identifier, task=None, plan=None):
        kind = {"methods": "method", "recipes": "recipe", "search_spaces": "search_space"}[directory]
        doc = self.document(identifier if identifier.startswith(kind + ":") else kind + ":" + identifier)
        if not doc or doc["type"] != kind:
            raise ValueError("Unknown " + kind + " reference in collection " + directory + ". Use search_knowledge or list_knowledge with collection=" + directory + ", follow the returned entry.read, and copy the exact entry.id; API documentation and evidence IDs use different fields")
        entry = dict(doc["payload"])
        if kind in {"recipe", "search_space"} and task:
            recipe = entry if kind == "recipe" else self.catalog_entry("recipes", entry["id"])
            pipeline = PipelineSpec.model_validate(recipe["pipeline"])
            if kind == "search_space":
                pipeline = pipeline.model_copy(update={"engine": "optuna", "search_owner": "optuna",
                    "search_space": entry["parameters"]})
            reasons = incompatibilities(pipeline, task, plan.strategy if plan else None, self.capabilities)
            if task.kind not in recipe["task_kinds"]:
                reasons.append("unsupported_task_family")
            entry.update(applicable=not reasons, exclusion_reasons=reasons,
                runtime_status="verified" if not reasons and plan and self.capabilities.verified(pipeline, task, plan.strategy) else "unverified")
        return entry

    def known_document(self, identifier):
        return identifier == "candidate_interface" or bool((self.document(identifier) or {}).get("type") == "documentation")


class AgentKnowledge:
    def __init__(self, controller):
        self.c = controller
        self.memory = IndexedKnowledgeMemory(controller.workdir / "knowledge_memory.sqlite")
        self.memory.db.execute("CREATE TABLE IF NOT EXISTS host_operations (id TEXT PRIMARY KEY, result_sha TEXT NOT NULL)")
        self.memory.db.execute("ATTACH DATABASE ? AS host_journal", (str(controller.workdir / "journal.sqlite"),))
        self.packet = None
        self.packet_key = None

    def selection_context(self, profile=None):
        c = self.c
        if c.task is None or c.plan is None or not c.journal.get("task_confirmation"):
            raise ValueError("Knowledge selection requires a confirmed task and validation")
        if profile is None:
            profile = c.journal.get("knowledge_fit_summary")
        if profile is None:
            raise ValueError("Knowledge selection requires the cached fit aggregate summary")
        leader = c.journal.get("leader_id") or c.journal.get("baseline_id")
        pipeline = c.lookup(leader)[0].pipeline.model_dump(mode="json") if leader else {}
        fingerprints = {"data_version": c.dataset.source_hash, "task_hash": c.task.hash,
            "feature_roles": digest({key: getattr(c.task, key) for key in
                ("feature_columns", "categorical_columns", "text_columns", "time_column", "group_column")}),
            "validation_hash": c.plan.hash, "environment": digest(c.capabilities.versions),
            "pipeline_hash": digest(pipeline)}
        aggregate = {"rows": profile["rows"]["fit"], "columns": profile["feature_count"],
            "missing": bool(profile["columns_with_missing"]), "categorical": bool(c.task.categorical_columns),
            "text": bool(c.task.text_columns), "groups": bool(c.task.group_column), "time": bool(c.task.time_column)}
        aggregate.update(profile.get("risk_flags", {}))
        note = c.context_memory.store.active_note() or {}
        hypothesis = c.journal.get("active_hypothesis") or {}
        question = str(note.get("question") or hypothesis.get("changed_component") or "")[:1024]
        last = c.journal.get("last_tool_result") or {}
        error = str(last.get("error") or (last.get("result") or {}).get("error") or "")[:512]
        return SelectionContext(c.task.kind, aggregate, c.plan.strategy,
            " ".join(str(pipeline.get(key, "")) for key in ("engine", "model", "representation")),
            {"cpu": c.executor.cpu, "memory_mb": c.executor.memory_mb}, error,
            capabilities=c.capabilities.summary(), fingerprints=fingerprints,
            metric=c.task.metric.name + ("_" + c.task.metric.average if c.task.metric.name == "f1" else ""),
            research_question=question)

    def select_packet(self, profile):
        c = self.c
        c.journal.put("knowledge_fit_summary", profile["summary"])
        context = self.selection_context(profile["summary"])
        c.references.context = context.binding()
        # Replay only the evidence projection after an interrupted write. No
        # training or delivery acknowledgement is repeated here.
        from .contracts import OperationRecord
        for (record,) in self.memory.db.execute("""SELECT record FROM host_journal.operations o
            WHERE json_extract(record,'$.status') IN ('succeeded','failed')
            AND NOT EXISTS (SELECT 1 FROM host_operations h WHERE h.id=o.id)""").fetchall():
            self.completed(OperationRecord.model_validate_json(record))
        from .contracts import ExperimentSpec, ExperimentResult
        for spec, result in self.memory.db.execute("""SELECT spec,result FROM host_journal.experiments e
            WHERE result IS NOT NULL AND NOT EXISTS (SELECT 1 FROM host_operations h WHERE h.id='experiment:'||e.id)""").fetchall():
            self.experiment(ExperimentSpec.model_validate_json(spec), ExperimentResult.model_validate_json(result), context.binding())
        packet_key = digest({"context": context.binding(), "progress": self.memory.progress_count,
            "corpus": c.references.version, "disabled": c.references.disabled_reason})
        if packet_key == self.packet_key and self.packet is not None:
            return self.packet
        if c.references.disabled_reason:
            self.packet = {"status": "integrity_error", "answer_usable": False, "materials": []}
        else:
            self.packet = select(c.references, context, memory=self.memory,
                deadline=c.budget.deadline, reserve_seconds=c.budget.final_reserve(c.finalist_costs()))
        self.packet_key = packet_key
        return self.packet

    def remember(self, result, context):
        entries = list(result.get("entries", []))
        if result.get("entry"):
            entries.append(result["entry"])
        for entry in entries:
            if entry.get("type") == "documentation" and entry.get("fragment"):
                span = entry["fragment"]
                if 0 < span["end"] - span["start"] <= 2400:
                    self.memory.fact(self.c.references, entry["id"], span["start"], span["end"])
            elif entry.get("type") == "method":
                doc = self.c.references.document(entry["id"])
                if all(self.c.references.document(identifier) for identifier in doc["payload"].get("references", [])):
                    self.memory.recommendation(self.c.references, entry["id"], context=context)

    def completed(self, operation):
        """An idempotent projection; the controller journal owns execution truth."""
        c = self.c
        if not c.journal.get("knowledge_fit_summary") or c.references.disabled_reason:
            return
        sha = digest(operation.result)
        old = self.memory.db.execute("SELECT result_sha FROM host_operations WHERE id=?", (operation.operation_id,)).fetchone()
        if old and old[0] == sha:
            return
        context = self.selection_context().binding()
        if operation.tool in {"get_reference", "read_knowledge"} and operation.status == "succeeded":
            from .tool_results import payload
            self.remember(payload(operation.result or {}), context)
        if operation.experiment_id:
            spec, result = c.lookup(operation.experiment_id)
            if result is not None:
                self.experiment(spec, result, context)
                for identifier in [*spec.method_refs, *spec.reference_ids, *filter(None, (spec.recipe_ref, spec.search_space_ref))]:
                    self.memory.referenced(identifier, operation_id=operation.operation_id)
                for reference in spec.hypothesis.evidence_refs:
                    c.journal.event("evidence_experiment_link", {"operation_id": operation.operation_id,
                        "experiment_id": spec.experiment_id, "reference": reference,
                        "configuration_hash": spec.pipeline.hash, "outcome": result.status,
                        "application_proven": False})
        with self.memory.db:
            self.memory.db.execute("INSERT OR REPLACE INTO host_operations VALUES (?,?)", (operation.operation_id, sha))

    def experiment(self, spec, result, context):
        key, sha = "experiment:" + spec.experiment_id, digest(result)
        old = self.memory.db.execute("SELECT result_sha FROM host_operations WHERE id=?", (key,)).fetchone()
        if old and old[0] == sha:
            return
        context = {**context, "pipeline_hash": digest(spec.pipeline.model_dump(mode="json")),
            "validation_hash": spec.validation_hash, "task_hash": spec.task_hash,
            "data_version": spec.data_version, "environment": digest(spec.library_versions)}
        self.memory.hypothesis(spec.hypothesis.changed_component[:2400],
            observation=spec.hypothesis.observation, expected_effect=spec.hypothesis.expected_effect,
            test=spec.hypothesis.acceptance, context=context, topic="hypothesis:" + spec.experiment_id)
        if result.status not in {"outcome_unknown", "interrupted"}:
            status = result.status if result.status in {"completed", "timeout"} else "failed"
            self.memory.experiment(spec.experiment_id, context=context, controller_lookup=lambda identifier: {
                "experiment_id": spec.experiment_id, "status": status,
                "metrics": {"score": result.score, "metric": self.c.task.metric.name, "scope": result.scope, "controller_status": result.status},
                "configuration_sha256": spec.pipeline.hash})
        with self.memory.db:
            self.memory.db.execute("INSERT OR REPLACE INTO host_operations VALUES (?,?)", (key, sha))

    def audit(self, prepared, payload):
        if self.packet is None:
            raise ValueError("Knowledge packet is missing from the decision snapshot")
        audit = audit_request(payload, self.packet)
        atomic_json(self.c.workdir / "knowledge_audits" / (prepared.request_id + ".json"), audit)
        if not audit["send_allowed"]:
            raise ValueError("Knowledge evidence was lost after context compression")
        return audit

    def provided(self, prepared):
        for entry in self.packet.get("materials", []):
            self.memory.provided(entry["id"], packet_sha256=digest(self.packet))

    def close(self):
        self.memory.close()
