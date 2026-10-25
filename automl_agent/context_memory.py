"""Connect permitted controller results to durable compressed Qwen requests."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from types import SimpleNamespace

from context_compressor import (ContextCompressor, ContextStore, Emission, ProfileCache,
    ProfileKey, Record, REQUIRED_FLAGS, Source, SourceSlice, Transport, configure, BudgetExceeded)

from .contracts import ExperimentResult, ExperimentSpec, ResearchState, TOOL_SCHEMAS, LEGACY_TOOL_SCHEMAS, TOOL_DESCRIPTIONS, ToolCommand, public_tool_schema
from .util import atomic_json, canonical, digest, redact, utf8_short
from .tool_results import operation_error, payload


def short(text, maximum=512):
    return utf8_short(text, maximum)


def context_route(command):
    return command.tool == "analyze_experiments" and bool(set(command.arguments) & {"context_read", "context_search"})


def experiment_summary(c, spec, result, source, change):
    """Prioritize measured ML feedback; every optional addition is byte bounded."""
    summary = {'model': short(spec.pipeline.model, 64), 'representation': spec.pipeline.representation,
        'imputation': {key: getattr(spec.pipeline.imputation, key) for key in ('numeric', 'categorical', 'indicators')},
        'pipeline_read': {'tool': 'read_run_records', 'arguments': {'ids': [spec.experiment_id],
            'view': 'pipeline', 'include_code': False}},
        'evidence_ref': {'kind': 'record', 'id': spec.experiment_id, 'version': source.version}}
    def add(key, value):
        if len(canonical({**summary, key: value}).encode()) <= 1536:
            summary[key] = value
            return True
        return False
    training = result.get('training_summary', {})
    stability = training.get('temporal_stability', result.get('temporal_stability', {}))
    if stability:
        scores = list(stability.get('scores', {}).items())[-3:]
        item = {key: stability.get(key) for key in ('status', 'scope', 'windows_hash')}
        item['windows'] = [{'id': short(name, 48), 'score': score} for name, score in scores]
        if scores:
            bias = stability.get('diagnostics', {}).get(scores[-1][0], {}).get('residual_bias', {})
            if len(c.task.targets) == 1 and c.task.targets[0] in bias:
                item['last_residual_bias'] = bias[c.task.targets[0]]
        for count in (3, 2, 1, 0):
            if add('temporal_stability', {**item, 'windows': item['windows'][-count:] if count else []}):
                break
    if c.task.kind == 'classification' and len(c.task.targets) == 1:
        reports = [r for r in result.get('feedback', {}).values() if r.get('classes')
            and r.get('status', 'available') == 'available']
        diagnostic = training.get('fit_diagnostics')
        if diagnostic:
            reports.append({**diagnostic, 'scope': 'fit_inner_holdout'})
        target = c.task.targets[0]
        for report in reports:
            rows = [row for row in report['classes'] if row.get('status') == 'available'
                and isinstance(row.get('f1'), (int, float)) and isinstance(row.get('support'), int)
                and row.get('class') in c.task.classes[target]]
            if rows:
                class_item = {'scope': report.get('scope'),
                    'class_order_ref': source.reference, 'weakest': [
                        {'index': c.task.classes[target].index(row['class']), 'f1': row['f1'],
                            'support': row['support'], 'uncertain': row.get('uncertain', True)}
                        for row in sorted(rows, key=lambda row: (row['f1'], c.task.classes[target].index(row['class'])))[:3]]}
                for count in (3, 2, 1):
                    if add('class_feedback', {**class_item, 'weakest': class_item['weakest'][:count]}):
                        break
                break
    change_item = {'basis': change['basis'],
        'declared': short(change['declared_component'], 64),
        'fields': [short(path, 64) for path in change['effective_fields'][:6]],
        'field_count': len(change['effective_fields']), 'claimed_field_unchanged': change['claimed_field_unchanged'],
        'mixed_change': change['mixed_change']}
    for count in (6, 3, 1, 0):
        if add('effective_change', {**change_item, 'fields': change_item['fields'][:count]}):
            break
    weakest = []
    for report_id, report in result.get('feedback', {}).items():
        if report.get('status') != 'available':
            continue
        slices = [(name, values) for name, values in report.get('slices', {}).items()
            if isinstance(values.get('score'), (int, float))]
        if slices:
            name, values = (min if c.task.metric.maximize else max)(slices, key=lambda item: item[1]['score'])
            weakest.append({'report_id': short(report_id, 64), 'slice': short(name, 64),
                'rows': values['rows'], 'score': values['score'], 'scope': report['scope']})
    for item in sorted(weakest, key=lambda row: row['score'], reverse=not c.task.metric.maximize)[:2]:
        add('weakest_slices', [*summary.get('weakest_slices', []), item])
    error = result.get('error')
    add('error', short(error, 384) if error else None)
    add('error_truncated', bool(error and summary.get('error') != error))
    parameters = spec.pipeline.hyperparameters
    if len(canonical(parameters).encode()) <= 512:
        add('parameters', parameters)
    if 'parameters' not in summary:
        add('parameters', {'complete_reference': source.reference})
    return summary


class ContextMemory:
    def __init__(self, controller):
        self.c = controller
        # Separate connections, one SQLite file: operation completion and its
        # delivery queue commit atomically, including crash recovery.
        self.store = ContextStore(controller.workdir / "journal.sqlite", tool_api_version=2)
        self.compressor = None

    def arguments(self, command):
        if context_route(command):
            self.store.handle_read(command.arguments)  # bounded, read-only validation
            return SimpleNamespace(knowledge_gap="Retrieve a versioned context fragment")
        return (TOOL_SCHEMAS.get(command.tool) or LEGACY_TOOL_SCHEMAS[command.tool]).model_validate(command.arguments)

    def validate(self, command):
        """No writes or execution inside accept_command's transaction."""
        c = self.c
        raw = command.as_dict()
        typed = ToolCommand.model_validate({k: v for k, v in raw.items() if k != "research_note"})
        if typed.tool not in TOOL_SCHEMAS:
            raise ValueError("This action is not in the public tool API v2 registry")
        args = self.arguments(typed)
        from .contracts import request_arguments
        normalized = typed.arguments if context_route(typed) else request_arguments(args)
        old = c.journal.operation(typed.operation_id)
        if old and (old.tool != typed.tool or old.request_hash != digest(normalized)):
            raise ValueError("Operation ID already has different arguments; use a new operation_id")
        if c.state != "searching":
            raise ValueError("Research tools require state searching")
        revision = c.journal.get("protocol_revision")
        if revision and revision["status"] not in {"completed", "aborted"}:
            raise ValueError("Protocol revision must be recovered first")
        if c.budget.remaining <= c.qwen.reserve_seconds:
            raise TimeoutError("Finalization reserve reached")
        if typed.tool in {"train_model", "run_automl", "tune_model", "run_custom_experiment"}:
            if args.apply_recipe:
                from .recipe_patch import resolve
                args.pipeline, _ = resolve(c, args.apply_recipe)
            from .recipe_patch import inherit_parent
            inherit_parent(c, args)
            from .compatibility import require_compatible
            engines = {"train_model": {"direct"}, "run_automl": {"flaml", "autogluon", "lightautoml"}, "tune_model": {"optuna"}}
            if typed.tool in engines and args.pipeline.engine not in engines[typed.tool]:
                raise ValueError("Tool and pipeline engine disagree")
            require_compatible(args.pipeline, c.task, c.plan.strategy, c.capabilities)
            c.make_spec(args, "admissibility_check")
            c.validate_evidence(args.hypothesis.evidence_refs)
        if typed.tool in {"inspect_dataset", "get_dataset_report"} and args.scope != "fit" and (args.include_target or args.predicate):
            raise ValueError("Target-based inspection and arbitrary predicates are restricted to fit")
        if typed.tool == "get_diagnostic_report":
            from .visual_diagnostics import DiagnosticTools
            DiagnosticTools(c).validate(args)
        if typed.tool == "analyze_experiments" and not context_route(typed):
            if args.report_id and args.report_id not in c.plan.feedback_policy.report_ids:
                raise ValueError("The requested external report is not registered")

    def persist_note(self):
        c = self.c
        note = self.store.active_note()
        state = ResearchState(task_hash=c.task.hash, validation_hash=c.plan.hash,
            questions=[note["question"]] if note else [],
            assumptions=[canonical({"source": "agent_assumption", "reading_goal": note["reading_goal"],
                "expected_evidence": note["expected_evidence"]})] if note else [],
            current_hypothesis=c.journal.get("active_hypothesis"),
            raw_best=c.journal.get("raw_best"), leader=c.journal.get("leader_id"),
            counters={"candidates": c.budget.counter("candidate_count"), "calls": c.budget.counter("llm_calls")})
        c.journal.put("research_state", state.model_dump(mode="json"))
        atomic_json(c.workdir / "research_state.json", state)

    def complete(self, operation):
        from .tool_results import envelope, payload
        raw = payload(operation.result) or {"status": operation.status, "outcome": operation.outcome}
        if operation.tool == "revise_validation" and "decision" not in raw:
            from .run_tools import revision_result
            raw = revision_result(self.c, operation.operation_id, self.c.plan.hash, raw.get("error"))
            if operation.outcome == "outcome_unknown":
                raw.update(status="outcome_unknown", revision_status="outcome_unknown")
        if operation.tool in TOOL_SCHEMAS:
            operation.result = envelope(self.c, operation, raw)
        result = operation.result or raw
        version = result.get("provenance", {}).get("result_hash", digest(result))
        scope = "reference" if operation.tool in {"get_reference", "search_knowledge", "list_knowledge", "read_knowledge"} else "registered_feedback"
        if operation.tool in {"inspect_dataset", "get_dataset_report", "get_diagnostic_report"}:
            scope = {"schema": "contract", "fit": "fit", "validation_features": "validation_features",
                "test_features": "test_features"}.get(raw.get("scope"), "controller")
        kind = "reference" if scope == "reference" else "analysis"
        if operation.status == "failed":
            scope, kind = "controller", "diagnostic"
        source = Source("operation:" + digest(operation.operation_id), version, canonical(result), scope,
            "external_data", origin=operation.tool)
        emissions = [Emission(source, kind=kind)]
        if operation.experiment_id and isinstance(raw.get("result"), dict):
            self.store.put_source(source)
            measured = raw["result"]
            card = {"experiment_id": operation.experiment_id, "status": measured.get("status"),
                "score": measured.get("score"), "scope": measured.get("scope"), "seconds": measured.get("seconds"),
                "details": self.store.route(source.reference)}
            emissions = [Emission(Source("card:" + digest(operation.operation_id), version, canonical(card),
                scope, "controller_fact", origin="training_result"), kind="training_card")]
        fragment = raw if operation.tool == "read_run_records" else result
        if (operation.tool == "read_run_records" or operation.tool not in TOOL_SCHEMAS) and all(
                key in fragment for key in ("source", "range", "content_sha256")):
            emissions = [SourceSlice.from_fragment(fragment)]
        # Preserve generated code independently of a short rejection diagnostic.
        sha = self.c.journal.get("proposed_code_sources", {}).get(operation.operation_id)
        if sha:
            from .util import file_hash
            path = self.c.workdir / "code" / f"{sha}.py"
            if file_hash(path) != sha:
                raise ValueError("Proposed code checksum mismatch")
            self.store.put_source(Source("code:" + sha, "1", path.read_text(encoding="utf-8"), "code", origin="agent_code"))
        def commit(db):
            db.execute("UPDATE operations SET record=? WHERE id=?", (canonical(operation), operation.operation_id))
            if operation.tool == "get_diagnostic_report" and raw.get("artifacts"):
                self.store._put_meta("diagnostic_image_operation", operation.operation_id)
        indexed = []
        if operation.tool in TOOL_SCHEMAS:
            indexed = [Record("operation:" + operation.operation_id, source.source_id, source.version, scope,
                kind="operation", component=short(operation.component or operation.tool, 128), engine=operation.engine or "controller", status=operation.status,
                error_code=result.get("error_code"), protocol=operation.protocol_hash or self.c.plan.hash, tool=operation.tool,
                parent_id=operation.parent_id, environment=digest(self.c.capabilities.versions)),
                Record(result["observation_id"], source.source_id, source.version, scope,
                    kind="observation", component=operation.tool, engine="controller", status=result["status"],
                    protocol=self.c.plan.hash, tool=operation.tool)]
            self.store.put_source(source)
        self.store.complete_operation(operation.operation_id, version, result, emissions, records=indexed, commit=commit)
        self.c.knowledge.completed(operation)

    def recover(self):
        # One-time migration of completed pre-integration operations. A current
        # completion is always atomic and cannot leave this migration gap.
        rows = self.c.journal.db.execute("""SELECT record FROM operations o
            WHERE json_extract(record,'$.status') NOT IN ('pending','running')
            AND NOT EXISTS (SELECT 1 FROM cc_operations c WHERE c.operation_id=o.id)""").fetchall()
        from .contracts import OperationRecord
        for (record,) in rows:
            self.complete(OperationRecord.model_validate_json(record))
        self.persist_note()

    def pending(self):
        commands = self.store.pending_commands(limit=1)
        if commands:
            return ToolCommand.model_validate({k: v for k, v in commands[0].items() if k != "research_note"})
        return None

    def index_experiments(self):
        c = self.c
        from .research import index_reviews
        self.latest_research_review = index_reviews(self)
        rows = c.journal.db.execute("""SELECT spec,result FROM experiments e WHERE result IS NOT NULL
            AND NOT EXISTS (SELECT 1 FROM cc_records r JOIN cc_sources s
                ON s.source_id=json_extract(r.payload,'$.source_id')
                AND s.version=json_extract(r.payload,'$.source_version') AND s.section='summary'
                WHERE r.record_id=e.id)""").fetchall()
        for raw_spec, raw_result in rows:
            spec, result = ExperimentSpec.model_validate_json(raw_spec), ExperimentResult.model_validate_json(raw_result)
            permitted = {"spec": spec.model_dump(mode="json"), "result": c.public_result(result)}
            source = Source("experiment:" + digest(spec.experiment_id), digest(permitted), canonical(permitted),
                "registered_feedback", origin="evaluated_experiment")
            self.store.put_source(source)
            recipe = {"experiment_id": spec.experiment_id, "pipeline": spec.pipeline.research_recipe(),
                "evidence_ref": {"kind": "record", "id": spec.experiment_id, "version": source.version}}
            self.store.put_source(Source(source.source_id, source.version, canonical(recipe), source.scope,
                "external_data", section="pipeline", origin=source.origin))
            from .research import effective_change
            parent = c.lookup(spec.parent_id)[0].pipeline if spec.parent_id else None
            change = effective_change(spec.pipeline, parent, spec.hypothesis)
            summary = experiment_summary(c, spec, permitted['result'], source, change)
            self.store.put_source(Source(source.source_id, source.version, canonical(self.store.route(summary)), source.scope,
                "external_data", section="summary", origin=source.origin))
            self.store.publish_records([Record(spec.experiment_id, source.source_id, source.version,
                component=short(spec.hypothesis.changed_component, 128), engine=spec.pipeline.engine,
                status=result.status, error_code=result.status if result.status != "completed" else None,
                protocol=spec.validation_hash, environment=digest(spec.library_versions),
                parent_id=spec.parent_id, metric=c.task.metric.name, maximize=c.task.metric.maximize,
                score=result.score, seconds=result.seconds, peak_memory_mib=result.peak_memory_mb,
                evaluation_scope=result.scope, checked_change=short("effective: " + ", ".join(change["effective_fields"])
                    if spec.parent_id else "standalone: " + spec.hypothesis.changed_component),
                dependencies=tuple(spec.pipeline.members[:8]))])

    def snapshot(self, system):
        started = time.perf_counter()
        c = self.c
        c.qwen.select_for_step()
        self.index_experiments()
        unified = getattr(c.dataset, "profile_mode", "legacy") == "unified"
        profile = ProfileCache(self.store).get_or_create(ProfileKey(c.dataset.source_hash,
            digest([c.task.hash, c.plan.hash, c.plan.fit]) if unified else digest(c.plan.fit), format_version="6" if unified else "3"),
            lambda: c.dataset.profile(c.plan.fit, protocol_hash=c.plan.hash).model_dump(mode="json"))
        # Always retain complete contract/text and environment behind versioned
        # routes; none of these sources can expose private control labels.
        self.store.put_source(Source("task_description", c.dataset.source_hash, c.dataset.text, "contract", origin="task_file"))
        cited_sources: dict[str, set[str]] = {}
        for field, references in c.task.evidence.items():
            for reference in references:
                if reference.startswith(("description:", "description_csv:")):
                    cited_sources.setdefault(reference, set()).add(field)
        for reference, fields in sorted(cited_sources.items()):
            source = c.dataset.sources.get(reference)
            if source:
                content = canonical({"source_id": reference, "cited_for": sorted(fields),
                    "sha256": source["sha256"], "text": source["text"]})
                self.store.put_source(Source("task_material:" + digest(reference)[:24], source["sha256"],
                    content, "contract", "external_data", origin="task_supporting_material"))
        capabilities = c.capabilities.summary()
        self.store.put_source(Source("environment", digest(capabilities), canonical(capabilities), "environment", origin="capabilities"))
        reserve = c.budget.final_reserve(c.finalist_costs())
        counters = {"candidates": c.budget.counter("candidate_count"), "calls": c.budget.counter("llm_calls")}
        policies = {"feedback": c.plan.feedback_policy.model_dump(mode="json"),
            "selection": c.plan.selection_policy.model_dump(mode="json"), "validation_strategy": c.plan.strategy,
            "validation_reason": c.plan.reason, "execution": c.journal.get("execution_policy"),
            "budget_mode": c.budget_mode, "training_seed": c.training_seed,
            "model_response_limit_seconds": c.qwen.response_limit_seconds}
        active = c.journal.get("active_hypothesis")
        work = {"baseline_id": c.journal.get("baseline_id"), "leader_id": c.journal.get("leader_id"),
            "suggested_new_operation_id": "step_" + str(counters["calls"] + 1)}
        if work["leader_id"]:
            work["leader_pipeline"] = {"tool": "read_run_records", "arguments": {"ids": [work["leader_id"]], "view": "pipeline"}}
        from .research import research_options, compact_research_options
        options = research_options(c)
        version = digest(options)
        details = self.store.put_source(Source("research_options", version, canonical(options),
            "controller", section="details", origin="research_coverage"))
        options = compact_research_options(options, details)
        work["research_options"] = self.store.put_source(Source("research_options", version, canonical(options),
            "controller", origin="research_coverage"))
        intents = c.journal.get("unexecuted_native_intents", [])
        if intents:
            content = {"requires_new_decision": True, "executed": False, "proposals": intents}
            work["unexecuted_intents"] = self.store.put_source(Source("unexecuted_native_intents", digest(content), canonical(content),
                "controller", "agent_assumption", origin="unexecuted_model_proposals"))
        work["related_ids"] = [self.research_guide_id]
        latest = c.journal.db.execute("SELECT id FROM experiments WHERE result IS NOT NULL ORDER BY rowid DESC LIMIT 1").fetchone()
        if latest:
            work["latest_id"] = latest[0]
            work["related_ids"].append(self.latest_research_review)
        last = payload(c.journal.get("last_tool_result") or {})
        last_error = last.get("error") or (last.get("result") or {}).get("error")
        if last_error:
            diagnostic = {'status': last.get('status'), 'error': redact(last_error),
                'read_routes': last.get('read_routes', [])}
            content = canonical(diagnostic)
            self.store.put_source(Source("last_command_error", digest(content), content, "controller", origin="controller_error"))
            work["diagnostic"] = short(canonical({**diagnostic, 'error': short(diagnostic['error'], 160)}))
        if active:
            reference = self.store.put_source(Source("active_hypothesis", digest(active), canonical(active), "controller",
                "agent_assumption", origin="agent_hypothesis"))
            work["active_operation_id"] = active.get("operation_id")
            caps = {"changed_component": 1024, "observation": 256, "expected_effect": 256, "acceptance": 128}
            hypothesis = {key: short(active.get(key, ""), cap) for key, cap in caps.items()}
            hypothesis.update(source="agent_assumption", read=reference)
            while len(canonical(hypothesis).encode()) > 1664:
                optional = [name for name in caps if name != "changed_component" and len(hypothesis[name].encode()) > 32]
                key = max(optional or list(caps), key=lambda name: len(canonical(hypothesis[name]).encode()))
                hypothesis[key] = short(hypothesis[key], max(0, len(hypothesis[key].encode()) // 2))
            hypothesis["truncated_fields"] = [key for key in caps if hypothesis[key] != active.get(key, "")]
            work["hypothesis"] = hypothesis
        # Keep IDs with their failures: merging solely by error text made older
        # rejected operations impossible to identify without another read.
        rejected = []
        for (raw,) in c.journal.db.execute("""SELECT record FROM operations
            WHERE json_extract(record,'$.status')='failed' ORDER BY rowid DESC LIMIT 4"""):
            operation = json.loads(raw)
            item = {"operation_id": operation["operation_id"], "error": short(operation_error(operation), 192)}
            if len(canonical([*rejected, item]).encode()) <= 768:
                rejected.append(item)
        if rejected:
            work["rejected_operations"] = rejected
        flags = {**REQUIRED_FLAGS, "stagnation": c.budget.counter("stagnation_count"),
            "unproductive_reads": c.budget.counter("unproductive_reads"),
            "model_imputation_attempted": c.journal.get("model_imputation_attempted", False)}
        configure(self.store, run_id=c.run_id, task=c.task.model_dump(mode="json"), policies=policies,
            policy_flags=flags, data_hash=c.dataset.source_hash, protocol_hash=c.plan.hash,
            environment_hash=digest(c.capabilities.versions), deadline_epoch=c.budget.deadline,
            reserve_seconds=reserve, cpu=c.executor.cpu, memory_mib=int(c.executor.memory_mb),
            counters=counters, work=work, profile=profile)
        if c.research_focus:
            system += "\nDevelopment objective, subject to all fixed contracts and permissions: " + canonical(c.research_focus)
        transport = Transport(system=system, tool_schemas={k: {**public_tool_schema(v), "description": TOOL_DESCRIPTIONS[k]} for k, v in TOOL_SCHEMAS.items()},
            model=c.qwen.model, response_reserve=c.qwen.output_tokens, extra=c.qwen.extra,
            reference_packet=c.knowledge.select_packet(profile))
        self.compressor = ContextCompressor(self.store, transport)
        images = []
        image_operation = self.store.get_meta("diagnostic_image_operation")
        if image_operation and not c.journal.get("vision_disabled"):
            operation = c.journal.operation(image_operation)
            from .visual_diagnostics import DiagnosticTools
            try:
                image = DiagnosticTools(c).image(image_operation, payload(operation.result))
                if image:
                    images.append(image)
            except (OSError, ValueError):
                c.journal.event("diagnostic_image_unavailable", {"operation_id": image_operation})
        with self.store.transaction():
            self.store._put_meta("diagnostic_images", images)
        snapshot = self.compressor.snapshot()
        for mode in (["json"] if c.qwen.transport == "json" else ["native", "json"]):
            try:
                self.compressor.preflight(snapshot, budget=c.qwen.context_tokens, modes=(mode,))
                break
            except BudgetExceeded as error:
                if mode == "json":
                    raise
                c.qwen.transport = "json"
                c.journal.put("transport", "json")
                c.journal.event("native_fallback", {"reason": str(error)[:500], "phase": "context_preflight"})
        self.persist_note()
        return replace(snapshot, build_seconds=time.perf_counter() - started)
