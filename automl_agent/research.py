"""Task-independent research guidance and controller-measured experiment reviews."""
from __future__ import annotations
import json

from context_compressor import Record, Source

from .contracts import ExperimentResult, ExperimentSpec, PipelineSpec
from .util import atomic_json, canonical, digest


def effective_change(pipeline, parent, hypothesis):
    """Compare normalized fresh recipes, independently of the stated intent."""
    if parent is None:
        return {'basis': 'standalone_recipe', 'declared_component': hypothesis.changed_component,
            'effective_fields': [], 'claimed_field_unchanged': None, 'mixed_change': False}
    from .pipelines import effective_parameters
    def recipe(spec):
        value = spec.research_recipe()
        if spec.engine in {'direct', 'optuna'}:
            value['hyperparameters'] = effective_parameters(spec)
        return value
    def paths(before, after, prefix=''):
        result = []
        for key in sorted(before.keys() | after.keys()):
            path = prefix + key
            if key in before and key in after and isinstance(before[key], dict) and isinstance(after[key], dict):
                result.extend(paths(before[key], after[key], path + '.'))
            elif key not in before or key not in after or before[key] != after[key]:
                result.append(path)
        return result
    try:
        before, after = recipe(parent), recipe(pipeline)
    except ValueError as error:
        return {'basis': 'normalization_unavailable', 'declared_component': hypothesis.changed_component,
            'effective_fields': [], 'claimed_field_unchanged': None, 'mixed_change': False,
            'reason': str(error), 'causality_established': False}
    fields = paths(before, after)
    claimed = hypothesis.changed_component.lower().strip()
    aliases = {'log1p': 'target_transform', 'log_transform': 'target_transform',
        'missing_indicators': 'imputation.indicators', 'model_family': 'model'}
    claimed = aliases.get(claimed, claimed)
    known = claimed.split('.')[0] in before
    return {'basis': 'normalized_parent_recipe', 'declared_component': hypothesis.changed_component,
        'effective_fields': fields, 'claimed_field_unchanged':
            not any(path == claimed or path.startswith(claimed + '.') for path in fields) if known else None,
        'mixed_change': len({path.split('.')[0] for path in fields}) > 1,
        'imputation': after['imputation'], 'causality_established': False}


RESEARCH_POLICY = """Classical ML only: no PyTorch, sklearn neural_network, MLP, neural embeddings or neural AutoML learners.
Use the existing knowledge tools to read source conditions; unsupported classical models need a separately validated permitted custom path.
Read guides as needed; collection=actions has executable cards.
current_work.leader_pipeline returns the full recipe. Children inherit all omitted fresh-fit fields and companions, never learned state. Same-model partial hyperparameters merge; changed_fields can explicitly reset a field.
List intentional removals in hypothesis.changed_fields; listed omissions reset.
research_options is a small lazy summary; read its details as needed.
Unattempted is not exhausted or inferior. An independent draft need not copy the leader.
current_work.unexecuted_intents contains unvalidated proposals.
Reassess with a new operation ID; never assume execution.
For long records follow the returned continuation; include_code is only for custom source, not declarative recipes.
For a pipeline use read_run_records(ids=[experiment_id], view=pipeline, include_code=false).
For residuals/confusion_matrix use get_diagnostic_report(experiment_id=..., scope=fit, columns=[], include_target=false).
Use preview_recipe/apply_recipe for a verified base and set/remove changes.
Copy recipe_hash from the complete pipeline read into base_hash; diagnostic source_recipe_hash is provenance, not a preview base.
Copy returned read routes, IDs and hashes exactly; never guess them or repair arguments by substituting identities.
Choose a bounded check for an observed error or unresolved question. Distinct representations/families are options, not a required sequence before local changes or finish. Use run_experiments for independent hypotheses.
Starter search ranges are examples, not API limits: verify supported typed alternatives in the source documentation.
Distinguish recipes from model families. Individual scores do not establish ensemble utility; inspect joint errors or strict OOF evidence.
Log transforms do not establish corrected tail bias: read measured residuals in the original scale.
Link each proposed experiment to an observed error and a falsifiable expected effect. Effective parent diffs govern attribution; mixed changes are not single-component ablations.
Finish may explain skipped_directions; no quota. For component-loss warnings, retrain or confirm component_audit_hash. Preserve finalization reserve.
Tied aggregate scores do not prove identical errors, irreducible error or optimality. Finish may retain a measured incumbent with explicit unknowns; affordable unattempted cards are suggestions, not required fits.
Search feedback, strict OOF and external quality are different evidence. Small rare-class gains remain uncertain.
Domain separation proposes a bounded weights/subset experiment; it never automatically removes features.
Score final labels after composition/decision and regression in the original scale including tail SSE.
An unavailable API may drain only durably admitted work and never reopens control.
"""

from .human_practice import GUIDE

RESEARCH_GUIDE = '\nVisual diagnostics: get_diagnostic_report returns numeric evidence plus an optional\nimage. Compare quantiles, skew, missingness and histogram shape; a long tail suggests\na transform/loss hypothesis, never automatic deletion. For confusion_matrix use the\nfull fit-inner matrix, declared class order and per-class precision/recall/F1;\nrare-class suppression cannot be reconstructed. Residuals are y_true-minus-y_pred\nin the original scale: inspect bias and aggregated prediction intervals. Curvature\nor changing spread is a hypothesis, not proof of a cause. Cite the numeric report,\nconsider an alternative explanation and compare one bounded experiment. Reports\nare exploratory fit-only evidence; group/time boundaries and final reserves bind.\n' + GUIDE


def experiment_review(spec, result, previous, task, policy):
    """Summarize permitted scores without promoting a local outcome into global knowledge."""
    candidates = [(s, r) for s, r in previous
        if r.status == "completed" and r.score is not None and r.scope != "pilot"
        and (s.task_hash, s.data_version, s.validation_hash, s.library_versions)
        == (spec.task_hash, spec.data_version, spec.validation_hash, spec.library_versions)
        and r.validation_hash == result.validation_hash == spec.validation_hash]
    parent = next(((s, r) for s, r in candidates if s.experiment_id == spec.parent_id), None)
    incumbent = (max if task.metric.maximize else min)(candidates, key=lambda pair: pair[1].score) if candidates else None
    comparator = parent or incumbent
    gain = None
    if result.status in {"outcome_unknown", "interrupted"}:
        outcome, next_step = "unresolved", "Recover the existing operation before interpreting or repeating it."
    elif result.status != "completed":
        outcome, next_step = "technical_failure", "Read the diagnostic; repair the evidenced technical cause within budget."
    elif result.scope == "pilot":
        outcome, next_step = "pilot_only", "Use this only for feasibility/cost; obtain a comparable full result if affordable."
    elif result.score is None or not comparator:
        outcome, next_step = "not_comparable", "Establish a scored comparator on the same protocol before a quality claim."
    else:
        reference = comparator[1].score
        gain = result.score - reference if task.metric.maximize else reference - result.score
        improved = policy.improvement(result.score, reference)
        outcome = "improved_locally" if improved else "no_material_gain"
        next_step = ("Check acceptance, costs and failure slices; retain this as conditional evidence only." if improved else
            "Preserve the incumbent; record the negative result and read sources for a distinct error-based hypothesis if affordable.")
        if improved and not policy.improvement(result.score, incumbent[1].score):
            outcome, next_step = "improved_parent_only", "Parent improved but the incumbent remains better; preserve it and review the hypothesis."
    parent_recipe = next((s.pipeline for s, _ in previous if s.experiment_id == spec.parent_id), None)
    return {"experiment_id": spec.experiment_id, "outcome": outcome,
        "effective_change": effective_change(spec.pipeline, parent_recipe, spec.hypothesis),
        "hypothesis": {"source": "agent_assumption", **spec.hypothesis.model_dump(mode="json")},
        "measurement": {"status": result.status, "score": result.score, "scope": result.scope,
            "metric": task.metric.model_dump(mode="json"), "maximize": task.metric.maximize,
            "seconds": result.seconds, "peak_memory_mb": result.peak_memory_mb,
            "fit_rows": result.training_summary.get("rows") if type(result.training_summary.get("rows")) is int else None,
            "auxiliary_fits": result.training_summary.get("auxiliary_fits") if type(result.training_summary.get("auxiliary_fits")) is int else None},
        "paired_comparison": result.training_summary.get("paired_comparison", {"status": "unavailable"}),
        "comparison": {"experiment_id": comparator[0].experiment_id if comparator else None,
            "basis": "parent" if parent else "best_prior_comparable" if comparator else "none",
            "score": comparator[1].score if comparator else None, "gain": gain,
            "incumbent_id": incumbent[0].experiment_id if incumbent else None,
            "incumbent_score": incumbent[1].score if incumbent else None,
            "acceptance": "requires_hypothesis_review", "causality_established": False},
        "reproduction": {"task_hash": spec.task_hash, "data_version": spec.data_version,
            "validation_hash": spec.validation_hash, "pipeline_hash": spec.pipeline.hash,
            "pipeline": spec.pipeline.model_dump(mode="json"), "library_versions": spec.library_versions,
            "budget_seconds": spec.budget_seconds, "method_refs": spec.method_refs,
            "reference_ids": spec.reference_ids},
        "source_status": "citations_are_not_proof_of_reading_or_application",
        "next_step": next_step, "transfer_status": "task_local_requires_revalidation"}


def index_reviews(context_memory):
    """Idempotent projection of durable experiment records; never read worker/data files."""
    c, store = context_memory.c, context_memory.store
    guide_id = "research_guide:" + digest(RESEARCH_GUIDE)[:16]
    if not store.record(guide_id):
        guide = Source(guide_id, digest(RESEARCH_GUIDE), RESEARCH_GUIDE, "reference", "external_data",
            origin="project_research_guidance")
        store.put_source(guide)
        store.publish_records([Record(guide_id, guide.source_id, guide.version, guide.scope,
            kind="reference", component="research_guide", engine="controller", status="available",
            checked_change="Read once: source-led research cycle; conditional methods, honest validation and negative results.")])
    context_memory.research_guide_id = guide_id
    previous, latest = [], None
    for row in c.journal.experiments():
        spec = ExperimentSpec.model_validate(row["spec"])
        if row["result"] is None:
            continue
        result = ExperimentResult.model_validate(row["result"])
        # Reviews contain an explicit whitelist; private feedback/predictions are
        # never included. The digest also changes if a recovered comparator changes.
        public = c.public_result(result)
        measured = result.model_copy(update={"score": public.get("score"), "seconds": public.get("seconds", 0),
            "training_summary": public.get("training_summary", {})})
        review = experiment_review(spec, measured, previous, c.task, c.plan.selection_policy)
        identifier = "research_review:" + digest(review)[:24]
        if not store.record(identifier):
            source = Source(identifier, digest(review), canonical(review), "registered_feedback",
                "external_data", origin="research_review")
            store.put_source(source)
            atomic_json(c.workdir / "research_reviews" / (identifier.split(":")[1] + ".json"), review)
            store.publish_records([Record(identifier, source.source_id, source.version, source.scope,
                kind="observation", component="research_review", engine="controller", status=result.status,
                protocol=spec.validation_hash, environment=digest(spec.library_versions),
                parent_id=spec.experiment_id, metric=c.task.metric.name, maximize=c.task.metric.maximize,
                evaluation_scope=result.scope, checked_change=review["outcome"] + ": " + review["next_step"])])
        previous.append((spec, measured))
        latest = identifier
    return latest


def research_options(c):
    """Expose coverage and real prerequisites, without prescribing a model quota."""
    from .compatibility import incompatibilities
    from .tool_results import payload
    templates = {name: PipelineSpec(engine=engine, model='automl', learners=learners, autogluon_profile=profile)
        for name, engine, learners, profile in [
            ('lightautoml', 'lightautoml', ['lgb'], 'selection'),
            ('autogluon_selection', 'autogluon', ['GBM'], 'selection'),
            ('autogluon_bagging', 'autogluon', ['GBM'], 'bagging'),
            ('autogluon_stacking', 'autogluon', ['GBM'], 'stacking'),
            ('flaml', 'flaml', ['lgbm'], 'selection')]}
    templates.update({name: PipelineSpec(model='lightgbm', shift_strategy=name)
        for name in ('importance_weighting', 'pseudo_label')})
    templates['optuna'] = PipelineSpec(engine='optuna', model='lightgbm')
    names = [*templates, 'weighted_oof', 'stacked_oof', 'history', 'related_tables']
    rows = {name: {'branch': name, 'status': 'no_admitted_attempt', 'measured_candidates': 0,
                   'technical_failures': 0} for name in names}

    def branches(pipeline):
        if not isinstance(pipeline, dict):
            return []
        result = []
        engine = pipeline.get('engine', 'direct')
        if engine == 'autogluon': result.append('autogluon_' + str(pipeline.get('autogluon_profile', 'selection')))
        elif isinstance(engine, str) and engine in templates: result.append(engine)
        if pipeline.get('shift_strategy', 'none') != 'none': result.append(pipeline['shift_strategy'])
        if engine == 'ensemble': result.append(pipeline.get('ensemble_strategy', 'uniform'))
        transforms = pipeline.get('transformations')
        if isinstance(transforms, list) and any(isinstance(op, dict) and op.get('name') == 'history' for op in transforms): result.append('history')
        return [name for name in result if isinstance(name, str) and name in rows]

    for spec, result in c.records():
        for name in branches(spec.pipeline.model_dump(mode='json')):
            row = rows[name]
            if result and result.status == 'completed' and result.scope != 'pilot' and result.score is not None:
                row['measured_candidates'] += 1
            elif result and result.status != 'completed': row['technical_failures'] += 1
    for seq, raw in c.journal.db.execute("SELECT seq,payload FROM events WHERE kind='qwen_decision' ORDER BY seq"):
        command = json.loads(raw)['command']
        args = command.get('arguments') or {}
        candidates = args.get('candidates')
        pipelines = [item.get('pipeline', {}) for item in candidates if isinstance(item, dict)] if command['tool'] == 'run_experiments' and isinstance(candidates, list) else [args.get('pipeline', {})]
        named = [name for pipeline in pipelines for name in branches(pipeline)]
        if command['tool'] == 'build_ensemble' and isinstance(args.get('strategy'), str) and args['strategy'] in rows: named.append(args['strategy'])
        operation = c.journal.operation(command['operation_id'])
        for name in set(named):
            row = rows[name];row['last_proposed_operation'] = command['operation_id']
            row['status'] = 'proposed_not_measured'
            if operation and (operation.status == 'failed' or payload(operation.result).get('status') in ('technical_failure', 'budget_too_small', 'outcome_unknown')):
                row['technical_failures'] += 1
                row['last_error'] = str(payload(operation.result).get('error', 'operation failed'))[:300]
    for name, row in rows.items():
        if row['measured_candidates']: row['status'] = 'measured'
        elif row['technical_failures']: row['status'] = 'technical_failure'
        reasons = incompatibilities(templates[name], c.task, c.plan.strategy) if name in templates else []
        if name == 'pseudo_label' and c.task.kind != 'classification': reasons.append('requires single-output classification')
        if name == 'history': reasons.append('requires evidence for an entity key and a time role; exact-feature equality is not identity')
        if name == 'history' and not c.task.time_column: reasons.append('no confirmed time role')
        if name == 'related_tables' and not c.task.related_tables: reasons.append('no confirmed supplied related tables')
        if name in ('weighted_oof', 'stacked_oof'):
            available = sum(bool(result and result.status == 'completed' and result.scope != 'pilot'
                and spec.validation_hash == c.plan.hash and spec.pipeline.engine not in ('custom', 'ensemble'))
                for spec, result in c.records())
            if available < 2: reasons.append('requires two completed built-in members on the same protocol')
        row['prerequisites'] = reasons
        if row['status'] == 'no_admitted_attempt' and reasons:
            row['status'] = 'requires_evidence' if name == 'history' and c.task.time_column else 'not_applicable_to_current_contract'
    from .directions import direction_map
    temporal_support = None
    if c.plan.strategy in {'temporal', 'temporal_group'}:
        from .validation import ordered_periods
        periods = ordered_periods(c.dataset.train.iloc[c.plan.fit][c.task.time_column]).nunique()
        horizon = c.task.horizon if c.task.kind == 'forecasting' else 1
        temporal_support = {'fit_periods':int(periods), 'strict_oof_transitions':max(0, (int(periods)-1)//horizon),
            'earliest_schedule':'declared fixed cap when no earlier inner split exists',
            'small_history_is_quality_evidence':False}
    return {'scope': 'admitted commands and measured research records; no hidden-control feedback', 'branches': list(rows.values()),
        'validation_protocol':{'mode':c.plan.evaluation_mode, 'control_revision_allowed':False,
            'initial_decision':next((row for row in c.plan.observations if row.get('type') == 'initial_protocol_decision'), None),
            'temporal_support':temporal_support},
        'direction_map': direction_map(c),
        'policy': 'Choose evidence-based affordable alternatives. No branch quota; technical failure is not measured poor quality. AutoML needs a fresh compatible recipe rather than the incumbent transforms/policy.'}


def compact_research_options(options, details):
    """Bound the first read to 3 KiB; full immutable evidence stays lazy."""
    mapping = options["direction_map"]
    compact = {"details": details, "branches": [], "branch_count": len(options["branches"]),
        "completed_directions": len(mapping["directions"]),
        "affordable_unattempted_count": len(mapping["affordable_unattempted"]),
        "affordable_unattempted": [], "technical_failure_is_poor_quality": False,
        "no_mandatory_model_count": True}
    for row in options["branches"]:
        card = {key: row[key] for key in ("branch", "status", "measured_candidates", "technical_failures")}
        candidate = {**compact, "branches": [*compact["branches"], card]}
        if len(canonical(candidate).encode("utf-8")) > 3072:
            break
        compact = candidate
    for row in mapping["affordable_unattempted"][:3]:
        card = {key: row[key] for key in ("id", "model", "budget_seconds", "cost_basis")}
        candidate = {**compact, "affordable_unattempted": [*compact["affordable_unattempted"], card]}
        if len(canonical(candidate).encode("utf-8")) > 3072:
            break
        compact = candidate
    return compact
