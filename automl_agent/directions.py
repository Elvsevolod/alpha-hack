"""Measured directions and affordable next actions, without a model quota."""
from .action_cards import cards
from .util import digest


def direction_map(c):
    from .research_memory import branch_axes, branch_key, enabled
    structured = enabled(c)
    measured, failures = {}, set()
    for spec, result in c.records():
        if spec.validation_hash != c.plan.hash:
            continue
        p = spec.pipeline
        axes = {"family": p.model, "representation": p.representation,
            "features": digest([p.include_features, p.exclude_features, [x.model_dump() for x in p.transformations]]),
            "objective": p.training_objective, "decision": p.prediction_policy, "transfer": p.shift_strategy,
            "composition": p.ensemble_strategy if p.engine == "ensemble" else "single"}
        if structured:
            axes = branch_axes(p)
        key = branch_key(p) if structured else digest(axes)
        if result and result.status == "completed" and result.scope != "pilot":
            old = measured.get(key)
            row = {"axes": axes, "experiment_id": spec.experiment_id, "score": result.score, "seconds": result.seconds}
            if structured:
                best = old
                if old is None or (result.score is not None and (old['score'] is None or
                        (result.score > old['score'] if c.task.metric.maximize else result.score < old['score']))):
                    best = row
                row = {**best, 'variant_count': (old or {}).get('variant_count', 0) + 1,
                    'latest_experiment_id': spec.experiment_id, 'latest_score': result.score}
            measured[key] = row
        elif result:
            failures.add(p.hash)
    alternatives = []
    from .contracts import PipelineSpec
    from .completion import candidate_costs
    models = {row["axes"]["family"] for row in measured.values()}
    engines = {spec.pipeline.engine for spec, result in c.records() if result and result.status == 'completed'
        and result.scope != 'pilot' and spec.validation_hash == c.plan.hash}
    for card in cards(c.task, c.plan.strategy).values():
        p = PipelineSpec.model_validate(card["pipeline"])
        already_measured = (branch_key(p) in measured if structured else
            p.model in models if p.engine == 'direct' else p.engine in engines)
        if not structured and card['id'].endswith('offsets'):
            already_measured = any(row['axes']['decision'] == 'class_offsets' for row in measured.values())
        if not card["applicable"] or already_measured or p.hash in failures:
            continue
        if hasattr(c, 'capabilities'):
            try:
                c.capabilities.require_available(p)
            except (ValueError, ImportError):
                continue
        if p.model == "knn" and len(c.plan.fit) * len(c.dataset.test) > 20_000_000:
            continue
        costs = candidate_costs(c, p)
        seconds = getattr(costs, "forecast", {}).get("candidate_fit_seconds")
        reserve = c.budget.final_reserve(costs or c.finalist_costs())
        cap = min(30., max(5., seconds * 1.5)) if seconds is not None else 15.
        if seconds is not None and seconds > 30 or c.budget.remaining <= reserve + cap + 5:
            continue
        alternatives.append({"id": card["id"], "model": p.model, "budget_seconds": cap, "apply": card["apply"],
            "cost_basis": "measured" if seconds is not None else "bounded_native_pilot", "useful_quality": "unestablished"})
    return {"directions": list(measured.values()), "affordable_unattempted": alternatives,
        "technical_failure_is_poor_quality": False, "no_mandatory_model_count": True,
        "catalog_exhaustion_is_research_exhaustion": False,
        "diagnostic_questions": ["Check residual/rare-class and transfer evidence before deciding that remaining branches are not useful"]}


def finish_readiness(c, args):
    mapping = direction_map(c)
    if not args.reason.strip():
        raise ValueError("Finish requires a concrete reason")
    skipped = args.skipped_directions
    if any(not reason.strip() for reason in skipped.values()):
        raise ValueError("Skipped directions require a concrete reason")
    if "all" in skipped and len(skipped["all"].strip()) < 20:
        raise ValueError("A global finish limitation must explain why the remaining actions are not useful")
    remaining = [row for row in mapping["affordable_unattempted"] if row["id"] not in skipped and "all" not in skipped]
    # The dispatcher separately verifies completed current-protocol evidence
    # and all finalization guarantees. Untried catalog entries do not establish
    # useful experiments and must not impose a method quota or exact-ID ritual.
    return {"ready": True, "available_actions": remaining, "skipped_directions": skipped,
        "remaining_actions_are_advisory": True, "optimality_established": False,
        "completed_directions": len(mapping["directions"]), "reason": args.reason,
        "catalog_exhaustion_is_research_exhaustion": False,
        "diagnostic_questions": mapping['diagnostic_questions'],
        "finalization_reserve_seconds": c.budget.final_reserve(c.finalist_costs()), "quota_required": False}
