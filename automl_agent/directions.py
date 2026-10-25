"""Measured directions and affordable next actions, without a model quota."""
from .action_cards import cards
from .util import digest


def direction_map(c):
    measured, failures = {}, set()
    for spec, result in c.records():
        if spec.validation_hash != c.plan.hash:
            continue
        p = spec.pipeline
        axes = {"family": p.model, "representation": p.representation,
            "features": digest([p.include_features, p.exclude_features, [x.model_dump() for x in p.transformations]]),
            "objective": p.training_objective, "decision": p.prediction_policy, "transfer": p.shift_strategy,
            "composition": p.ensemble_strategy if p.engine == "ensemble" else "single"}
        key = digest(axes)
        if result and result.status == "completed" and result.scope != "pilot":
            measured[key] = {"axes": axes, "experiment_id": spec.experiment_id, "score": result.score, "seconds": result.seconds}
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
        already_measured = p.model in models if p.engine == 'direct' else p.engine in engines
        if card['id'].endswith('offsets'):
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
    skipped = args.skipped_directions
    if any(not reason.strip() for reason in skipped.values()):
        raise ValueError("Skipped directions require a concrete reason")
    if "all" in skipped and len(skipped["all"].strip()) < 20:
        raise ValueError("A global finish limitation must explain why the remaining actions are not useful")
    remaining = [row for row in mapping["affordable_unattempted"] if row["id"] not in skipped and "all" not in skipped]
    return {"ready": not remaining, "available_actions": remaining, "skipped_directions": skipped,
        "completed_directions": len(mapping["directions"]), "reason": args.reason,
        "catalog_exhaustion_is_research_exhaustion": False,
        "diagnostic_questions": mapping['diagnostic_questions'],
        "finalization_reserve_seconds": c.budget.final_reserve(c.finalist_costs()), "quota_required": False}
