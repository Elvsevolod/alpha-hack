"""Durable protocol replacement; the active protocol changes only at commit."""
from pathlib import Path
import time

from .contracts import ExperimentSpec, PredictionBundle, ValidationPlan
from .evaluation import choose_leader
from .publication import Publisher
from .util import atomic_json, file_hash, read_json


def refit_composition(c, spec, item):
    """Relearn only from revised fit; member OOF owns durable no-retry identities."""
    from .learned_blend import oof_forecast, member_oof, fit_blend, fit_final_layer
    deadline = item["composition_deadline"]
    forecast = oof_forecast(c, spec.pipeline.members)
    available = min(deadline - time.time(), c.budget.remaining - c.budget.final_reserve(c.finalist_costs()))
    c.journal.event("protocol_oof_cost_forecast", {"experiment_id": spec.experiment_id,
        "available_seconds": available, **forecast})
    if forecast["seconds"] > available:
        raise ValueError("Insufficient complete-fold OOF budget for protocol composition refit")
    reports = []
    for index, member in enumerate(spec.pipeline.members):
        remaining = min(deadline - time.time(), c.budget.remaining - c.budget.final_reserve(c.finalist_costs()))
        if remaining < .2:
            raise ValueError("Protocol OOF allocation exhausted before all members completed")
        pending = sum(forecast["member_seconds"][name] for name in spec.pipeline.members[index:])
        allocation = remaining * forecast["member_seconds"][member] / pending if pending else remaining
        reports.append(member_oof(c, member, max(.1, allocation)))
    labels = c.dataset.train.iloc[c.plan.fit][c.task.targets].copy()
    labels.index = [c.dataset.row_keys[i] for i in c.plan.fit]
    pipeline = spec.pipeline.model_copy(update={"ensemble_weights": None, "ensemble_meta": {}, "learned_decision": {}})
    if pipeline.ensemble_strategy != "uniform":
        weights, meta, evidence = fit_blend(c.task, reports, labels, pipeline.ensemble_strategy, 42)
        pipeline = pipeline.model_copy(update={"ensemble_weights": weights, "ensemble_meta": meta})
        c.journal.event("protocol_oof_blend_fitted", {"experiment_id": spec.experiment_id,
            "protocol_hash": c.plan.hash, **evidence})
    if item["composition_refit"]["final_decision"] == "regularized":
        pipeline = fit_final_layer(c.task, reports, labels, pipeline)
    if time.time() >= deadline:
        raise ValueError("Protocol composition refit exhausted the candidate allocation")
    return spec.model_copy(update={"pipeline": pipeline})


def continue_revision(controller):
    c = controller
    operation = c.journal.get("protocol_revision")
    if not operation or operation["status"] in {"completed", "aborted"}:
        return operation
    old = ValidationPlan.model_validate(operation["old"])
    revised = ValidationPlan.model_validate(operation["proposed"])
    if operation["deadline"] != c.budget.deadline or old.control != revised.control:
        raise ValueError("Protocol recovery changed the original deadline or closed control")
    if operation["status"] == "committed":
        c.publisher.recover()
        operation["status"] = "completed"
        c.journal.put("protocol_revision", operation)
        atomic_json(c.workdir / "validation_plan.json", revised)
        return operation
    try:
        for item in operation["candidates"]:
            spec = ExperimentSpec.model_validate(item["spec"])
            if (spec.pipeline.engine == "ensemble" and "composition_refit" not in item
                    and (spec.pipeline.ensemble_strategy != "uniform" or spec.pipeline.learned_decision)):
                raise ValueError("Legacy protocol transaction lacks fit-only composition refit provenance")
            directory = c.workdir / "attempts" / spec.experiment_id
            path = directory / "worker_result.json"
            if item["status"] == "completed":
                if not path.is_file() or file_hash(path) != item["manifest_sha256"]:
                    raise ValueError("Completed protocol attempt manifest checksum mismatch")
                c.verify_dependency_artifacts(spec.experiment_id)
                continue
            c.plan = revised
            try:
                if item["status"] == "pending":
                    costs = c.finalist_costs()
                    if c.budget.remaining < spec.budget_seconds + c.budget.final_reserve(costs):
                        raise ValueError("Insufficient remaining time to complete protocol revision")
                    c.budget.claim_candidate()
                    item["status"] = "refitting" if item.get("composition_refit") else "ready"
                    if item.get("composition_refit"):
                        item["composition_started"] = time.time()
                        item["composition_deadline"] = time.time() + spec.budget_seconds
                    c.journal.put("protocol_revision", operation)
                if item["status"] == "refitting":
                    from .learned_blend import OOFCacheError
                    try:
                        spec = refit_composition(c, spec, item)
                    except OOFCacheError as error:
                        item["status"] = "outcome_unknown" if error.detail["status"] == "outcome_unknown" else "technical_failure"
                        item["oof_error"] = error.detail
                        raise
                    item.update(spec=spec.model_dump(mode="json"), status="ready",
                        research_oof_seconds=time.time() - item["composition_started"])
                    c.journal.put("protocol_revision", operation)
                if item["status"] == "running":
                    if not path.is_file():
                        item["status"] = "outcome_unknown"
                        raise ValueError("Interrupted protocol attempt has unknown outcome; no blind retry")
                    raw = read_json(path)
                    if raw.get("status") != "completed":
                        raise ValueError("Recovered protocol attempt failed")
                    for relative, sha in raw.get("hashes", {}).items():
                        candidate = (directory / relative).resolve()
                        if not candidate.is_relative_to(directory.resolve()) or file_hash(candidate) != sha:
                            raise ValueError("Protocol recovery artifact checksum mismatch")
                    result = c.evaluate_worker(spec, raw, directory)
                elif item["status"] == "ready":
                    if item.get("composition_refit") and time.time() >= item["composition_deadline"]:
                        raise ValueError("Protocol composition allocation expired before execution")
                    c.journal.save_experiment(spec)
                    item["status"] = "running"
                    c.journal.put("protocol_revision", operation)
                    kwargs = {"execution_deadline": item["composition_deadline"]} if item.get("composition_refit") else {}
                    result = c.execute_experiment(spec, automatic_pilot=False, **kwargs)
                else:
                    raise ValueError("Protocol attempt cannot be resumed safely: " + item["status"])
                if item.get("composition_refit") and path.is_file():
                    raw = read_json(path)
                    seconds = item["research_oof_seconds"]
                    if "research_oof_seconds" not in raw["training_summary"]:
                        raw["training_summary"]["research_oof_seconds"] = seconds
                        raw["seconds"] += seconds
                        atomic_json(path, raw)
                    result.training_summary["research_oof_seconds"] = seconds
                    result.seconds = raw["seconds"]
                    result.artifacts["worker_manifest_sha256"] = file_hash(path)
                c.journal.save_experiment(spec, result)
                if result.status != "completed" or result.score is None:
                    item["status"] = result.status
                    raise ValueError("Protocol re-evaluation failed: " + result.status)
                item.update(status="completed", manifest_sha256=file_hash(directory / "worker_result.json"))
                c.journal.put("protocol_revision", operation)
            finally:
                c.plan = old
        results = [c.lookup(item["spec"]["experiment_id"])[1] for item in operation["candidates"]]
        leader, raw_best = choose_leader(results, revised.selection_policy)
        if leader is None:
            raise ValueError("Protocol revision has no eligible leader")
        winner, _ = c.lookup(leader.experiment_id)
        bundle = PredictionBundle.model_validate(read_json(Path(leader.artifacts["directory"]) / "test_predictions.json"))
        publisher = Publisher(c.dataset, revised, c.workdir, c.journal)
        pending = publisher.prepare(bundle, winner.experiment_id, "fit", winner.pipeline.members,
            {"score": leader.score, "protocol_revision": operation["operation_id"]})
        operation.update(status="committed", publication=pending)
        c.journal.put_many({"validation": revised.model_dump(mode="json"), "baseline_id": operation["replacements"][operation["baseline_id"]],
            "leader_id": leader.experiment_id, "raw_best": raw_best, "improvement_anchor": raw_best,
            "stagnation_count": 0, "protocol_revision": operation, "publication_pending": pending},
            event=("protocol_revision_commit", {"operation_id": operation["operation_id"], "old": old.hash, "new": revised.hash}))
        c.plan, c.publisher = revised, publisher
        publisher.recover()
        operation["status"] = "completed"
        c.journal.put("protocol_revision", operation)
        atomic_json(c.workdir / "validation_plan.json", revised)
        return operation
    except (ValueError, OSError, TimeoutError) as error:
        # Once committed, publication recovery is mandatory; never roll back IDs.
        if operation["status"] == "committed":
            raise
        operation.update(status="aborted", reason=str(error), ended_at=time.time())
        c.journal.put("protocol_revision", operation)
        c.journal.event("protocol_revision_aborted", {"operation_id": operation["operation_id"], "reason": str(error)})
        c.plan = old
        return operation
