"""Durable protocol replacement; the active protocol changes only at commit."""
from pathlib import Path
import time

from .contracts import ExperimentSpec, PredictionBundle, ValidationPlan
from .evaluation import choose_leader
from .publication import Publisher
from .util import atomic_json, file_hash, read_json


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
            directory = c.workdir / "attempts" / spec.experiment_id
            path = directory / "worker_result.json"
            if item["status"] == "completed":
                if not path.is_file() or file_hash(path) != item["manifest_sha256"]:
                    raise ValueError("Completed protocol attempt manifest checksum mismatch")
                c.verify_dependency_artifacts(spec.experiment_id)
                continue
            c.plan = revised
            try:
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
                elif item["status"] == "pending":
                    costs = c.finalist_costs()
                    if c.budget.remaining < spec.budget_seconds + c.budget.final_reserve(costs):
                        raise ValueError("Insufficient remaining time to complete protocol revision")
                    c.budget.claim_candidate()
                    c.journal.save_experiment(spec)
                    item["status"] = "running"
                    c.journal.put("protocol_revision", operation)
                    result = c.execute_experiment(spec, automatic_pilot=False)
                else:
                    raise ValueError("Protocol attempt cannot be resumed safely: " + item["status"])
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
