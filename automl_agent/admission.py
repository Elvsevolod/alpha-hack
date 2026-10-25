"""Durable, validated native work which can finish without another model call."""
import time

from .contracts import ToolCommand, TrainArgs
from .compatibility import require_compatible
from .util import digest


def admit(c, command, shared_deadline=None):
    if c.state != "searching" or c.journal.get("control_opened") or c.journal.get("control_status", "not_opened") != "not_opened":
        return False
    if command.tool != "train_model":
        return False
    args = TrainArgs.model_validate(command.arguments)
    if args.apply_recipe:
        from .recipe_patch import resolve
        args.pipeline, _ = resolve(c, args.apply_recipe)
    from .recipe_patch import inherit_parent
    inherit_parent(c, args)
    if args.pipeline.engine != "direct" or args.pipeline.custom_component != "pipeline":
        return False
    require_compatible(args.pipeline, c.task, c.plan.strategy, c.capabilities)
    c.validate_evidence(args.hypothesis.evidence_refs)
    c.make_spec(args, "queue_admission")
    from .completion import candidate_costs
    costs = candidate_costs(c, args.pipeline)
    measured = getattr(costs, "forecast", {}).get("candidate_fit_seconds")
    if measured is None and args.budget_seconds > 30 or measured is not None and measured >= 120:
        return False
    queue = c.journal.get("admitted_queue", [])
    data = {"command": command.model_dump(mode="json"), "task": c.task.hash, "data": c.dataset.source_hash,
        "protocol": c.plan.hash, "recipe": args.pipeline.hash, "critic_required": False,
        "expires_epoch": min(c.budget.deadline, shared_deadline) if shared_deadline is not None else c.budget.deadline}
    identity = digest(data)
    old = next((item for item in queue if item["command"]["operation_id"] == command.operation_id), None)
    if old:
        if old["identity"] != identity:
            raise ValueError("Admitted operation ID changed")
        return old["status"] == "ready"
    queue.append({**data, "identity": identity, "status": "ready", "admitted_at": time.time()})
    c.journal.put("admitted_queue", queue)
    return True


def drain(c, reason):
    c.journal.put("channel_state", {"status": "degraded", "reason": str(reason), "new_generation_allowed": False})
    queue = c.journal.get("admitted_queue", [])
    executed = []
    for item in queue:
        if c.state != "searching" or c.journal.get("control_opened") or c.journal.get("control_status", "not_opened") != "not_opened":
            break
        if item["status"] == "running":
            previous = c.journal.operation(item["command"]["operation_id"])
            item["status"] = "already_recorded" if previous and previous.status in {"succeeded", "failed"} else "outcome_unknown"
            c.journal.put("admitted_queue", queue)
        if item["status"] != "ready":
            continue
        body = {key: item[key] for key in ("command", "task", "data", "protocol", "recipe", "critic_required", "expires_epoch")}
        if digest(body) != item["identity"] or (item["task"], item["data"], item["protocol"]) != (c.task.hash, c.dataset.source_hash, c.plan.hash):
            item["status"] = "rejected_stale_admission"
            c.journal.put("admitted_queue", queue)
            continue
        if time.time() >= item["expires_epoch"]:
            item["status"] = "expired_shared_budget"
            c.journal.put("admitted_queue", queue)
            continue
        command = ToolCommand.model_validate(item["command"])
        previous = c.journal.operation(command.operation_id)
        if previous:
            item["status"] = "already_recorded" if previous.status in {"succeeded", "failed"} else "outcome_unknown"
            c.journal.put("admitted_queue", queue)
            continue
        args = TrainArgs.model_validate(command.arguments)
        if args.apply_recipe:
            from .recipe_patch import resolve
            args.pipeline, _ = resolve(c, args.apply_recipe)
        from .recipe_patch import inherit_parent
        inherit_parent(c, args)
        if args.pipeline.hash != item["recipe"]:
            item["status"] = "rejected_stale_recipe"
            c.journal.put("admitted_queue", queue)
            continue
        if args.budget_seconds > item["expires_epoch"] - time.time():
            item["status"] = "expired_shared_budget"
            c.journal.put("admitted_queue", queue)
            continue
        from .completion import candidate_costs
        costs = candidate_costs(c, args.pipeline)
        if c.budget.remaining <= c.budget.final_reserve(costs or c.finalist_costs()) + args.budget_seconds:
            break
        # Persist before execution: an interruption never creates a second fit.
        item["status"] = "running"
        c.journal.put("admitted_queue", queue)
        try:
            c.dispatch(command)
            item["status"] = "completed"
            executed.append(command.operation_id)
        except (ValueError, TimeoutError) as error:
            item.update(status="failed", reason=str(error))
        finally:
            c.journal.put("admitted_queue", queue)
    c.journal.event("degraded_queue_finished", {"executed": executed, "search_can_resume_after_control": False})
    return executed
