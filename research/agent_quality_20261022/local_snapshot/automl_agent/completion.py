"""One dependency graph for estimates, preparation and artifact protection."""
from pathlib import Path

from .util import digest, file_hash, read_json


class CompletionCosts(list):
    """Keep the existing cost-list interface with one complete phase forecast."""
    def __init__(self, values, forecast):
        super().__init__(values)
        self.forecast = forecast


def dependency_order(controller, identifiers):
    ordered, visiting = [], set()
    def visit(identifier):
        if identifier in visiting:
            raise ValueError("Cyclic experiment dependency graph")
        if identifier in ordered:
            return
        spec, result = controller.lookup(identifier)
        if not result or result.status != "completed":
            raise ValueError("Dependency has no completed result: " + identifier)
        visiting.add(identifier)
        for member in spec.pipeline.members:
            visit(member)
        visiting.remove(identifier)
        ordered.append(identifier)
    for identifier in identifiers:
        visit(identifier)
    return ordered


def estimate(controller, identifiers, rows=None, phase=None):
    c = controller
    rows = c.plan.fit if rows is None else rows
    row_hash = digest([c.dataset.row_keys[i] for i in rows])
    policies = c.journal.get("finalization_plan", {}).get("policies", {})
    costs, disk, nodes = {}, 0, []
    for identifier in dependency_order(c, identifiers):
        spec, result = c.lookup(identifier)
        pipeline = policies.get(identifier, {}).get("frozen_pipeline", spec.pipeline.model_dump(mode="json"))
        pipeline_hash = digest(pipeline)
        directory = Path(result.artifacts["directory"])
        size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        training_rows = max(1, result.training_summary.get("rows", len(c.plan.fit)))
        base = max(.2, result.seconds) if spec.pipeline.engine != "ensemble" else max(.1, result.seconds)
        budget = result.training_summary.get("budget") or {}
        fitted = budget.get("selected_fit_actual_seconds")
        if fitted is None and budget.get("targets") and all("selected_fit_actual_seconds" in target for target in budget["targets"].values()):
            fitted = sum(target["selected_fit_actual_seconds"] for target in budget["targets"].values())
        if fitted is not None:
            training_rows = max(1, budget.get("selected_fit_rows", training_rows))
            overhead = max(.2, result.seconds - result.training_summary.get("seconds", 0)
                - result.training_summary.get("automatic_pilot_seconds", 0))
            base = max(.2, fitted + overhead)
        cost = base * max(1, len(rows) / training_rows) * 1.25 + .2
        # Actual earlier phases refine the initial row-scaled upper estimate.
        for observation in c.journal.get("completion_costs", []):
            if observation["experiment_id"] == identifier and observation["pipeline_hash"] == pipeline_hash:
                cost = max(cost, observation["actual_seconds"] * len(rows) / max(1, observation["rows"]) * 1.25)
        ready = False
        if phase:
            path = c.workdir / "finalization" / phase / identifier / "worker_result.json"
            if path.is_file():
                raw = read_json(path)
                ready = raw.get("status") == "completed" and raw.get("fit_row_keys_hash") == row_hash and digest(raw.get("frozen_pipeline")) == pipeline_hash
                if ready:
                    ready = all((path.parent / relative).is_file() and file_hash(path.parent / relative) == sha for relative, sha in raw.get("hashes", {}).items())
        costs[identifier] = 0 if ready else cost
        nodes.append({"experiment_id": identifier, "seconds": costs[identifier], "rows": len(rows),
            "row_hash": row_hash, "pipeline_hash": pipeline_hash, "ready": ready, "disk_bytes": size,
            "initial_uncertainty_multiplier": 1.25})
        if not ready:
            disk += int(size * max(1, len(rows) / training_rows))
    return {"seconds": sum(costs.values()), "node_seconds": costs, "disk_bytes": disk,
        "nodes": nodes, "phase": phase, "rows_hash": row_hash}


def allocation(controller, identifiers, selected, rows, phase, available):
    graph = estimate(controller, identifiers, rows, phase)
    own = estimate(controller, [selected], rows, phase)["seconds"]
    return min(available, available * own / max(.001, graph["seconds"]))
