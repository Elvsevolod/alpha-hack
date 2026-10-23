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


def measured_cost(summary, seconds, rows, default_rows, ensemble=False):
    """The same row-scaled estimate for completed fits and admission pilots."""
    training_rows = max(1, summary.get("rows", default_rows))
    base = max(.1 if ensemble else .2, seconds)
    budget = summary.get("budget") or {}
    fitted = budget.get("selected_fit_actual_seconds")
    if fitted is None and budget.get("targets") and all("selected_fit_actual_seconds" in target for target in budget["targets"].values()):
        fitted = sum(target["selected_fit_actual_seconds"] for target in budget["targets"].values())
    if fitted is not None:
        training_rows = max(1, budget.get("selected_fit_rows", training_rows))
        overhead = max(.2, seconds - summary.get("seconds", 0) - summary.get("automatic_pilot_seconds", 0))
        base = max(.2, fitted + overhead)
    phases = summary.get("execution_phases") or {}
    if all(key in phases for key in ("loaded_fit_rows", "load_and_construct_seconds",
            "fit_seconds", "save_load_seconds", "predict_and_verify_seconds")):
        fit = phases["fit_seconds"] if fitted is None else fitted
        variable = phases["save_load_seconds"] + phases["predict_and_verify_seconds"]
        loading = phases["load_and_construct_seconds"] + phases.get("controller_staging_seconds", 0)
        startup = max(.2, seconds - (phases["fit_seconds"] if fitted is None else summary.get("seconds", 0))
            - variable - loading - summary.get("automatic_pilot_seconds", 0))
        # Pilot loads/stages the complete fit before sampling. Keep that row
        # basis distinct; preserve row scaling of model save and prediction work.
        return ((max(.2, fit) + variable) * max(1, rows / training_rows)
            + loading * max(1, rows / max(1, phases["loaded_fit_rows"])) + startup) * 1.25 + .2
    return base * max(1, rows / training_rows) * 1.25 + .2


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
        training_rows = max(1, (result.training_summary.get("budget") or {}).get("selected_fit_rows", training_rows))
        cost = measured_cost(result.training_summary, result.seconds, len(rows), len(c.plan.fit), spec.pipeline.engine == "ensemble")
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


def forecast(controller, identifiers):
    c = controller
    fit = estimate(c, identifiers)
    control = estimate(c, identifiers, sorted(c.plan.fit + c.plan.validation), "control")
    refits = [estimate(c, [identifier], list(range(len(c.dataset.train))), "refit") for identifier in identifiers]
    refit = max(refits, key=lambda graph: graph["seconds"], default={"seconds": 0, "disk_bytes": 0})
    publication = max(2, fit["seconds"] * .1)
    safety = min(60, c.budget.limits.seconds * .03)
    return CompletionCosts(fit["node_seconds"].values(), {
        "seconds": control["seconds"] + refit["seconds"] + publication + safety,
        "control": control, "refit": refit, "control_scoring_and_publication_seconds": publication,
        "safety_seconds": safety, "disk_bytes": control["disk_bytes"] + max((g["disk_bytes"] for g in refits), default=0)})


def candidate_costs(controller, pipeline, measurement=None):
    """Reserve for a possible new winner; an unmeasured recipe stays unknown."""
    c = controller
    if pipeline.custom_component == "analysis":
        return None
    if pipeline.engine == "ensemble":
        base = c.finalist_costs()
        current = getattr(base, "forecast", None) or forecast(c, []).forecast
        ids = list(dict.fromkeys([*current["control"]["node_seconds"], *pipeline.members]))
        proposed = forecast(c, ids)
        cost = lambda rows: measured_cost({}, .1, rows, len(c.plan.fit), ensemble=True)
        refit = estimate(c, pipeline.members, list(range(len(c.dataset.train))), "refit")["seconds"] + cost(len(c.dataset.train))
        f = proposed.forecast
        f["seconds"] = f["control"]["seconds"] + cost(len(c.plan.fit) + len(c.plan.validation)) + max(current["refit"]["seconds"], refit) + f["control_scoring_and_publication_seconds"] + f["safety_seconds"]
        f.update(candidate_fit_seconds=cost(len(c.plan.fit)), candidate_refit_seconds=refit,
            cost_basis="measured_dependency_graph", uncertainty="blend_overhead_is_an_estimate")
        return proposed
    samples = []
    if measurement is not None:
        samples.append((measurement["training_summary"], measurement.get("wall_seconds", measurement.get("seconds", 0))))
    else:
        recipe = pipeline.model_dump(exclude={"seed", "n_trials"})
        for spec, result in c.records():
            if result and result.status == "completed" and result.scope != "pilot" and spec.validation_hash == c.plan.hash and spec.pipeline.model_dump(exclude={"seed", "n_trials"}) == recipe:
                samples.append((result.training_summary, result.seconds))
    if not samples:
        return None
    base = c.finalist_costs()
    current = getattr(base, "forecast", None)
    if current is None:
        current = forecast(c, []).forecast
    cost = lambda rows: max(measured_cost(summary, seconds, rows, len(c.plan.fit)) for summary, seconds in samples)
    fit = cost(len(c.plan.fit))
    control = cost(len(c.plan.fit) + len(c.plan.validation))
    refit = cost(len(c.dataset.train))
    publication = max(2, (sum(base) + fit) * .1)
    proposed = {**current, "seconds": current["control"]["seconds"] + control
        + max(current["refit"]["seconds"], refit) + publication + current["safety_seconds"],
        "candidate_fit_seconds": fit, "candidate_control_seconds": control, "candidate_refit_seconds": refit,
        "control_scoring_and_publication_seconds": publication,
        "cost_basis": "row_scaled_pilot" if measurement is not None else "measured_same_recipe",
        "uncertainty": "custom_runtime_not_bounded_by_sample" if pipeline.engine == "custom" else "row_scaling_is_an_estimate"}
    return CompletionCosts([*base, fit], proposed)


def allocation(controller, identifiers, selected, rows, phase, available):
    graph = estimate(controller, identifiers, rows, phase)
    own = estimate(controller, [selected], rows, phase)["node_seconds"]
    # Shared dependencies belong to the selected subgraph, once. Give it slack
    # while retaining the estimates of every other unique pending node.
    reserve = sum(seconds for node, seconds in graph["node_seconds"].items() if node not in own)
    return max(0., available - reserve)
