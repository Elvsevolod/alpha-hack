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


def measured_cost(summary, seconds, rows, default_rows, ensemble=False, pipeline=None):
    """Forecast frozen selected fits separately from auxiliary work and fixed I/O."""
    from .cost_model import phase_forecast
    return phase_forecast(summary, seconds, rows, default_rows, ensemble, pipeline)


def candidate_measured_cost(summary, seconds, rows, default_rows, pipeline):
    """Estimate one candidate fit, including HPO that frozen refits skip."""
    frozen_fit = measured_cost(summary, seconds, rows, default_rows)
    budget = summary.get("budget") or {}
    auxiliary = max(0., float(budget.get("auxiliary_fit_actual_seconds", 0)) - float(budget.get("repeatable_auxiliary_fit_seconds", 0)))
    frozen_fit += auxiliary * max(1., rows / max(1, summary.get("rows", default_rows))) * 1.25
    targets = budget.get("targets") or {}
    search_seconds = sum(max(0.0, float(item.get("search_seconds", 0))) for item in targets.values())
    if not targets:
        search_seconds = max(0.0, float(budget.get("search_seconds", 0)))
    if not search_seconds:
        return frozen_fit

    trial_scale = 1.0
    if pipeline.engine == "autogluon":
        requested = pipeline.learners or ["GBM", "CAT", "XGB"]
        planned = min(pipeline.n_trials, len(requested)) if pipeline.n_trials is not None else len(requested)
        observed = max((len(item.get("configured_learners", [])) for item in targets.values()), default=0)
        if getattr(pipeline, 'autogluon_configurations', {}):
            planned = sum(len(pipeline.autogluon_configurations.get(name, [{}])) for name in requested[:planned])
            observed = max((item.get("configured_count", 0) for item in targets.values()), default=0)
        trial_scale = max(1.0, planned / max(1, observed))
    elif pipeline.n_trials is not None:
        observed = sum(int(item.get("attempted_trials", item.get("completed_trials", 0))) for item in targets.values())
        if not observed:
            observed = len((summary.get("inner_scores") or {}).get("trials", []))
        trial_scale = max(1.0, pipeline.n_trials / max(1, observed))

    search_rows = max(1, int(summary.get("rows", default_rows)))
    scaled_search = search_seconds * max(1, rows / search_rows) * trial_scale * 1.25
    return frozen_fit + scaled_search


def estimate(controller, identifiers, rows=None, phase=None):
    from .cost_model import iteration_multiplier
    c = controller
    rows = c.plan.fit if rows is None else rows
    row_hash = digest([c.dataset.row_keys[i] for i in rows])
    policies = c.journal.get("finalization_plan", {}).get("policies", {})
    costs, disk, nodes = {}, 0, []
    for identifier in dependency_order(c, identifiers):
        spec, result = c.lookup(identifier)
        pipeline = policies.get(identifier, {}).get("frozen_pipeline")
        if pipeline is None:
            manifest = result.artifacts.get('worker_manifest')
            pipeline = read_json(Path(manifest)).get('frozen_pipeline') if manifest and Path(manifest).is_file() else None
            pipeline = pipeline or spec.pipeline.model_dump(mode="json")
        pipeline_hash = digest(pipeline)
        directory = Path(result.artifacts["directory"])
        size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        training_rows = max(1, result.training_summary.get("rows", len(c.plan.fit)))
        training_rows = max(1, (result.training_summary.get("budget") or {}).get("selected_fit_rows", training_rows))
        cost = measured_cost(result.training_summary, result.seconds, len(rows), len(c.plan.fit), spec.pipeline.engine == "ensemble", pipeline)
        # Actual earlier phases refine the initial row-scaled upper estimate.
        for observation in c.journal.get("completion_costs", []):
            if observation["experiment_id"] == identifier and observation["pipeline_hash"] == pipeline_hash:
                cost = max(cost, observation["actual_seconds"] * len(rows) / max(1, observation["rows"])
                    * iteration_multiplier(pipeline, len(rows), observation['rows']) * 1.25)
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
            "iteration_multiplier": iteration_multiplier(pipeline, len(rows), training_rows),
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
        from .cost_model import family, complexity
        recipe = family(pipeline)
        for spec, result in c.records():
            if result and result.status == "completed" and result.scope != "pilot" and spec.validation_hash == c.plan.hash and family(spec.pipeline) == recipe:
                summary = dict(result.training_summary)
                ratio = complexity(pipeline) / complexity(spec.pipeline)
                from .cost_model import selected_seconds
                summary["budget"] = {**(summary.get("budget") or {}), "selected_fit_actual_seconds": selected_seconds(summary) * ratio}
                summary["cost_curve"] = None
                samples.append((summary, result.seconds))
    if not samples:
        return None
    base = c.finalist_costs()
    current = getattr(base, "forecast", None)
    if current is None:
        current = forecast(c, []).forecast
    frozen_cost = lambda rows: max(measured_cost(summary, seconds, rows, len(c.plan.fit), pipeline=pipeline) for summary, seconds in samples)
    candidate_fit = lambda rows: max(candidate_measured_cost(summary, seconds, rows, len(c.plan.fit), pipeline)
        for summary, seconds in samples)
    search_population = len(c.plan.fit) + len(c.plan.validation) if getattr(c.plan, "evaluation_mode", "holdout") == "rolling" else len(c.plan.fit)
    fit = candidate_fit(search_population)
    if getattr(c.plan, "evaluation_mode", "holdout") == "rolling":
        fit *= 1 + sum(len(part["fit"]) / max(1, search_population) for part in c.plan.development_folds)
    control = frozen_cost(len(c.plan.fit) + len(c.plan.validation))
    refit = frozen_cost(len(c.dataset.train))
    publication = max(2, (sum(base) + fit) * .1)
    proposed = {**current, "seconds": current["control"]["seconds"] + control
        + max(current["refit"]["seconds"], refit) + publication + current["safety_seconds"],
        "candidate_fit_seconds": fit, "candidate_control_seconds": control, "candidate_refit_seconds": refit,
        "control_scoring_and_publication_seconds": publication,
        "cost_basis": "row_and_search_scaled_pilot" if measurement is not None else "measured_related_family_and_complexity",
        "uncertainty": ("custom_runtime_not_bounded_by_sample" if pipeline.engine == "custom" else
            "uncapped_trial_search_is_time_limited" if pipeline.n_trials is None and pipeline.engine in {"flaml", "optuna"} else
            "row_and_trial_scaling_is_an_estimate")}
    return CompletionCosts([*base, fit], proposed)


def allocation(controller, identifiers, selected, rows, phase, available):
    graph = estimate(controller, identifiers, rows, phase)
    own = estimate(controller, [selected], rows, phase)["node_seconds"]
    # Shared dependencies belong to the selected subgraph, once. Give it slack
    # while retaining the estimates of every other unique pending node.
    reserve = sum(seconds for node, seconds in graph["node_seconds"].items() if node not in own)
    return max(0., available - reserve)
