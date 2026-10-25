"""Explicit configuration inheritance; learned objects never cross fit boundaries."""
import copy

from .contracts import PipelineSpec, LEARNED_PIPELINE_FIELDS
from .compatibility import require_compatible
from .util import digest


PARENT_FIELDS = tuple(PipelineSpec().research_recipe())


def removed_components(before, after, frozen=False):
    """Fresh-fit choices that disappeared, including part of a transform chain."""
    removed = []
    defaults = PipelineSpec()
    for name in PARENT_FIELDS:
        if name in {"postprocess_fit_policy", "decision_method"} and after.prediction_policy == "default":
            continue
        if name == "log_inverse" and after.target_transform == "none":
            continue
        if frozen and name in {"engine", "search_space", "n_trials", "hpo_sampler", "continue_study"}:
            continue
        old, new = getattr(before, name), getattr(after, name)
        if name == "transformations":
            remaining = [item.model_dump(mode="json") for item in new]
            for item in old:
                value = item.model_dump(mode="json")
                if value not in remaining:
                    removed.append(name)
                    break
                remaining.remove(value)
        elif name in {"prediction_policy", "target_transform", "shift_strategy"} and old not in ("default", "none") and old != new:
            removed.append(name)
        elif old != getattr(defaults, name) and new == getattr(defaults, name):
            removed.append(name)
    return removed


def inherit_parent(c, args):
    """Resolve omissions before compatibility, cost, hashing or any fit."""
    if not args.parent_id:
        return
    parent, _ = c.lookup(args.parent_id)
    supplied = args.pipeline.model_fields_set
    changes = args.hypothesis.changed_fields
    updates = {name: copy.deepcopy(getattr(parent.pipeline, name)) for name in PARENT_FIELDS
        if name not in supplied and name not in changes}
    if ("hyperparameters" in supplied and args.pipeline.hyperparameters and "hyperparameters" not in changes
            and updates.get('model', args.pipeline.model) == parent.pipeline.model):
        if parent.pipeline.engine in {'direct', 'optuna'}:
            from .pipelines import effective_parameters
            override = args.pipeline.model_copy(update={'model':parent.pipeline.model})
            updates['hyperparameters'] = {**effective_parameters(parent.pipeline), **effective_parameters(override)}
        else:
            updates["hyperparameters"] = {**copy.deepcopy(parent.pipeline.hyperparameters), **args.pipeline.hyperparameters}
    if "imputation" in supplied and "imputation" not in changes:
        updates["imputation"] = {**parent.pipeline.imputation.model_dump(),
            **args.pipeline.imputation.model_dump(exclude_unset=True)}
    if args.pipeline.prediction_policy == "default" and "prediction_policy" in supplied:
        for name in ("postprocess_fit_policy", "decision_method"):
            updates.pop(name, None)
    if args.pipeline.target_transform == "none" and "target_transform" in supplied:
        updates.pop("log_inverse", None)
    values = args.pipeline.model_dump(mode="json")
    resolved = PipelineSpec.model_validate({**values, **updates})
    missing = [name for name in removed_components(parent.pipeline, resolved) if name not in changes]
    if missing:
        raise ValueError("Explicit parent component removal requires hypothesis.changed_fields: " + ", ".join(missing))
    args.pipeline = resolved
    args._inherited_fields = sorted(set(args._inherited_fields) | set(updates))


def patch(base, expected_hash, changes):
    before = base.research_recipe()
    if digest(before) != expected_hash:
        raise ValueError("Stale base recipe hash; read the current complete recipe")
    after = copy.deepcopy(before)
    paths = set()
    for change in changes:
        parts = change.path.split(".")
        if change.path in paths or any(change.path.startswith(old + ".") or old.startswith(change.path + ".") for old in paths):
            raise ValueError("Overlapping recipe changes are ambiguous")
        paths.add(change.path)
        if parts[0] not in before or parts[0] in LEARNED_PIPELINE_FIELDS:
            raise ValueError("Unknown or learned recipe field: " + change.path)
        parent = after
        for part in parts[:-1]:
            if part not in parent or not isinstance(parent[part], dict):
                raise ValueError("Unknown recipe path: " + change.path)
            parent = parent[part]
        key = parts[-1]
        if len(parts) > 1 and parts[0] != "hyperparameters" and key not in parent:
            raise ValueError("Unknown typed recipe path: " + change.path)
        if change.operation == "remove":
            if key not in parent:
                raise ValueError("Cannot remove an absent recipe field")
            if len(parts) == 1:
                parent[key] = copy.deepcopy(getattr(PipelineSpec(), key))
            else:
                del parent[key]
        else:
            parent[key] = change.value
    result = PipelineSpec.model_validate(after)
    if result.engine in {"direct", "optuna"}:
        from .pipelines import effective_parameters, CONTROLLED_PARAMETERS
        if set(result.hyperparameters) & CONTROLLED_PARAMETERS.get(result.model, set()):
            raise ValueError("Recipe changes cannot override controller-managed resources or seeds")
        result = result.model_copy(update={"hyperparameters": effective_parameters(result)})
        if result.engine == "optuna":
            from .search import normalized_search_space
            result = result.model_copy(update={"search_space": normalized_search_space(result)})
    normalized = result.research_recipe()
    difference = {key: {"before": before[key], "after": normalized[key]} for key in before if before[key] != normalized[key]}
    return result, {"pipeline": normalized, "recipe_hash": digest(normalized), "diff": difference,
        "learned_state": "fresh fit required; no offsets, weights or fitted transformers inherited"}


def resolve(c, application):
    if application.source == "experiment":
        saved, result = c.lookup(application.base_id)
        if result is None or result.status != "completed" or saved.data_version != c.dataset.source_hash:
            raise ValueError("Base recipe requires a completed experiment in this task")
        base = saved.pipeline
    else:
        entry = c.references.catalog_entry("recipes", application.base_id, c.task, c.plan)
        if entry.get("applicable") is False or "pipeline" not in entry:
            raise ValueError("Knowledge recipe is not executable for this task")
        base = PipelineSpec.model_validate(entry["pipeline"])
    if base.engine == "ensemble":
        raise ValueError("Ensemble inheritance requires build_ensemble with explicit verified member IDs")
    result, preview = patch(base, application.base_hash, application.changes)
    require_compatible(result, c.task, c.plan.strategy, c.capabilities)
    return result, {"status": "success", "base_id": application.base_id, **preview}
