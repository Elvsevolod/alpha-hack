"""Small executable, task-bound cards supplementing the immutable API corpus."""
from .contracts import PipelineSpec
from .compatibility import incompatibilities
from .util import digest


def cards(task, strategy=None):
    linear = "logistic" if task.classes else "ridge"
    templates = {
        "torch_tabular": (PipelineSpec(model="torch_tabular",hyperparameters={"encoding":"quantile","epochs":50}),
            "Experimental neural tabular MLP numeric embeddings categorical complementarity",
            "One CPU thread; nested epoch selection plus fresh replay; profile OOF/refit/full inference",
            "Single-target RMSE/Macro F1 only; no external weights; agent may decline; quality is task-dependent"),
        "linear": (PipelineSpec(model=linear), "Smooth regularized linear baseline; compare representations before local tuning", "Small CPU fit; sparse inference", "Reject unsupported target or text representation; preserve time/groups"),
        "knn": (PipelineSpec(model="knn"), "KNN neighbors distance scaling representation complementary errors", "Bounded donor matrix and prediction batches; profile full test inference", "Reject oversized donor matrix or unsupported output; useful quality is unestablished"),
        "boosting": (PipelineSpec(model="lightgbm", hyperparameters={"n_estimators": 150}), "CPU boosting baseline and categorical regularization", "Bounded native fit plus measured refit", "Reject incompatible objective, inactive bagging or resource overrides"),
        "catboost": (PipelineSpec(model="catboost", representation="native", hyperparameters={"iterations": 150, "depth": 6}), "Compare a distinct categorical boosting family on the same transfer", "Bounded pilot and measured completion graph", "Require the installed CPU adapter and preserve all boundaries"),
        "xgboost": (PipelineSpec(model="xgboost", hyperparameters={"n_estimators": 150, "max_depth": 4}), "Compare a distinct regularized boosting family", "Bounded pilot and measured completion graph", "Require the installed CPU adapter and preserve all boundaries"),
        "optuna": (PipelineSpec(engine="optuna", model="lightgbm", hyperparameters={"n_estimators": 150},
            search_space={"num_leaves": {"type": "int", "low": 15, "high": 63},
                "reg_lambda": {"type": "float", "low": .01, "high": 10., "log": True}}),
            "Test regularization and capacity with adaptive fit-only HPO", "Measured trials plus protected selected fit; bounded TPE startup", "Never infer exhaustion from one boosting baseline"),
        "autogluon": (PipelineSpec(engine="autogluon", model="library_search"), "Compare a bounded six-configuration classical portfolio", "Fresh engine search plus frozen selected refit", "Bagging/stacking remain separate explicitly costed hypotheses"),
    }
    if task.kind == "classification" and len(task.classes.get(task.targets[0], [])) >= 3 and task.output.prediction_type == "labels":
        templates["offsets"] = (PipelineSpec(model="logistic", prediction_policy="class_offsets", postprocess_fit_policy="pooled_oof", decision_method="regularized"),
            "Macro F1 rare classes final decision offsets after probabilities", "Three fresh fold fits plus selected model", "Reject probability output; evaluate on other rows and retain argmax on no benefit")
    result = {}
    for name, (pipeline, description, price, refusal) in templates.items():
        reasons = incompatibilities(pipeline, task, strategy)
        if task.kind not in {"regression", "classification", "multioutput_regression", "multioutput_classification", "multilabel"}:
            reasons.append("card_requires_supported_tabular_target")
        recipe = pipeline.research_recipe()
        identifier = "action:" + name
        result[identifier] = {"id": identifier, "title": description, "pipeline": recipe, "recipe_hash": digest(recipe),
            "task_kinds": ["regression", "classification"] if name=="torch_tabular" else ["regression", "classification", "multioutput_regression", "multioutput_classification", "multilabel"],
            "conditions": "Confirmed current task, fit-only preprocessing, shared development rows; no external data",
            "diagnostic": "Read transfer/support and fit profile before selecting a representation",
            "price": price, "validation_boundaries": "Every learned stage is inside the permitted fold fit; final quality uses other rows",
            "refusal": refusal, "applicable": not reasons, "exclusion_reasons": reasons,
            "apply": {"source": "knowledge", "base_id": identifier, "base_hash": digest(recipe), "changes": []}}
    return result


def version(task, strategy=None):
    return "runtime_actions_" + digest(cards(task, strategy))[:20]


def discover(c, args, catalog=False):
    from .knowledge_tools import cursor, position, continuation
    registry = cards(c.task, c.plan.strategy)
    current = version(c.task, c.plan.strategy)
    if args.filters.library or args.filters.version:
        return {"status": "incompatible", "error_code": "actions_use_runtime_recipe_versions", "entries": [], "total": 0}
    query = "" if catalog else args.query.casefold()
    terms = set(query.split())
    selected = [card for card in registry.values() if not terms or terms & set((card["id"] + " " + card["title"]).casefold().split())]
    selected.sort(key=lambda card: (-len(terms & set(card["title"].casefold().split())), card["id"]))
    binding = {"collection": "actions", "version": current, "query": query}
    offset = position(args.cursor, binding)
    values = []
    for card in selected[offset:offset + args.limit]:
        values.append({"id": card["id"], "title": card["title"], "applicable": card["applicable"],
            "exclusion_reasons": card["exclusion_reasons"], "answer_confirmed": False,
            "read": {"tool": "read_knowledge", "arguments": {"collection": "actions", "id": card["id"], "corpus_version": current}}})
    token = cursor(binding, offset + len(values)) if offset + len(values) < len(selected) else None
    return {"status": "success" if values else "no_matches", "entries": values, "total": len(selected),
        "corpus": current, "next_cursor": token, "continuation": continuation("list_knowledge" if catalog else "search_knowledge", args, token),
        "complete": token is None, "catalog_browse": catalog, "kind": "project_owned_executable_actions"}


def read(c, args):
    current = version(c.task, c.plan.strategy)
    if args.corpus_version != current:
        return {"status": "missing_version", "error_code": "action_version_changed"}
    card = cards(c.task, c.plan.strategy).get(args.id)
    if card is None:
        return {"status": "unknown_id"}
    from .util import canonical
    if len(canonical(card)) > args.max_chars - 900:
        return {"status": "budget_too_small", "error_code": "complete_action_card_exceeds_budget", "minimum_max_chars": min(12000, len(canonical(card)) + 900)}
    return {"status": "success", "entry": card, "complete": True, "kind": "project_owned_executable_actions", "corpus": current,
        "reference": {"kind": "knowledge", "collection": "actions", "id": args.id, "version": current}}
