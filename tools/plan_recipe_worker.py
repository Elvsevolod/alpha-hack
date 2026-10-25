"""Execute one frozen offline recipe. No Qwen or evaluation labels are loaded."""
import json
from pathlib import Path
import sys
import time


def composition(spec, members, task, X, y, query, context):
    from automl_agent.api import prediction_bundle
    from automl_agent.contracts import PipelineSpec, TrainingSummary
    from automl_agent.decision import decide
    from automl_agent.learned_blend import combine, fit_blend, fit_final_layer
    from automl_agent.pipelines import build_pipeline
    from automl_agent.research_evaluation import oof_predictions
    started = time.monotonic()
    reports, predictions, summaries = [], [], []
    specs = [PipelineSpec.model_validate(value) for value in members]
    learned = spec.ensemble_strategy != "uniform" or spec.prediction_policy == "class_offsets"
    if learned:
        for member in specs:
            remaining = context["budget_seconds"] - (time.monotonic() - started)
            reports.append(oof_predictions(member, task, X, y, {**context, "budget_seconds": remaining * .5 / len(specs)}))
        if spec.ensemble_strategy != "uniform":
            weights, meta, _ = fit_blend(task, reports, y, spec.ensemble_strategy, spec.seed)
            spec = spec.model_copy(update={"ensemble_weights": weights, "ensemble_meta": meta})
        if spec.prediction_policy == "class_offsets":
            spec = fit_final_layer(task, reports, y, spec)
    for member in specs:
        local = {**context, "budget_seconds": (context["budget_seconds"] - (time.monotonic() - started)) / (len(specs) - len(summaries))}
        model = build_pipeline(member, task, local)
        summaries.append(model.fit(X, y, local))
        bundle = model.predict(query, local)
        predictions.append(bundle.calibrated_probabilities or bundle.raw_probabilities or bundle.values)
    values = combine(task, predictions, spec)
    decision = {target: decide(values[target], task.classes[target], spec.learned_decision.get(target)).tolist()
        for target in task.targets} if spec.learned_decision else None
    bundle = prediction_bundle(decision or values, query, context, "labels" if decision else "probabilities" if task.classes else "values",
        task.classes, raw_probabilities={key: value.tolist() for key, value in values.items()} if task.classes else {}, decision_values=decision or {})
    summary = TrainingSummary(rows=len(X), seconds=time.monotonic() - started,
        auxiliary_fits=sum(value.auxiliary_fits for value in summaries) + sum(len(report["folds"]) for report in reports),
        budget={"members": [value.model_dump(mode="json") for value in summaries], "strict_oof_members": len(reports),
            "all_meta_stages_fit_only": True})
    return bundle, summary, spec


def run(request_path):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text())
    sys.path.insert(0, request["runtime_root"])
    from automl_agent.contracts import PipelineSpec, TaskSpec
    from automl_agent.pipelines import build_pipeline
    from automl_agent.worker import read_frame
    from automl_agent.util import atomic_json
    from threadpoolctl import threadpool_limits
    unsupported = {key: value for key, value in request["task"].items() if key not in TaskSpec.model_fields}
    if any(value not in (None, [], {}, "", 0, False) for value in unsupported.values()):
        raise ValueError("Baseline cannot represent this nonempty task field")
    task = TaskSpec.model_validate({key: value for key, value in request["task"].items() if key in TaskSpec.model_fields})
    recipe = dict(request["recipe"])
    members = recipe.pop("benchmark_members", None)
    spec = PipelineSpec.model_validate(recipe)
    directory = request_path.parent
    X, y, query = [read_frame(directory / name) for name in ("fit_X.json", "fit_y.json", "query_X.json")]
    started = time.monotonic()
    ctx = {"cpu": request["resources"]["cpu"], "memory_mb": request["resources"]["memory_mb"],
        "budget_seconds": request["resources"]["seconds"], "strategy": request["strategy"], "protocol_hash": request["protocol_hash"],
        "workdir": str(directory), "task": task.model_dump(mode="json"), "split_id": "external_development",
        "fit_row_keys": X.index.tolist(), "data_hash": request["data_hash"], "library_versions": request["library_versions"]}
    if request.get("rolling_folds"):
        ctx["rolling_folds"] = request["rolling_folds"]
        ctx["split_id"] = "validation"
    if spec.shift_strategy != "none":
        ctx["unlabelled_features"] = query.copy()
    # A timeout remains scoreable and visible, including its early fallback.
    if task.classes:
        fallback = {target: [y[target].mode().iloc[0]] * len(query) for target in task.targets}
        kind = "labels"
    else:
        fallback = {target: [float(y[target].mean())] * len(query) for target in task.targets}
        kind = "values"
    atomic_json(directory / "prediction.json", {"values": fallback, "prediction_type": kind, "fallback": True,
        "row_keys": query.index.tolist()})
    with threadpool_limits(limits=ctx["cpu"]):
        fit_started = time.monotonic()
        if members:
            bundle, summary, frozen = composition(spec, members, task, X, y, query, ctx)
            fit_seconds, prediction_seconds = time.monotonic() - fit_started, None
        else:
            pipeline = build_pipeline(spec, task, ctx)
            summary = pipeline.fit(X, y, ctx)
            fit_seconds = time.monotonic() - fit_started
            predict_started = time.monotonic()
            bundle = pipeline.predict(query, ctx)
            prediction_seconds = time.monotonic() - predict_started
            frozen = pipeline.frozen_spec()
    atomic_json(directory / "prediction.json", {"values": bundle.values, "prediction_type": bundle.prediction_type,
        "row_keys": bundle.row_keys, "raw_probabilities": bundle.raw_probabilities, "fallback": False})
    atomic_json(directory / "result.json", {"status": "completed", "evidence_type": "offline_recipe",
        "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds, "seconds": time.monotonic() - started,
        "summary": summary.model_dump(mode="json"), "frozen_recipe": frozen.model_dump(mode="json")})


if __name__ == "__main__":
    run(sys.argv[1])
