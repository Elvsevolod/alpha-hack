"""One fixed baseline and aggregate error diagnostics inside fit only."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from ..contracts import PipelineSpec, TaskSpec
from ..evaluation import score_predictions
from ..pipelines import build_pipeline
from ..util import digest
from ..validation import inner_split
from .temporal import period_errors, paired_comparison


def classification_errors(actual, predicted, classes):
    """Report each metric using its own denominator; absence of predictions is evidence."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=classes, average=None, zero_division=0)
    result = []
    for i, label in enumerate(classes):
        actual_ok = support[i] >= 30
        predicted_rows = int((predicted == label).sum())
        predicted_ok = predicted_rows >= 30
        row = {"class": label,
            "status": "available" if actual_ok and predicted_ok else "partial" if actual_ok or predicted_ok else "suppressed",
            "metric_status": {"precision": "available" if predicted_ok else "insufficient_predicted_support",
                "recall": "available" if actual_ok else "insufficient_actual_support",
                "f1": "available" if actual_ok and predicted_ok else "insufficient_aggregate_support"}}
        if actual_ok:
            row.update(actual_rows=int(support[i]), recall=float(recall[i]), no_predictions=predicted_rows == 0)
        if predicted_ok:
            row.update(predicted_rows=predicted_rows, precision=float(precision[i]))
        if actual_ok and predicted_ok:
            row["f1"] = float(f1[i])
        else:
            row["reason"] = "minimum_aggregate_support_per_metric"
        result.append(row)
    return result


def regression_errors(X, actual, predicted, training_target):
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    squared = error ** 2
    total = float(squared.sum())
    result = {"rmse": float(np.sqrt(squared.mean())), "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()), "tails": [], "missingness": []}
    for quantile in (.9, .99):
        threshold = float(np.quantile(training_target, quantile))
        mask = np.asarray(actual) >= threshold
        n = int(mask.sum())
        result["tails"].append({"training_quantile": quantile, "threshold": threshold,
            "status": "available" if n >= 30 else "suppressed",
            **({"rows": n, "rmse": float(np.sqrt(squared[mask].mean())),
                "squared_error_fraction": float(squared[mask].sum() / total) if total else 0.,
                "bias":float(error[mask].mean()), "mae":float(np.abs(error[mask]).mean())} if n >= 30 else {})})
    for column in X:
        mask = X[column].isna().to_numpy()
        if not mask.any():
            continue
        groups = []
        for name, selected in (("missing", mask), ("present", ~mask)):
            n = int(selected.sum())
            groups.append({"group": name, "status": "available" if n >= 30 else "suppressed",
                **({"rows": n, "rmse": float(np.sqrt(squared[selected].mean())),
                    "mae": float(np.abs(error[selected]).mean()), "bias": float(error[selected].mean())} if n >= 30 else {})})
        result["missingness"].append({"column": column, "groups": groups,
            "limitation": "Groups differ in composition; this is not a causal imputation comparison."})
    return result


def run(request, X, y):
    task = TaskSpec.model_validate(request["task"])
    if task.kind not in {"regression", "classification"} or len(task.targets) != 1:
        raise ValueError("Error diagnostic supports single-target regression/classification only")
    context = request["context"]
    evaluation = context.get("evaluation_contract", {})
    a, b = inner_split(X, y, task, context["strategy"], request["seed"], temporal_window=evaluation.get("temporal_window"), gap_periods=evaluation.get("gap_periods",0))
    spec = PipelineSpec(model="lightgbm", representation="native", seed=request["seed"],
        hyperparameters={"n_estimators": 300, "learning_rate": .05, "num_leaves": 31})
    if request.get("pipeline"):
        spec = PipelineSpec.model_validate(request["pipeline"])
    from .policy import validate_pipeline
    validate_pipeline(spec)
    pipeline = build_pipeline(spec, task, context)
    pipeline.fit(X.iloc[a].copy(), y.iloc[a].copy(), context)
    prediction = pipeline.predict(X.iloc[b].copy(), {**context, "task": task.model_dump(mode="json"),
        "split_id": "fit_inner_holdout"})
    target = task.targets[0]
    values = np.asarray(prediction.values[target])
    report = {"scope": "fit_inner_holdout", "fit_rows": len(a), "evaluation_rows": len(b),
        "fit_keys_hash": digest(list(X.iloc[a].index)), "evaluation_keys_hash": digest(list(X.iloc[b].index)),
        "pipeline": spec.model_dump(mode="json"), "score": score_predictions(task, y.iloc[b],
            {target: values}, prediction.prediction_type), "metric": task.metric.name,
        "model_fits": 1, "api_calls": 0, "outer_validation_used": False, "control_used": False,
        "limitation": "One inner split; comparative effects require a matched experiment. No organizer score or external-test validation."}
    report["evaluation_contract"] = evaluation
    report["evaluation_protocol_hash"] = digest({"evaluation":evaluation,"fit":report["fit_keys_hash"],"eval":report["evaluation_keys_hash"]})
    report["support_status"] = "limited" if len(b)<30 else "available"
    report["support_limitations"] = (["Inner holdout has fewer than 30 rows; effect estimates are unstable."] if len(b)<30 else [])
    if context["strategy"] in {"temporal","temporal_group"}:
        report["evaluation_periods"] = int(X.iloc[b][task.time_column].nunique())
        report["support_status"] = "limited" if len(b)<30 or report["evaluation_periods"]<20 or not evaluation.get("horizon_covered") else "available"
        report["support_limitations"].extend(evaluation.get("limitations", ["Temporal holdout length limits transfer conclusions; compare the external transfer horizon."]))
        if not evaluation.get("horizon_covered"):
            report["support_limitations"].append("Inner window does not cover the external observed-period horizon.")
    report["inner_feature_transfer"] = []
    for column in task.categorical_columns:
        unseen = ~X.iloc[b][column].isin(set(X.iloc[a][column]))
        if unseen.any():
            report["inner_feature_transfer"].append({"column":column, "evaluation_rows":len(b),
                "unseen_rows":int(unseen.sum()), "unseen_fraction":float(unseen.mean()),
                "limitation":"Inner training-category coverage differs; this does not locate the cause of errors."})
    if task.classes:
        classes = task.classes[target]
        predicted = np.asarray(prediction.classes[target])[values.argmax(axis=1)] if prediction.prediction_type == "probabilities" else values
        report["class_diagnostics_version"] = 2
        report["classes"] = classification_errors(y.iloc[b][target], predicted, classes)
    else:
        report["errors"] = regression_errors(X.iloc[b], y.iloc[b][target], values.reshape(-1), y.iloc[a][target])
        if context["strategy"] in {"temporal","temporal_group"}:
            report["period_errors"] = period_errors(X.iloc[b][task.time_column], y.iloc[b][target], values)
            if request.get("baseline_temporal_report"):
                report["temporal_comparison"] = paired_comparison(request["baseline_temporal_report"], report)
    return {"status": "completed", "report": report}
