"""Bounded fit-only diagnostics; published PNGs contain only published aggregates."""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import PipelineSpec, PredictionBundle, TaskSpec
from .util import atomic_json, canonical, digest, file_hash, read_json


VERSION = 1
MIN_SUPPORT = 30
MAX_IMAGE_BYTES = 65536


def number(value):
    return float(value) if np.isfinite(value) else None


def distribution_report(frame, categorical_columns=()):
    columns = {}
    for name in frame:
        series = frame[name]
        item = {"rows": len(series), "missing": int(series.isna().sum())}
        if name not in categorical_columns and pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=float, na_value=np.nan)
            valid = values[np.isfinite(values)]
            item.update(type="numeric", finite_rows=len(valid), nonfinite=int((~np.isfinite(values) & ~pd.isna(values)).sum()))
            if len(valid):
                with np.errstate(over="ignore", invalid="ignore"):
                    quantiles = np.quantile(valid, [.01, .05, .25, .5, .75, .95, .99])
                    iqr = quantiles[4] - quantiles[2]
                    counts, edges = np.histogram(valid, bins=30)
                    item.update(mean=number(valid.mean()), std=number(valid.std(ddof=1)) if len(valid) > 1 else None,
                        min=number(valid.min()), max=number(valid.max()),
                        quantiles={str(p): number(v) for p, v in zip((1, 5, 25, 50, 75, 95, 99), quantiles)},
                        skew=number(pd.Series(valid).skew()) if len(valid) >= 3 and valid.min() != valid.max() else None,
                        iqr_outlier_fraction=float(((valid < quantiles[2] - 1.5 * iqr) | (valid > quantiles[4] + 1.5 * iqr)).mean()),
                        histogram={"counts": counts.tolist(), "edges": edges.tolist()})
            else:
                item["reason"] = "no_finite_values"
        else:
            counts = series.dropna().astype(str).value_counts()
            item.update(type="categorical", unique=len(counts),
                categories=[{"label": label, "count": int(count)} for label, count in counts.iloc[:15].items()],
                other=int(counts.iloc[15:].sum()))
        columns[name] = item
    return {"kind": "distributions", "rows": len(frame), "columns": columns, "sampled": False}


def confusion_report(actual, predicted, labels):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    matrix = confusion_matrix(actual, predicted, labels=labels)
    support, predicted_support = matrix.sum(axis=1), matrix.sum(axis=0)
    if (support < MIN_SUPPORT).any() or (predicted_support < MIN_SUPPORT).any():
        return {"kind": "confusion_matrix", "status": "suppressed", "reason": "minimum_class_aggregate_support"}
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, labels=labels, zero_division=0)
    pairs = sorted(((int(matrix[i, j]), i, j) for i in range(len(labels)) for j in range(len(labels)) if i != j), reverse=True)
    result = {"kind": "confusion_matrix", "rows": int(support.sum()),
        "classes": [{"label": label, "support": int(support[i]), "predicted_support": int(predicted_support[i]),
                     "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])} for i, label in enumerate(labels)],
        "macro_f1": float(np.mean(f1)), "top_confusions": [
            {"actual": labels[i], "predicted": labels[j], "count": count} for count, i, j in pairs[:5] if count]}
    if len(labels) <= 20:
        result.update(matrix=matrix.tolist(), normalized=(matrix / support[:, None]).tolist(), labels=labels)
    else:
        result["image_unavailable"] = "more_than_20_classes"
    return result


def residual_report(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Residual inputs must be aligned and finite")
    if len(actual) < MIN_SUPPORT:
        return {"kind": "residuals", "status": "suppressed", "reason": "minimum_aggregate_support"}
    residual = actual - predicted
    absolute, square = np.abs(residual), residual ** 2
    bins = min(10, len(actual) // MIN_SUPPORT)
    order = np.argsort(predicted, kind="stable")
    groups = np.array_split(order, bins)
    intervals = [{"rows": len(rows), "prediction_mean": float(predicted[rows].mean()),
        "residual_mean": float(residual[rows].mean()), "residual_median": float(np.median(residual[rows])),
        "residual_q10": float(np.quantile(residual[rows], .1)), "residual_q90": float(np.quantile(residual[rows], .9))} for rows in groups]
    counts, edges = np.histogram(residual, bins=30)
    tail = np.argsort(absolute)[-max(1, int(np.ceil(len(actual) * .1))):]
    return {"kind": "residuals", "rows": len(actual), "definition": "y_true_minus_y_pred_original_scale",
        "rmse": float(np.sqrt(square.mean())), "mae": float(absolute.mean()), "mean": float(residual.mean()),
        "median": float(np.median(residual)), "absolute_error_quantiles": {
            str(p): float(np.quantile(absolute, p / 100)) for p in (50, 90, 95, 99)},
        "top_10_percent_squared_error_fraction": float(square[tail].sum() / square.sum()) if square.sum() else 0.,
        "histogram": {"counts": counts.tolist(), "edges": edges.tolist()}, "prediction_intervals": intervals}


def render_report(report, path):
    if report.get("status") == "suppressed" or report.get("image_unavailable"):
        return report.get("reason") or report["image_unavailable"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kind = report["kind"]
    if kind == "distributions":
        fig, axes = plt.subplots(1, len(report["columns"]), figsize=(4 * len(report["columns"]), 3.8), squeeze=False)
        for ax, (name, item) in zip(axes[0], report["columns"].items()):
            if item["type"] == "numeric" and item.get("histogram"):
                hist = item["histogram"]
                ax.stairs(hist["counts"], hist["edges"], fill=True, color="#4477aa")
                ax.set_xlabel("Value")
            elif item["type"] == "categorical":
                rows = item["categories"] + ([{"label": "(other)", "count": item["other"]}] if item["other"] else [])
                ax.barh([str(row["label"])[:36] for row in rows], [row["count"] for row in rows], color="#4477aa")
            else:
                ax.text(.5, .5, "No finite observations", ha="center", transform=ax.transAxes)
            ax.set_title(str(name)[:60], fontsize=10)
            ax.set_ylabel("Count" if item["type"] == "numeric" else "Category")
    elif kind == "confusion_matrix":
        import seaborn as sns
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, values, title, fmt in zip(axes, (report["matrix"], report["normalized"]), ("Counts", "Normalized by actual class"), ("d", ".2f")):
            sns.heatmap(np.asarray(values), annot=True, fmt=fmt, cmap="Blues", cbar=False,
                xticklabels=[str(x)[:20] for x in report["labels"]], yticklabels=[str(x)[:20] for x in report["labels"]],
                annot_kws={"fontsize": 8}, ax=ax)
            ax.set(title=title, xlabel="Predicted", ylabel="Actual")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        hist = report["histogram"]
        axes[0].stairs(hist["counts"], hist["edges"], fill=True, color="#4477aa")
        axes[0].set(xlabel="Actual minus prediction", ylabel="Count", title="Residual distribution")
        rows = report["prediction_intervals"]
        x = [row["prediction_mean"] for row in rows]
        axes[1].plot(x, [row["residual_mean"] for row in rows], marker="o", label="Mean")
        axes[1].fill_between(x, [row["residual_q10"] for row in rows], [row["residual_q90"] for row in rows], alpha=.2, label="10–90%")
        axes[1].axhline(0, color="gray", linewidth=.8)
        axes[1].set(xlabel="Prediction, aggregated intervals", ylabel="Residual", title="Residuals by prediction")
        axes[1].legend()
    try:
        fig.suptitle(f"{report.get('evaluation_scope', report['scope'])} | n={report.get('rows', 'suppressed')} | aggregated diagnostics", fontsize=10)
        fig.tight_layout()
        content = io.BytesIO()
        fig.savefig(content, format="png", dpi=100)
        if len(content.getvalue()) > MAX_IMAGE_BYTES:
            return "png_exceeds_64_kib"
        path.write_bytes(content.getvalue())
    finally:
        plt.close(fig)
    return None


def run_worker(request, directory):
    from .worker import read_frame
    frame = read_frame(directory / "fit_X.json")
    if request["kind"] == "distributions":
        report = distribution_report(frame, request.get("categorical_columns", []))
        report["scope"] = request["scope"]
    else:
        from .evaluation import align_predictions
        from .pipelines import build_pipeline
        from .validation import inner_split
        task = TaskSpec.model_validate(request["task"])
        pipeline_spec = PipelineSpec.model_validate(request["pipeline"])
        target = task.targets[0]
        y = read_frame(directory / "fit_y.json")
        context = request["context"]
        context["task"] = task.model_dump(mode="json")
        a, b = inner_split(frame, y, task, context["strategy"], pipeline_spec.seed)
        pipeline = build_pipeline(pipeline_spec, task, context)
        pipeline.fit(frame.iloc[a].copy(), y.iloc[a].copy(), {**context, "fit_row_keys": list(frame.iloc[a].index)})
        bundle = pipeline.predict(frame.iloc[b].copy(), {**context, "split_id": "diagnostic"})
        bundle = PredictionBundle.model_validate(bundle.model_dump(mode="json") if hasattr(bundle, "model_dump") else bundle)
        predictions = align_predictions(bundle, list(frame.iloc[b].index), task, context["protocol_hash"], "diagnostic")
        values = np.asarray(predictions[target])
        if request["kind"] == "confusion_matrix":
            labels = task.classes[target]
            predicted = np.asarray(labels)[np.argmax(values, axis=1)] if bundle.prediction_type == "probabilities" else values
            report = confusion_report(y.iloc[b][target], predicted, labels)
        else:
            report = residual_report(y.iloc[b][target].to_numpy(dtype=float), values)
        report.update(scope="fit", evaluation_scope="fit_inner_holdout", target=target,
            fit_rows=len(a), evaluation_rows=len(b), fit_keys_hash=digest(list(frame.iloc[a].index)),
            evaluation_keys_hash=digest(list(frame.iloc[b].index)), exploratory=True,
            recipe_hash=pipeline_spec.hash, source_recipe_hash=request["source_recipe_hash"])
    report.update(diagnostic_version=VERSION, data_version=request["data_version"], protocol_hash=request["protocol_hash"],
                  experiment_id=request.get("experiment_id"))
    # Fail before publication for non-finite or otherwise invalid JSON.
    canonical(report)
    report_path = directory / "diagnostic_report.json"
    atomic_json(report_path, report)
    started = time.monotonic()
    try:
        unavailable = render_report(report, directory / "diagnostic.png")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        unavailable = "render_failed:" + type(error).__name__
    hashes = {report_path.name: file_hash(report_path)}
    if not unavailable:
        hashes["diagnostic.png"] = file_hash(directory / "diagnostic.png")
    return {"status": "completed", "hashes": hashes, "image_unavailable": unavailable,
            "render_seconds": time.monotonic() - started}


class DiagnosticTools:
    def __init__(self, controller):
        self.c = controller

    def validate(self, args):
        c = self.c
        if args.kind == "distributions":
            allowed = set(c.task.feature_columns + (c.task.targets if args.include_target and args.scope == "fit" else []))
            columns = args.columns or c.task.feature_columns[:3]
            if not columns or not set(columns) <= allowed:
                raise ValueError("Diagnostic column is unavailable in this scope")
            if args.scope != "fit" and args.include_target:
                raise ValueError("Diagnostic targets require fit scope")
            return columns, None, None
        if args.scope != "fit" or args.include_target or args.columns:
            raise ValueError("Model diagnostics require fit scope, experiment_id, and no column overrides")
        spec, result = c.lookup(args.experiment_id)
        expected = "classification" if args.kind == "confusion_matrix" else "regression"
        if c.task.kind != expected or len(c.task.targets) != 1:
            raise ValueError("This diagnostic requires single-target " + expected)
        if not result or result.status != "completed" or result.scope == "pilot" or spec.validation_hash != c.plan.hash:
            raise ValueError("Diagnostic requires a completed experiment on the active protocol")
        if spec.pipeline.engine not in {"direct", "optuna"}:
            raise ValueError("Model diagnostics currently support direct and selected Optuna configurations")
        pipeline = spec.pipeline
        if pipeline.engine == "optuna":
            path = Path(result.artifacts["worker_manifest"])
            if not path.resolve().is_relative_to(c.workdir.resolve()) or file_hash(path) != result.artifacts["worker_manifest_sha256"]:
                raise ValueError("Selected pipeline manifest is unavailable or changed")
            pipeline = PipelineSpec.model_validate(read_json(path)["frozen_pipeline"])
        pipeline = pipeline.model_copy(update={"engine": "direct", "search_owner": "none", "n_trials": None,
            "search_space": {}, "fit_diagnostics": "none", "class_offsets": None,
            "prediction_policy": spec.pipeline.prediction_policy, "postprocess_fit_policy": spec.pipeline.postprocess_fit_policy})
        return [], spec, pipeline

    def run(self, args):
        from context_compressor import Source
        from .executor import stage_frame
        c = self.c
        columns, spec, pipeline = self.validate(args)
        identity = {"version": VERSION, "data": c.dataset.source_hash, "protocol": c.plan.hash,
            "kind": args.kind, "scope": args.scope, "columns": columns,
            "recipe": pipeline.hash if pipeline else None, "source_recipe": spec.pipeline.hash if spec else None,
            "experiment_id": args.experiment_id, "budget_seconds": args.budget_seconds}
        key = digest(identity)
        directory = c.workdir / "diagnostics" / key
        complete_path = directory / "published.json"
        failure_path = directory / "failed.json"
        if failure_path.is_file():
            return {"status": "technical_failure", "scope": args.scope, "error_code": read_json(failure_path)["status"],
                    "cached": True, "artifacts": []}
        cached = complete_path.is_file()
        if cached:
            measured = read_json(complete_path)
            for name, sha in measured["hashes"].items():
                if file_hash(directory / name) != sha:
                    raise ValueError("Cached diagnostic artifact changed")
        else:
            reserve = c.budget.final_reserve(c.finalist_costs())
            available = c.budget.remaining - reserve
            estimate = max(5., (c.lookup(args.experiment_id)[1].seconds if spec else 0) * 1.5 + 5.)
            if available < min(args.budget_seconds, estimate) + 2 or args.budget_seconds < estimate:
                return {"status": "budget_too_small", "scope": args.scope, "error_code": "diagnostic_fit_or_render_reserve",
                        "estimated_seconds": estimate, "artifacts": []}
            # A started but unconfirmed worker is never blindly trained again.
            if (directory / "request.json").exists():
                return {"status": "outcome_unknown", "scope": args.scope, "error_code": "diagnostic_attempt_unconfirmed", "artifacts": []}
            directory.mkdir(parents=True, exist_ok=True)
            if spec:
                request = c.stage(directory, spec.model_copy(update={"budget_seconds": args.budget_seconds}), c.plan.fit, {}, pipeline,
                                  mode="visual_diagnostics")
            else:
                frame = c.dataset.test if args.scope == "test_features" else c.dataset.train.iloc[c.plan.fit if args.scope == "fit" else c.plan.validation]
                stage_frame(directory / "fit_X.json", frame[columns].copy())
                request = {"mode": "visual_diagnostics", "context": {"budget_seconds": args.budget_seconds}}
            request.update(kind=args.kind, scope=args.scope, data_version=c.dataset.source_hash, protocol_hash=c.plan.hash,
                experiment_id=args.experiment_id, source_recipe_hash=spec.pipeline.hash if spec else None,
                categorical_columns=c.task.categorical_columns + (c.task.targets if c.task.classes else []))
            duration = min(args.budget_seconds, c.budget.remaining - reserve)
            if duration < 5:
                return {"status": "budget_too_small", "scope": args.scope, "error_code": "diagnostic_staging_reached_reserve", "artifacts": []}
            measured = c.executor.run(directory, request, duration)
            c.journal.event("visual_diagnostic_execution", {"key": key, "seconds": measured.get("wall_seconds", measured.get("seconds")),
                "kind": args.kind, "status": measured["status"], "additional_fit": bool(spec)})
            if measured["status"] != "completed":
                atomic_json(failure_path, {"status": measured["status"]})
                return {"status": "technical_failure", "scope": args.scope, "error_code": measured["status"], "artifacts": []}
            atomic_json(complete_path, measured)
        report = read_json(directory / "diagnostic_report.json")
        source = Source("diagnostic:" + key, digest(report), canonical(report), args.scope, "external_data", origin="get_diagnostic_report")
        c.context_memory.store.put_source(source)
        summary = {name: value for name, value in report.items() if name not in {"matrix", "normalized", "labels", "histogram", "prediction_intervals", "classes"}}
        if "columns" in summary:
            summary["columns"] = {name: {k: v for k, v in item.items() if k not in {"histogram", "categories"}} for name, item in summary["columns"].items()}
        if "classes" in report:
            summary["weakest_classes"] = sorted(report["classes"], key=lambda row: row["f1"])[:3]
        if len(canonical(summary).encode()) > 3500:
            summary = {"kind": args.kind, "scope": args.scope, "details_in_versioned_source": True}
        artifacts = []
        if "diagnostic.png" in measured["hashes"]:
            import struct
            path = directory / "diagnostic.png"
            width, height = struct.unpack(">II", path.read_bytes()[16:24])
            artifacts = [{"artifact_id": key, "path": str(path.relative_to(c.workdir)), "mime_type": "image/png",
                "sha256": measured["hashes"]["diagnostic.png"], "bytes": path.stat().st_size, "width": width, "height": height,
                "scope": args.scope, "source_id": source.source_id, "source_version": source.version,
                "data_version": report["data_version"], "protocol_hash": report["protocol_hash"]}]
        return {"status": "success", "scope": args.scope, "diagnostic_version": VERSION, "summary": summary,
                "report_status": report.get("status", "available"), "read": c.context_memory.store.route(source.reference), "cached": cached,
                "artifacts": artifacts, "image_unavailable": measured.get("image_unavailable")}

    def image(self, operation_id, result):
        artifacts = result.get("artifacts", [])
        if not artifacts:
            return None
        artifact = dict(artifacts[0])
        path = (self.c.workdir / artifact.pop("path")).resolve()
        if not path.is_relative_to((self.c.workdir / "diagnostics").resolve()) or not path.is_file() or file_hash(path) != artifact["sha256"]:
            raise ValueError("Diagnostic image is unavailable or changed")
        artifact.update(operation_id=operation_id, data_url="data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii"))
        return artifact
