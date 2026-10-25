from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
import traceback
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

from .contracts import CustomAnalysisReport, PipelineSpec, PredictionBundle, TaskSpec, TrainingSummary
from .custom import TransformerPipeline, load_module, validate_code
from .dependency_policy import install_dependency_guard
from .pipelines import build_pipeline
from .util import atomic_json, digest, file_hash, read_json, redact
from .validation import InsufficientPilotSupport, pilot_population


def apply_access_policy(directory: Path, extra_read=()):
    package = Path(__file__).resolve().parent
    allowed_read = [directory.resolve(), package, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
        Path("/System/Library"), Path("/Library/Apple"), Path("/usr/lib"), Path("/usr/share"), Path("/etc"),
        Path("/private/etc"), Path("/dev"), Path("/proc"), Path("/sys"), *[Path(p).resolve() for p in extra_read]]
    allowed_write = [directory.resolve(), Path("/dev/null"), Path("/dev/shm")]

    def under(path, roots):
        return any(path == root or path.is_relative_to(root) for root in roots)

    def audit(event, args):
        if event == "open":
            target = args[0]
            if not isinstance(target, (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(target)).resolve()
            mode, flags = args[1], args[2]
            writing = bool(isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing and not under(path, allowed_write):
                raise PermissionError("Candidate write outside its attempt directory")
            if not writing and not under(path, allowed_read):
                raise PermissionError("Candidate read outside the allowed data and package directories")
        elif event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.fork", "os.forkpty"}:
            raise PermissionError("Candidate process creation is disabled")
        elif event == "socket.connect":
            connection = args[0]
            if connection.family in (socket.AF_INET, socket.AF_INET6):
                raise PermissionError("Candidate network access is disabled")

    sys.addaudithook(audit)


def read_frame(path):
    # Trusted staging format, decoded without pickle or inferred identifiers.
    content = read_json(Path(path))
    frame = pd.DataFrame(content["data"], columns=content["columns"], index=content["index"])
    for column in content.get("numeric", []):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def policy_probe(request, directory):
    apply_access_policy(directory)
    probes = {}
    for name, path in request["probe_paths"].items():
        try:
            with open(path, "rb") as stream:
                stream.read(1)
            probes[name] = "readable"
        except PermissionError:
            probes[name] = "blocked_python_io"
        except OSError:
            probes[name] = "unavailable"
    for name, path in request["write_probe_paths"].items():
        try:
            fd = os.open(path, os.O_WRONLY)
            os.close(fd)
            probes[name] = "writable"
        except PermissionError:
            probes[name] = "blocked_python_io"
        except OSError:
            probes[name] = "unavailable"
    try:
        with socket.socket() as connection:
            connection.connect(("127.0.0.1", 9))
        probes["network"] = "connection_attempt_allowed"
    except PermissionError:
        probes["network"] = "blocked_python_socket"
    except OSError:
        probes["network"] = "connection_attempt_allowed"
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "--version"], timeout=2, capture_output=True)
        probes["installer_subprocess"] = "unexpectedly_allowed"
    except PermissionError:
        probes["installer_subprocess"] = "blocked_python_process"
    probes["credential_environment"] = "absent" if not any(k for k in os.environ if any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET"))) else "present"
    probes["cpu_affinity"] = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "unsupported_on_host"
    return {"status": "completed", "probes": probes}


def run_request(request, directory):
    phases = {}
    phase_started = time.monotonic()
    if request["mode"] == "probe":
        return policy_probe(request, directory)
    if request.get("pipeline", {}).get("engine") == "lightautoml":
        # Trusted pinned dependency initialization probes CPU information via a
        # subprocess. Do this before the fit/candidate process-creation guard;
        # Executor supplies no API credentials and already monitors this tree.
        from lightautoml.automl.presets.tabular_presets import TabularAutoML  # noqa: F401
        from lightautoml.tasks import Task  # noqa: F401
    extras = request.get("read_dependencies", [])
    apply_access_policy(directory, extras)
    if request["mode"] == "correlations":
        from .profiling import correlations
        report = correlations(read_frame(directory / "correlation_frame.json"), request["excluded_columns"])
        atomic_json(directory / "correlations.json", report)
        return {"status": "completed", "hashes": {"correlations.json": file_hash(directory / "correlations.json")}}
    if request["mode"] == "domain_diagnostic":
        from .research_evaluation import domain_diagnostic
        report = domain_diagnostic(read_frame(directory / "domain_train.json"), read_frame(directory / "domain_test.json"), request["cpu"], request["seed"])
        atomic_json(directory / "domain_report.json", report)
        return {"status": "completed", "hashes": {"domain_report.json": file_hash(directory / "domain_report.json")}}
    if request["mode"] == "visual_diagnostics":
        from .visual_diagnostics import run_worker
        request.setdefault("context", {})["workdir"] = str(directory)
        return run_worker(request, directory)
    if request["mode"] == "clustering":
        from .clustering import ClusteringContract, run
        report = run(directory, ClusteringContract.model_validate(request["contract"]))
        return {"status": "completed", "report": report,
            "outputs": {"submission": "submission.csv"}, "hashes": {"submission.csv": file_hash(directory / "submission.csv")}}
    task = TaskSpec.model_validate(request["task"])
    spec = PipelineSpec.model_validate(request["pipeline"])
    context = request["context"]
    context["workdir"] = str(directory)
    context["task"] = task.model_dump(mode="json")
    context["execution_mode"] = request["mode"]
    X, y = read_frame(directory / "fit_X.json"), read_frame(directory / "fit_y.json")
    phases["loaded_fit_rows"] = len(X)
    context["fit_row_keys"] = list(X.index)
    if spec.shift_strategy != "none":
        context["unlabelled_features"] = read_frame(directory / "unlabelled_X.json")
    if request["mode"] == "oof":
        from .research_evaluation import oof_predictions
        report = oof_predictions(spec, task, X, y, context)
        atomic_json(directory / "oof_predictions.json", report)
        return {"status": "completed", "hashes": {"oof_predictions.json": file_hash(directory / "oof_predictions.json")},
            "oof_coverage": report["coverage"], "oof_folds_hash": report["folds_hash"]}
    if request.get("pilot"):
        positions, context["pilot_support"] = pilot_population(X, y, task, context["strategy"], request.get("pilot_rows", 400), spec.seed, spec)
        X, y = X.iloc[positions], y.iloc[positions]
    if spec.engine == "custom":
        path = directory / "candidate.py"
        validate_code(path.read_text(encoding="utf-8"), spec.custom_component)
        if file_hash(path) != spec.custom_hash:
            raise ValueError("Custom code checksum mismatch")
        install_dependency_guard(path, directory)
        module = load_module(path)
        if spec.custom_component == "analysis":
            result = module.analyze(X.copy(), y.copy(), context)
            atomic_json(directory / "analysis_raw.json", result)
            report = CustomAnalysisReport.model_validate(result)
            permitted = set(X.columns) | set(y.columns)
            statistics = []
            for item in report.statistics:
                if not set(item.columns) <= permitted or item.support_rows > len(X):
                    raise ValueError("Custom analysis has invalid variable roles or support")
                if item.support_rows < 30:
                    statistics.append({"name": item.name, "columns": item.columns, "status": "suppressed", "reason": "minimum_aggregate_support"})
                else:
                    statistics.append({**item.model_dump(mode="json"), "status": "available"})
            safe = {"statistics": statistics, "scope": "fit", "row_data_not_returned": True}
            atomic_json(directory / "analysis.json", safe)
            return {"status": "completed", "analysis": safe, "scope": "fit"}
        pipeline = TransformerPipeline(module.build_transformer(context), spec, task) if spec.custom_component == "transformer" else module.build_pipeline(context)
    elif request["mode"] == "refit" and request.get("model_path") and spec.engine in {"flaml", "autogluon", "lightautoml"}:
        with Path(request["model_path"]).open("rb") as stream:
            pipeline = cloudpickle.load(stream)
    else:
        pipeline = build_pipeline(spec, task, context)
    if spec.shift_strategy != "none":
        context["unlabelled_features"] = read_frame(directory / "unlabelled_X.json")
    phases["load_and_construct_seconds"] = time.monotonic() - phase_started
    phase_started = time.monotonic()
    if request["mode"] == "refit" and hasattr(pipeline, "refit"):
        summary = pipeline.refit(X.copy(), y.copy(), context)
    else:
        summary = pipeline.fit(X.copy(), y.copy(), context)
    summary = summary.model_dump(mode="json") if hasattr(summary, "model_dump") else summary
    summary = TrainingSummary.model_validate(summary).model_dump(mode="json")
    if request.get("pilot"):
        summary["budget"] = {**(summary.get("budget") or {}), "pilot_support": context["pilot_support"]}
    phases["fit_seconds"] = time.monotonic() - phase_started
    if summary["rows"] != len(X):
        raise ValueError("Candidate training summary does not match the supplied fit rows")
    model_path = directory / "model.pkl"
    phase_started = time.monotonic()
    with model_path.open("wb") as stream:
        cloudpickle.dump(pipeline, stream)
    with model_path.open("rb") as stream:
        restored = cloudpickle.load(stream)
    phases["save_load_seconds"] = time.monotonic() - phase_started
    phase_started = time.monotonic()
    outputs, hashes = {}, {"model.pkl": file_hash(model_path)}
    features = getattr(restored, "preprocessor", None)
    if features is not None and hasattr(features, "output_features"):
        schema = {"names": features.output_features, "feature_count": len(features.output_features)}
        atomic_json(directory / "output_schema.json", schema)
        hashes["output_schema.json"] = file_hash(directory / "output_schema.json")
        if summary.get("preprocessing") is not None:
            summary["preprocessing"].update(schema_reference="output_schema.json", schema_sha256=hashes["output_schema.json"])
    phases["prediction_rows"] = 0
    phases["prediction_seconds"] = 0.
    for part in request["predict_splits"]:
        features = read_frame(directory / f"{part}_X.json")
        phases["prediction_rows"] += len(features)
        prediction_context = {**context, "split_id": part}
        prediction_started = time.monotonic()
        predicted = restored.predict(features.copy(), prediction_context)
        phases["prediction_seconds"] += time.monotonic() - prediction_started
        if hasattr(predicted, "model_dump"):
            predicted = predicted.model_dump(mode="json")
        bundle = PredictionBundle.model_validate(predicted)
        from .identity import artifact_scope
        purpose = "search_validation" if context.get("rolling_folds") and part == "validation" else "frozen_inference" if request["mode"] == "refit" else "research_validation"
        bundle.provenance = artifact_scope(context, spec, task, X.index, features.index, purpose)
        path = directory / f"{part}_predictions.json"
        atomic_json(path, bundle)
        outputs[part] = path.name
        hashes[path.name] = file_hash(path)
        if len(features):
            short = features.iloc[:min(5, len(features))]
            before = pipeline.predict(short.copy(), prediction_context)
            after = restored.predict(short.copy(), prediction_context)
            before = PredictionBundle.model_validate(before.model_dump() if hasattr(before, "model_dump") else before)
            after = PredictionBundle.model_validate(after.model_dump() if hasattr(after, "model_dump") else after)
            if before.row_keys != after.row_keys or before.classes != after.classes:
                raise ValueError("Model serialization changed row/class identity")
            for target in task.targets:
                left, right = np.asarray(before.values[target]), np.asarray(after.values[target])
                equal = np.array_equal(left, right) if left.dtype.kind in {"U", "S", "O"} else np.allclose(left, right, rtol=1e-8, atol=1e-10)
                if not equal:
                    raise ValueError("Model serialization changed predictions")
    frozen = restored.frozen_spec() if hasattr(restored, "frozen_spec") else spec
    phases["predict_and_verify_seconds"] = time.monotonic() - phase_started
    phases["controller_staging_seconds"] = context.get("staging_seconds", 0)
    phases["forecast_prediction_rows"] = context.get("forecast_prediction_rows", phases["prediction_rows"])
    summary["execution_phases"] = phases
    from .identity import artifact_scope
    summary["provenance"] = artifact_scope(context, spec, task, X.index, purpose="fitted_pipeline")
    for path in directory.rglob("*"):
        if path.is_file() and (any(part.startswith("autogluon_") for part in path.relative_to(directory).parts)
                or path.name.startswith("flaml_") or path.name in {"candidate.py", "optuna_trials.json"}):
            hashes[str(path.relative_to(directory))] = file_hash(path)
    return {"status": "completed", "outputs": outputs, "hashes": hashes, "training_summary": summary,
        "frozen_pipeline": frozen.model_dump(mode="json"), "fit_row_keys_hash": digest(list(X.index)),
        "fit_rows": len(X), "serialization_verified": True}


def main(request_path: str):
    path = Path(request_path).resolve()
    directory = path.parent
    started = time.monotonic()
    request = read_json(path)
    request_hash = digest(request) if request.get('mode') == 'oof' else None
    try:
        result = run_request(request, directory)
    except BaseException as error:
        status = "insufficient_pilot_support" if isinstance(error, InsufficientPilotSupport) else "import_error" if isinstance(error, (ImportError, ModuleNotFoundError)) else "memory_limit" if isinstance(error, MemoryError) else "error"
        first_line = str(error).splitlines()[0] if str(error) else ""
        first_line = re.sub(r"(['\"])(.*?)\1", lambda m: m[0] if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,100}", m[2]) else "'<value>'", first_line)
        if re.fullmatch(r"[0-9.,; +\-]+", first_line):
            first_line = "Numeric input details retained locally"
        result = {"status": status, "error": redact(f"{type(error).__name__}: {first_line[:700]}"),
            "traceback": redact(traceback.format_exc()), "diagnostic_code": status,
            "diagnostic_artifact": str(directory / 'worker_result.json')}
    result["seconds"] = time.monotonic() - started
    if request.get('mode') == 'oof':
        result['request_hash'] = request_hash
    atomic_json(directory / "worker_result.json", result)


if __name__ == "__main__":
    main(sys.argv[1])
