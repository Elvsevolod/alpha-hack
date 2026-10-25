"""Frozen development-only comparisons; no paid requests and no auto-promotion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_agent.contracts import PipelineSpec
from automl_agent.data import Dataset
from automl_agent.evaluation import score_predictions, confusion_details
from automl_agent.executor import stage_frame, terminate_tree
from automl_agent.identity import feature_groups
from automl_agent.protocols import configure
from automl_agent.util import atomic_json, digest, file_hash, read_json
from automl_agent.validation import choose_validation, ordered_periods


def tasks():
    result = {}
    for name, folder in (("regression", "Регрессия"), ("classification", "Многоклассовая классификация")):
        dataset = Dataset(ROOT / "data_for_participants" / folder / "task.md")
        task, unresolved = dataset.draft()
        if task is None or unresolved:
            raise ValueError("Offline benchmark requires a resolved explicit contract")
        dataset.activate(task)
        result[name] = dataset
    return result


def frozen_protocol(output):
    from automl_agent.capabilities import installed_versions
    datasets = tasks()
    references = read_json(ROOT / "outputs/plan_v081_implementation/references.json")["references"]
    versions = {name: {"runtime_root": row["runtime_path"], "zip_sha256": row["zip_sha256"]} for name, row in references.items()}
    versions["candidate"] = {"runtime_root": str(ROOT), "sources": {str(p.relative_to(ROOT)): file_hash(p)
        for p in [*sorted((ROOT / "automl_agent").glob("*.py")), *sorted((ROOT / "context_compressor").glob("*.py")), ROOT / "solution.py"]}}
    common = {"regression": [dict(model="ridge", hyperparameters={"alpha": 10.}),
        dict(model="lightgbm", hyperparameters={"n_estimators": 150, "num_leaves": 15, "learning_rate": .05}),
        dict(model="extra_trees", hyperparameters={"n_estimators": 60, "min_samples_leaf": 3})],
        "classification": [dict(model="logistic", hyperparameters={"C": .1, "max_iter": 300}),
        dict(model="lightgbm", hyperparameters={"n_estimators": 150, "num_leaves": 15, "learning_rate": .05}),
        dict(model="extra_trees", hyperparameters={"n_estimators": 60, "min_samples_leaf": 3})]}
    ablations = {"regression": [{"id": "drop_period", "base": 1, "exclude_period": True},
        {"id": "object_only", "base": 1, "object_only": True}, {"id": "knn", "recipe": {"model": "knn", "hyperparameters": {"n_neighbors": 15, "weights": "distance"}}},
        {"id": "huber", "base": 1, "changes": {"training_objective": "huber", "hyperparameters": {"n_estimators": 150, "num_leaves": 15, "learning_rate": .05, "alpha": .9}}},
        {"id": "log_naive", "base": 1, "changes": {"target_transform": "log1p"}},
        {"id": "log_smearing", "base": 1, "changes": {"target_transform": "log1p", "log_inverse": "oof_smearing"}}],
        "classification": [{"id": "knn", "recipe": {"model": "knn", "hyperparameters": {"n_neighbors": 15, "weights": "distance"}}},
        {"id": "decision_legacy", "base": 1, "changes": {"prediction_policy": "class_offsets", "postprocess_fit_policy": "inner_holdout"}},
        {"id": "decision_oof", "base": 1, "changes": {"prediction_policy": "class_offsets", "postprocess_fit_policy": "pooled_oof", "decision_method": "regularized"}},
        {"id": "weights", "base": 1, "changes": {"shift_strategy": "importance_weighting"}}]}
    splits, protocol_splits = {}, {}
    for name, dataset in datasets.items():
        plan = choose_validation(dataset, seed=42)
        if name == "regression":
            period = ordered_periods(dataset.train[dataset.task.time_column])
            last = period.max()
            fit = np.flatnonzero(period < last).tolist()
            evaluation = np.flatnonzero(period == last).tolist()
            stress = evaluation
        else:
            revised = configure(dataset, plan, "low_overlap")
            fit, evaluation = revised.fit, revised.validation
            stress = revised.observations[-1]["stress_rows"]
        splits[name] = {"fit": fit, "evaluation": evaluation, "stress": stress,
            "data_hash": dataset.source_hash, "task": dataset.task.model_dump(mode="json"),
            "fit_keys_hash": digest(fit), "evaluation_keys_hash": digest(evaluation), "known_development_labels": True}
        if name == "regression":
            from automl_agent.validation import inner_split
            a, b = inner_split(dataset.features(plan.fit), dataset.train.iloc[plan.fit][dataset.task.targets].reset_index(drop=True), dataset.task, plan.strategy, 42)
            rolling = configure(dataset, plan, "rolling")
            pool = sorted(rolling.fit + rolling.validation)
            positions = {row: index for index, row in enumerate(pool)}
            protocol_splits[name] = {
                "old_inner": {"fit": [plan.fit[i] for i in a], "evaluation": [plan.fit[i] for i in b]},
                "old_holdout": {"fit": plan.fit, "evaluation": plan.validation},
                "rolling": {"fit": pool, "evaluation": rolling.validation,
                    "rolling_folds": [{key: [positions[row] for row in value] for key, value in part.items()} for part in rolling.development_folds]}}
            ablations[name].append({"id": "weights", "base": 1, "changes": {"shift_strategy": "importance_weighting"}})
            dictionary = pd.read_csv(dataset.paths["data_dictionary.csv"])
            temperature = dictionary.loc[dictionary["role"].eq("feature") & dictionary["description_ru"].str.contains("температур", case=False), "feature"].tolist()
            ablations[name].append({"id": "weather_aggregate", "base": 1, "changes": {"transformations": [{"name": "aggregate",
                "parameters": {"columns": temperature, "functions": ["mean", "std", "min", "max"], "prefix": "temperature_block", "drop_source": True}}]}})
        else:
            from types import SimpleNamespace
            local = SimpleNamespace(task=dataset.task, train=dataset.train.iloc[fit].reset_index(drop=True),
                test=dataset.test, source_hash=digest([dataset.source_hash, fit]))
            inner = choose_validation(local)
            protocol_splits[name] = {}
            for mode in ("holdout", "exact_group", "low_overlap"):
                p = configure(local, inner, mode)
                protocol_splits[name][mode] = {"fit": [fit[i] for i in p.fit], "evaluation": [fit[i] for i in p.validation]}
            target = dataset.task.targets[0]
            counts = dataset.train.iloc[fit][target].value_counts()
            class_weight = {str(label): float(np.sqrt(counts.max() / count)) for label, count in counts.items()}
            for method in ("argmax", "legacy", "regularized"):
                changes = {"hyperparameters": {**common[name][1]["hyperparameters"], "class_weight": class_weight}}
                if method != "argmax":
                    changes.update(prediction_policy="class_offsets", postprocess_fit_policy="pooled_oof" if method == "regularized" else "inner_holdout", decision_method="regularized" if method == "regularized" else "legacy")
                ablations[name].append({"id": "sqrt_weights_" + method, "base": 1, "changes": changes})
            for profile in ("selection", "portfolio"):
                recipe = {"engine": "autogluon", "model": "automl", "learners": ["GBM"], "n_trials": 1}
                if profile == "portfolio":
                    recipe["autogluon_configurations"] = {"GBM": [{"num_leaves": 15, "n_estimators": 150}, {"num_leaves": 31, "n_estimators": 150}]}
                ablations[name].append({"id": "autogluon_" + profile, "recipe": recipe})
        blend_members = [common[name][1], common[name][2]]
        if name == "regression":
            blend_members = [common[name][1], {**common[name][1], "training_objective": "huber"}]
        for strategy in ("uniform", "weighted_oof", "stacked_oof"):
            recipe = {"engine": "ensemble", "model": "blend", "members": ["fixed_a", "fixed_b"],
                "ensemble_strategy": strategy, "benchmark_members": blend_members}
            ablations[name].append({"id": "blend_" + strategy, "recipe": recipe})
            if name == "classification":
                ablations[name].append({"id": "blend_" + strategy + "_final_offsets", "recipe": {**recipe,
                    "prediction_policy": "class_offsets", "postprocess_fit_policy": "pooled_oof"}})
    manifest = {"format": 1, "evidence_type": "offline_recipe", "versions": versions,
        "library_versions": installed_versions(),
        "harness_sources": {str(path.relative_to(ROOT)): file_hash(path) for path in
            (Path(__file__), ROOT / "tools/plan_recipe_worker.py", ROOT / "tools/plan_acceptance.py")},
        "requirements_sha256": file_hash(ROOT / "requirements.txt"), "resources": {"cpu": 2, "memory_mb": 4096, "seconds": 45},
        "seed": 42, "common_portfolio": common, "ablations": ablations, "splits": splits, "protocol_splits": protocol_splits,
        "approval_for_qwen": False, "quality_status": "development_only_not_fresh_control",
        "acceptance": {"rmse_mean_no_worse": True, "macro_f1_mean_no_worse": True,
            "useful_relative_rmse": .01, "useful_absolute_f1": .005, "max_pair_rmse_loss": .005, "max_pair_f1_loss": .002,
            "minimum_full_agent_repeats": 3}, "automatic_promotion": False}
    manifest["protocol_hash"] = digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    if path.exists():
        raise ValueError("Frozen protocol exists; create a separately named cycle instead of changing it")
    atomic_json(path, manifest)
    return manifest


def compatible(manifest):
    from automl_agent.capabilities import installed_versions
    expected = manifest["protocol_hash"]
    if digest({k: v for k, v in manifest.items() if k != "protocol_hash"}) != expected:
        raise ValueError("Frozen protocol changed")
    if file_hash(ROOT / "requirements.txt") != manifest["requirements_sha256"]:
        raise ValueError("Dependencies changed since protocol freeze")
    if installed_versions() != manifest["library_versions"]:
        raise ValueError("Installed library versions changed since protocol freeze")
    for name, expected_hash in manifest["harness_sources"].items():
        if file_hash(ROOT / name) != expected_hash:
            raise ValueError("Comparison harness changed after freeze: " + name)
    for name, row in manifest["versions"]["candidate"]["sources"].items():
        if file_hash(ROOT / name) != row:
            raise ValueError("Candidate changed after freeze: " + name)
    for version in ("v076", "v081"):
        if file_hash(ROOT / "dist" / f"solution_{version}.zip") != manifest["versions"][version]["zip_sha256"]:
            raise ValueError("An immutable baseline changed")


def run_one(manifest, directory, dataset, split, version, recipe, identifier):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    rows, evaluation = split["fit"], split["evaluation"]
    X = dataset.features(rows); X.index = [f"train:{i}" for i in rows]
    y = dataset.train.iloc[rows][dataset.task.targets].copy(); y.index = X.index
    query = dataset.features(evaluation)
    # Rolling feedback retrieves predictions made before these historical labels
    # were fitted. Its keys must match the development rows, not a new test set.
    prefix = "train" if split.get("rolling_folds") else "external"
    query.index = [f"{prefix}:{i}" for i in evaluation]
    for name, frame in (("fit_X", X), ("fit_y", y), ("query_X", query)):
        stage_frame(directory / (name + ".json"), frame)
    from automl_agent.capabilities import installed_versions
    request = {"runtime_root": manifest["versions"][version]["runtime_root"], "task": split["task"], "recipe": recipe,
        "resources": manifest["resources"], "protocol_hash": manifest["protocol_hash"], "data_hash": split["data_hash"],
        "strategy": "temporal" if dataset.task.time_column else "random", "library_versions": installed_versions()}
    if split.get("rolling_folds"):
        request["rolling_folds"] = split["rolling_folds"]
    atomic_json(directory / "request.json", request)
    env = {key: value for key, value in os.environ.items() if not any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET")) and not key.startswith(("OPENAI_", "PYTHONPATH"))}
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic(); peak = cpu_seconds = 0.; reason = None
    with (directory / "execution.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(ROOT / "tools/plan_recipe_worker.py"), str(directory / "request.json")],
            cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    tree = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
                    memory = sum(item.memory_info().rss for item in tree if item.is_running()) / 1024 ** 2
                    peak = max(peak, memory)
                    cpu_seconds = max(cpu_seconds, sum(sum(item.cpu_times()[:2]) for item in tree if item.is_running()))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if time.monotonic() - started > manifest["resources"]["seconds"] or peak > manifest["resources"]["memory_mb"]:
                    reason = "timeout" if peak <= manifest["resources"]["memory_mb"] else "memory_limit"
                    terminate_tree(process)
                    break
                time.sleep(.1)
        except KeyboardInterrupt:
            reason = "interrupted"
            terminate_tree(process)
    result = read_json(directory / "result.json") if (directory / "result.json").exists() else {"status": reason or "error"}
    row = {"version": version, "recipe_id": identifier, "task": dataset.task.kind,
        "status": reason or result["status"], "evidence_type": "offline_recipe", "real_qwen": False,
        "seed": manifest["seed"], "evaluation_keys_hash": digest(evaluation), "output_schema": split["task"]["output"],
        "seconds": time.monotonic() - started, "peak_memory_mb": peak, "cpu_seconds": cpu_seconds,
        "protocol_hash": manifest["protocol_hash"], "data_hash": split["data_hash"], "split_hash": digest([rows, evaluation]),
        "resources": manifest["resources"], "valid_csv": False, "raw_metric": None, "stress_metric": None}
    if (directory / "prediction.json").is_file():
        prediction = read_json(directory / "prediction.json")
        if prediction["row_keys"] != query.index.tolist():
            raise ValueError("Prediction row order changed")
        values = {target: np.asarray(value) for target, value in prediction["values"].items()}
        actual = dataset.train.iloc[evaluation][dataset.task.targets]
        row["raw_metric"] = score_predictions(dataset.task, actual, values, prediction["prediction_type"])
        mask = np.isin(evaluation, split["stress"])
        row["stress_metric"] = score_predictions(dataset.task, actual.iloc[np.flatnonzero(mask)],
            {target: value[mask] for target, value in values.items()}, prediction["prediction_type"])
        output = dataset.train.iloc[evaluation][dataset.task.id_columns].copy()
        for target in dataset.task.targets:
            value = values[target]
            labels = np.asarray(dataset.task.classes[target])[value.argmax(axis=1)] if dataset.task.classes and prediction["prediction_type"] == "probabilities" else value
            output[target] = labels
            if dataset.task.classes:
                row["per_class"] = confusion_details(actual[target], labels, dataset.task.classes[target])
            else:
                square = (actual[target].to_numpy() - value) ** 2
                threshold = float(y[target].quantile(.99))
                row["sse"] = float(square.sum()); row["rows"] = len(square)
                row["tail_threshold_from_fit"] = threshold
                row["tail_sse"] = float(square[actual[target].to_numpy() >= threshold].sum())
        output.to_csv(directory / "submission.csv", index=False)
        row.update(valid_csv=True, fallback=prediction["fallback"], final_csv_sha256=file_hash(directory / "submission.csv"))
    row["disk_bytes"] = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    atomic_json(directory / "measurement.json", row)
    return row


def run(output):
    manifest = read_json(output / "protocol.json")
    compatible(manifest)
    datasets = tasks()
    rows = []
    for task_index, (name, dataset) in enumerate(datasets.items()):
        split = manifest["splits"][name]
        if dataset.source_hash != split["data_hash"]:
            raise ValueError("Task data changed after freeze")
        versions = list(manifest["versions"])
        if task_index % 2:
            versions.reverse()
        for index, recipe in enumerate(manifest["common_portfolio"][name]):
            for version in versions:
                identifier = f"common_{index}"
                row = run_one(manifest, output / name / version / identifier, dataset, split, version, recipe, identifier)
                rows.append(row); atomic_json(output / "results.json", rows)
                print(json.dumps({key: row[key] for key in ("version", "recipe_id", "task", "status", "raw_metric", "seconds")}), flush=True)
                if row["status"] == "interrupted":
                    raise KeyboardInterrupt("Interrupted recipe saved; series stopped")
        for experiment in manifest["ablations"][name]:
            recipe = dict(experiment.get("recipe", manifest["common_portfolio"][name][experiment.get("base", 0)]))
            recipe.update(experiment.get("changes", {}))
            if experiment.get("exclude_period"):
                recipe["exclude_features"] = [dataset.task.time_column]
            if experiment.get("object_only"):
                # Development metadata is explicit; no such names enter runtime.
                dictionary = pd.read_csv(dataset.paths["data_dictionary.csv"])
                weather = dictionary["description_ru"].str.contains("температур|осадк|снег|градус|ветр|туман|дней", case=False, regex=True)
                recipe["exclude_features"] = dictionary.loc[weather & dictionary["role"].eq("feature"), "feature"].tolist()
            row = run_one(manifest, output / name / "candidate" / experiment["id"], dataset, split, "candidate", recipe, experiment["id"])
            rows.append(row); atomic_json(output / "results.json", rows)
            print(json.dumps({key: row[key] for key in ("version", "recipe_id", "task", "status", "raw_metric", "seconds")}), flush=True)
            if row["status"] == "interrupted":
                raise KeyboardInterrupt("Interrupted recipe saved; series stopped")
        for protocol, local_split in manifest["protocol_splits"][name].items():
            internal = {**split, **local_split, "stress": local_split["evaluation"]}
            for index, recipe in enumerate(manifest["common_portfolio"][name]):
                identifier = f"protocol_{protocol}_{index}"
                row = run_one(manifest, output / name / "protocols" / identifier, dataset, internal, "candidate", recipe, identifier)
                row["evidence_type"] = "offline_protocol_search_feedback"
                atomic_json(output / name / "protocols" / identifier / "measurement.json", row)
                rows.append(row); atomic_json(output / "results.json", rows)
                print(json.dumps({key: row[key] for key in ("version", "recipe_id", "task", "status", "raw_metric", "seconds")}), flush=True)
                if row["status"] == "interrupted":
                    raise KeyboardInterrupt("Interrupted recipe saved; series stopped")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "run"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen_protocol(args.output) if args.mode == "freeze" else run(args.output)
