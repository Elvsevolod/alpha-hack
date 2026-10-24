"""Frozen outer evaluation of the actual published CSV, separate from search.

Prepare, select AutoML on research validation, freeze, execute every registered
run, and only then open outer labels. Containers receive task copies without
the sealed_evaluation directory. This tooling is never shipped in the solution.
"""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis, PipelineSpec, TaskSpec, ToolCommand
from automl_agent.controller import Controller, SYSTEM
from automl_agent.data import Dataset
from automl_agent.evaluation import score_predictions
from automl_agent.util import atomic_json, digest, file_hash, read_json
from automl_agent.validation import choose_validation, selection_for

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["reference", "fixed_automl", "agent", "v008", "fixed_portfolio"]
PORTFOLIO = [
    {"id": "catboost", "fraction": .25, "pipeline": {"model": "catboost", "representation": "native", "imputation": {"numeric": "native"},
        "hyperparameters": {"iterations": 2500, "depth": 6, "learning_rate": .04}}},
    {"id": "lightgbm", "fraction": .2, "pipeline": {"model": "lightgbm", "representation": "native", "imputation": {"numeric": "native"},
        "hyperparameters": {"n_estimators": 2500, "num_leaves": 31, "learning_rate": .03, "reg_lambda": 5}}},
    {"id": "extra_trees", "fraction": .15, "pipeline": {"model": "extra_trees", "hyperparameters": {"n_estimators": 500, "min_samples_leaf": 2}}},
    {"id": "flaml", "fraction": .4, "pipeline": {"engine": "flaml", "model": "library_search", "n_trials": None, "learners": ["lgbm", "extra_tree", "catboost"]}},
]
H = Hypothesis(observation="Prespecified comparison arm on frozen research data",
    expected_effect="Measure the actual selected result under an equal available budget",
    changed_component="fixed_comparator", acceptance="Published outer RMSE/Macro F1, reliability and costs reported separately")


def code_hash(root=ROOT):
    return digest({str(p.relative_to(root)): file_hash(p) for p in [root / "solution.py", *sorted((root / "automl_agent").rglob("*.py")),
        *[p for p in sorted((root / "context_compressor").glob("*.py")) if p.name != "verify.py"]]})


def indices_hash(plan):
    return digest({name: getattr(plan, name) for name in ("fit", "validation", "control")})


def archived_runtime(archive_path):
    """Inspect the saved flat archive without executing its bootstrap."""
    with zipfile.ZipFile(archive_path) as archive:
        source = archive.read("agent_bundle.py").decode("utf-8")
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"BUNDLE_SHA256", "FILES", "PAYLOAD"}:
            values[node.targets[0].id] = ast.literal_eval(node.value)
    body = base64.b85decode(values["PAYLOAD"])
    if hashlib.sha256(body).hexdigest() != values["BUNDLE_SHA256"]:
        raise ValueError("Previous archive embedded runtime hash mismatch")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    if set(contents) != set(values["FILES"]) or any(hashlib.sha256(content).hexdigest() != values["FILES"][name] for name, content in contents.items()):
        raise ValueError("Previous archive source integrity mismatch")
    return contents


def immutable(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def dispatch_data(outcome):
    data = outcome.get("data")
    return data if isinstance(data, dict) else outcome


def dispatch_result(outcome):
    result = dispatch_data(outcome).get("result")
    if not isinstance(result, dict):
        result = outcome.get("result")
    return result if isinstance(result, dict) else {}


def run_workspace(task_path, output):
    task_path = Path(task_path)
    workspace = Path(output).parent / "workspaces" / Path(output).stem
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(task_path.parent, workspace, ignore=shutil.ignore_patterns(".agent_work", "submission.csv"))
    return workspace / task_path.name


def paired_win(new, reference, policy):
    if len(new) != 3 or len(reference) != 3 or any(x is None or not np.isfinite(x) for x in [*new, *reference]):
        return {"accepted": False, "reason": "incomplete_or_failed_pairs"}
    differences = [(a - b) if policy.maximize else (b - a) for a, b in zip(new, reference)]
    epsilons = [policy.tolerance(b) for b in reference]
    meaningful = [d > e and not np.isclose(d, e, rtol=1e-12, atol=0) for d, e in zip(differences, epsilons)]
    drawdown = [bool(d >= -e or np.isclose(d, -e, rtol=1e-12, atol=0)) for d, e in zip(differences, epsilons)]
    mean = float(np.mean(differences))
    return {"accepted": sum(meaningful) >= 2 and all(drawdown) and mean > 0,
        "differences": differences, "epsilon": epsilons, "D": epsilons, "meaningful_pairs": sum(meaningful),
        "drawdown_satisfied": drawdown, "mean": mean, "median": float(np.median(differences)), "worst": min(differences)}


def prepare(source, output):
    if output.exists():
        raise ValueError("Use a new protocol directory; existing evidence is immutable")
    output.mkdir(parents=True)
    tasks = {}
    for path in sorted(source.glob("*/task.md")):
        dataset = Dataset(path)
        task, questions = dataset.draft()
        if task is None:
            raise ValueError(str(questions))
        dataset.activate(task)
        outer = choose_validation(dataset, seed=42)
        development = sorted(outer.fit + outer.validation)
        holdout = sorted(outer.control)
        directory = output / "tasks" / task.kind
        directory.mkdir(parents=True)
        dataset.train.iloc[development].to_csv(directory / "train.csv", index=False)
        dataset.train.iloc[holdout].drop(columns=task.targets).to_csv(directory / "test.csv", index=False)
        template = dataset.train.iloc[holdout][task.id_columns].copy()
        for target in task.targets:
            template[target] = 0
        template[task.output.columns].to_csv(directory / "sample_submission.csv", index=False)
        if dataset.paths["data_dictionary.csv"]:
            shutil.copyfile(dataset.paths["data_dictionary.csv"], directory / "data_dictionary.csv")
        (directory / "task.md").write_text("Comparative development copy. Row counts and paths below refer to the original description; use the supplied local train.csv/test.csv/sample_submission.csv. The outer evaluation labels are unavailable.\n\n" + dataset.text)
        values = task.model_dump(mode="json", exclude={"source_hash"})
        values["explicit_validation"] = outer.strategy
        immutable(directory / "task_spec.json", values)
        labels = dataset.train.iloc[holdout][task.id_columns + task.targets]
        sealed = output / "sealed_evaluation" / task.kind
        sealed.mkdir(parents=True)
        labels.to_csv(sealed / "labels.csv", index=False)
        copied = Dataset(directory / "task.md")
        confirmed, _ = copied.draft()
        copied.activate(confirmed)
        plan = choose_validation(copied, seed=42)
        tasks[task.kind] = {"source_hash": dataset.source_hash, "task": confirmed.model_dump(mode="json"),
            "data_version": copied.source_hash, "validation_hash": plan.hash, "validation": plan.model_dump(mode="json"),
            "outer_fit_indices_hash": digest(development), "outer_test_indices_hash": digest(holdout),
            "outer_strategy": outer.strategy, "outer_labels_sha256": file_hash(sealed / "labels.csv"),
            "outer_labels_previously_used": True, "evaluation_scope": "development_public_data",
            "files": {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()}}
    immutable(output / "DATA_MANIFEST.json", {"tasks": tasks, "code_sha256": code_hash(),
        "outer_labels_not_available_to_search_or_refit": True, "transferability": "not_established"})
    return tasks


def select_automl(task_path, output, seconds=240):
    """Preliminary selection uses only research validation, never outer labels."""
    output.mkdir(parents=True, exist_ok=False)
    immutable(output / "MANIFEST.json", {"stage": "preliminary_automl_selection", "code_sha256": code_hash(),
        "data_version": Dataset(task_path).source_hash, "candidate_seconds": seconds, "training_seed": 42,
        "validation_seed": 42, "n_trials": 40, "cpu": 6, "memory_gib": 5,
        "planned": ["flaml", "autogluon"], "selection": "best_research_validation_primary_metric",
        "outer_labels_available": False, "closed_control_opened": False})
    rows = []
    for engine in ("flaml", "autogluon"):
        directory = output / engine
        shutil.copytree(task_path.parent, directory)
        c = Controller(directory / "task.md", Limits(seconds=seconds * 2 + 60, cpu=6, memory_mb=5120), run_id="preliminary")
        try:
            c.prepare()
            pipeline = PipelineSpec(engine=engine, model="library_search", n_trials=40,
                learners=["lgbm", "extra_tree", "catboost"] if engine == "flaml" else ["GBM", "CAT", "XGB"])
            outcome = c.dispatch(ToolCommand(operation_id="engine_selection", tool="run_automl", arguments={
                "pipeline": pipeline.model_dump(mode="json"), "hypothesis": H.model_dump(), "budget_seconds": seconds}))
            rows.append({"engine": engine, "outcome": outcome, "validation_hash": c.plan.hash,
                "data_version": c.dataset.source_hash, "control_not_opened": not c.journal.get("control_opened")})
            atomic_json(output / "PRELIMINARY.json", {"runs": rows + [{"engine": name, "status": "not_started"}
                for name in ("flaml", "autogluon") if name not in {r["engine"] for r in rows}], "planned": 2, "outer_scores_used": False})
        finally:
            c.close()
    valid = [r for r in rows if dispatch_result(r["outcome"]).get("status") == "completed"
        and dispatch_result(r["outcome"]).get("score") is not None]
    if not valid:
        raise ValueError("No AutoML engine completed a valid preliminary candidate")
    winner = (max if c.task.metric.maximize else min)(valid, key=lambda r: dispatch_result(r["outcome"])["score"])
    immutable(output / "SELECTED.json", {"engine": winner["engine"], "selection": "research_validation_only", "runs": rows,
        "code_sha256": code_hash(), "outer_scores_used": False})
    return winner["engine"]


def validate_selection(record, task_data, expected_code_hash, expected_task_hash):
    if record.get("code_sha256") != expected_code_hash:
        raise ValueError("AutoML selection belongs to a different runtime")
    if (record.get("engine") not in {"flaml", "autogluon"}
            or record.get("selection") != "research_validation_only" or record.get("outer_scores_used") is not False):
        raise ValueError("AutoML selection must precede outer evaluation")
    runs = record.get("runs", [])
    if [run.get("engine") for run in runs] != ["flaml", "autogluon"] or any(run["data_version"] != task_data["data_version"]
            or run["validation_hash"] != task_data["validation_hash"] or run["control_not_opened"] is not True for run in runs):
        raise ValueError("AutoML selection belongs to a different data/protocol or opened control")
    task = TaskSpec.model_validate(task_data["task"])
    metric = selection_for(task)
    valid = []
    for run in runs:
        outcome = run.get("outcome", {})
        result = dispatch_result(outcome)
        experiment = dispatch_data(outcome).get("experiment", {})
        pipeline = experiment.get("pipeline", {})
        if (pipeline.get("engine") != run["engine"] or experiment.get("task_hash") != expected_task_hash
                or experiment.get("validation_hash") != run["validation_hash"]
                or experiment.get("data_version") != run["data_version"]
                or experiment.get("status") != "full" or result.get("scope") != "full"
                or result.get("validation_hash") != run["validation_hash"]
                or result.get("experiment_id") != experiment.get("experiment_id")):
            raise ValueError("AutoML selection result does not match its recorded experiment")
        score = result.get("score")
        if result.get("status") == "completed" and isinstance(score, (int, float, np.number)) and np.isfinite(score):
            valid.append((run["engine"], float(score)))
    if not valid:
        raise ValueError("AutoML selection has no completed candidate with a finite validation score")
    winner = (max if metric.maximize else min)(valid, key=lambda item: item[1])[0]
    if record["engine"] != winner:
        raise ValueError("Selected AutoML engine did not win on research validation")
    return record["engine"]


def freeze(output, selected, candidate):
    data = read_json(output / "DATA_MANIFEST.json")
    if set(selected) != set(data["tasks"]):
        raise ValueError("A validation-selected AutoML engine is required for each task")
    engines, selection_evidence = {}, {}
    for kind, path in selected.items():
        path = Path(path).resolve()
        record = read_json(path)
        task_dataset = Dataset(output / "tasks" / kind / "task.md")
        task = task_dataset.explicit()
        if task is None:
            raise ValueError("The frozen task must have an explicit, reproducible contract")
        task = task_dataset.activate(task)
        if task_dataset.source_hash != data["tasks"][kind]["data_version"]:
            raise ValueError("Frozen task data differs from the data manifest")
        engines[kind] = validate_selection(record, data["tasks"][kind], code_hash(), task.hash)
        selection_evidence[kind] = {"path": str(path), "sha256": file_hash(path), "code_sha256": record["code_sha256"]}
    runs = []
    for seed in (17, 42, 73):
        for arm in ARMS:
            for kind in sorted(data["tasks"]):
                runs.append({"id": f"{kind}_{arm}_{seed}_0", "kind": kind, "arm": arm, "seed": seed, "repeat": 0})
    for repeat in (1, 2):
        for arm in ("agent", "v008"):
            for kind in sorted(data["tasks"]):
                runs.append({"id": f"{kind}_{arm}_42_{repeat}", "kind": kind, "arm": arm, "seed": 42, "repeat": repeat})
    archive = ROOT / "dist/solution_v008.zip"
    old = archived_runtime(archive)
    current = archived_runtime(candidate)
    candidate_core_hash = digest({name: hashlib.sha256(body).hexdigest() for name, body in current.items()
        if name == "solution.py" or name.startswith(("automl_agent/", "context_compressor/")) and name.endswith(".py")})
    if candidate_core_hash != code_hash() or hashlib.sha256(current["knowledge_runtime/knowledge.lock.json"]).hexdigest() != file_hash(ROOT / "knowledge_runtime/knowledge.lock.json"):
        raise ValueError("Candidate archive does not contain this exact runtime and knowledge lock")
    limits = {}
    budget_source = old["automl_agent/budget.py"].decode()
    for node in ast.parse(budget_source).body:
        if isinstance(node, ast.ClassDef) and node.name == "Limits":
            limits = {field.target.id: ast.get_source_segment(budget_source, field.value) for field in node.body if isinstance(field, ast.AnnAssign) and field.value is not None}
    manifest = {"protocol": "v010_outer_selected_csv_v1", "data_manifest_sha256": file_hash(output / "DATA_MANIFEST.json"),
        "code_sha256": code_hash(), "v008_sha256": file_hash(archive), "selected_automl": engines,
        "candidate_archive": candidate.name, "candidate_archive_sha256": file_hash(candidate),
        "knowledge_lock_sha256": file_hash(ROOT / "knowledge_runtime/knowledge.lock.json"),
        "requirements_sha256": file_hash(ROOT / "requirements.txt"), "system_prompt_sha256": digest(SYSTEM),
        "comparison_tooling_sha256": digest({name: file_hash(ROOT / "tools" / name)
            for name in ("benchmark.py", "comparison_protocol.py", "run_comparison_series.py")}),
        "automl_selection_evidence": selection_evidence,
        "v008_code_sha256": digest({name: hashlib.sha256(body).hexdigest() for name, body in old.items()
            if name == "solution.py" or name.startswith(("automl_agent/", "context_compressor/")) and name.endswith(".py")}),
        "v008_builtin_limits": limits, "v008_protocol_implementation_sha256": hashlib.sha256(old["automl_agent/validation.py"]).hexdigest(),
        "roles": {"reference": "internal fixed anchor", "fixed_automl": "validation-selected fixed AutoML",
            "fixed_portfolio": "prespecified queue with frozen fractions", "v008": "unmodified previous archive",
            "organizer_baseline": "unknown numerical thresholds; excluded from matrix"},
        "portfolio": PORTFOLIO, "portfolio_selection": "controller's unchanged frozen control rule",
        "internal_seconds": 10500, "external_seconds": 10800, "cpu": 6, "memory_gib": 5,
        "minimum_finish_reserve_seconds": 300, "max_candidates": None, "max_qwen_calls": None,
        "automl_full": {"n_trials": None, "max_iter_argument_present": False, "early_stop": False,
            "normal_stops": ["calculated_time_budget_exhausted", "engine_space_exhausted"], "phase_reserves": "measured and logged"},
        "paired_rule": {"regression": {"direction": "reference-new", "min_delta_abs": 1e-12, "min_delta_rel": .001},
            "classification": {"direction": "new-reference", "min_delta_abs": .001, "min_delta_rel": 0},
            "D_equals_epsilon": True, "wins_required": 2, "pairs": 3, "mean_strictly_positive": True},
        "technical_failure_rate_allowed": 0, "all_runs_counted": True,
        "series_failure_policy": "After a failed pair, retain both actual outcomes, cancel unstarted runs and repair before a new immutable series",
        "fallback_csv": "score actual valid CSV and retain incomplete/failure status; missing invalid score stays null",
        "qwen_repeats": "3 per stochastic arm/task at training seed 42 including main run; no endpoint seed unless supported",
        "reuse_policy": "No automatic reuse from other studies", "runs": runs,
        "metrics_scope": "development public data, not independent transferability or official threshold"}
    immutable(output / "MANIFEST.json", manifest)
    immutable(output / "REGISTRY.json", [{**run, "status": "not_started"} for run in runs])
    return manifest


def run_one(task_path, arm, seed, seconds, output, engine=None):
    if arm == "fixed_automl" and engine not in {"flaml", "autogluon"}:
        raise ValueError("fixed_automl requires --engine flaml or --engine autogluon")
    if arm != "fixed_automl" and engine is not None:
        raise ValueError("--engine is only valid for fixed_automl")
    if output.exists():
        raise ValueError("Run receipt already exists; recover the original run instead of overwriting")
    task_path = run_workspace(task_path, output)
    mode = "full_comparison" if arm in {"fixed_automl", "fixed_portfolio"} else "research"
    c = Controller(task_path, Limits(seconds=seconds, cpu=6, memory_mb=5120), run_id="comparison",
        training_seed=seed, budget_mode=mode,
        research_focus="Choose hypotheses autonomously using the immutable protocol. Full wall-clock budget includes model calls, control and refit. Outer labels are unavailable; do not revise the frozen comparison split.")
    c.journal.put("comparison_protocol_locked", True)
    started = time.time()
    queue_outcomes = []
    try:
        if arm in {"agent", "reference"}:
            report = c.run(no_llm=arm == "reference")
        else:
            c.prepare()
            available = max(1, c.budget.remaining - c.budget.final_reserve(c.finalist_costs()))
            if arm == "fixed_automl":
                queue = [{"id": engine, "fraction": 1, "pipeline": {"engine": engine, "model": "library_search", "n_trials": None,
                    "learners": ["lgbm", "extra_tree", "catboost"] if engine == "flaml" else ["GBM", "CAT", "XGB"]}}]
            elif arm == "fixed_portfolio":
                queue = PORTFOLIO
            else:
                raise ValueError("Use the unmodified archive entry point for v008")
            for item in queue:
                pipeline = PipelineSpec.model_validate({**item["pipeline"], "seed": seed})
                allocation = min(available * item["fraction"], max(0, c.budget.remaining - c.budget.final_reserve(c.finalist_costs())))
                if allocation <= 1:
                    c.journal.event("fixed_queue_not_started", {"id": item["id"], "reason": "finish_reserve"})
                    queue_outcomes.append({"id": item["id"], "status": "not_started", "reason": "finish_reserve"})
                    continue
                outcome = c.dispatch(ToolCommand(operation_id="fixed_" + item["id"], tool="run_automl" if pipeline.engine in {"flaml", "autogluon"} else "train_model",
                    arguments={"pipeline": pipeline.model_dump(mode="json"), "hypothesis": H.model_dump(), "budget_seconds": allocation}))
                result = dispatch_result(outcome)
                data = dispatch_data(outcome)
                experiment = data.get("experiment", {})
                queue_outcomes.append({"id": item["id"], "status": result.get("status", outcome.get("status")),
                    "experiment_id": result.get("experiment_id", experiment.get("experiment_id")),
                    "error": result.get("error", outcome.get("error"))})
            c.journal.put("imputation_skip_reason", "Frozen fixed arm; no adaptive queue changes")
            c.freeze("Fixed prespecified comparison queue completed")
            c.finalize()
            report = c.report()
        publication = report.get("publication")
        if publication and c.publisher.output.is_file():
            shutil.copyfile(c.publisher.output, output.with_suffix(".csv"))
        receipt = {"arm": arm, "seed": seed, "report": report, "elapsed_seconds": time.time() - started,
            "data_version": c.dataset.source_hash, "validation_hash": c.plan.hash if c.plan else None,
            "validation_indices_hash": indices_hash(c.plan), "queue": queue_outcomes,
            "research": [{"experiment_id": spec.experiment_id, "engine": spec.pipeline.engine,
                "model": spec.pipeline.model, "n_trials": spec.pipeline.n_trials, "budget_mode": spec.budget_mode, "budget_seconds": spec.budget_seconds,
                "scope": result.scope, "status": result.status, "validation_score": result.score,
                "budget": result.training_summary.get("budget"), "error": result.error}
                for spec, result in c.records() if result],
            "code_sha256": code_hash(), "csv_sha256": file_hash(output.with_suffix(".csv")) if output.with_suffix(".csv").exists() else None,
            "knowledge_lock_sha256": file_hash(ROOT / "knowledge_runtime/knowledge.lock.json"),
            "requirements_sha256": file_hash(ROOT / "requirements.txt"), "system_prompt_sha256": digest(SYSTEM),
            "outer_score": None, "outer_labels_opened": False}
        immutable(output, receipt)
        return receipt
    finally:
        c.close()


def receipt_issues(receipt, run, manifest, entry):
    problems = []
    report = receipt.get("report", {})
    for field in ("state", "control_status", "refit_status"):
        if report.get(field) != "completed":
            problems.append(field + "_incomplete")
    if report.get("failure"):
        problems.append("controller_failure")
    if (report.get("publication") or {}).get("scope") != "full_train_refit":
        problems.append("published_csv_not_full_refit")
    expected_code = manifest["v008_code_sha256"] if run["arm"] == "v008" else manifest["code_sha256"]
    if receipt.get("code_sha256") != expected_code or receipt.get("data_version") != entry["data_version"]:
        problems.append("source_or_data_mismatch")
    if run["arm"] != "v008" and any(receipt.get(key) != manifest.get(key) for key in
            ("knowledge_lock_sha256", "requirements_sha256", "system_prompt_sha256")):
        problems.append("knowledge_environment_or_prompt_mismatch")
    expected_rows = digest({name: entry["validation"][name] for name in ("fit", "validation", "control")})
    if receipt.get("validation_indices_hash") != expected_rows:
        problems.append("validation_rows_mismatch")
    if receipt.get("arm") != run["arm"] or receipt.get("seed") != run["seed"]:
        problems.append("run_identity_mismatch")
    if run["arm"] in {"fixed_automl", "fixed_portfolio"}:
        expected_ids = ([manifest["selected_automl"][run["kind"]]] if run["arm"] == "fixed_automl"
            else [item["id"] for item in manifest["portfolio"]])
        queue = receipt.get("queue", [])
        queue_ids = [item.get("id") for item in queue]
        if (queue_ids != expected_ids or any(item.get("status") != "completed" for item in queue)):
            problems.append("fixed_queue_incomplete")
        engine = manifest["selected_automl"][run["kind"]] if run["arm"] == "fixed_automl" else "flaml"
        expected_models = ({engine: "library_search"} if run["arm"] == "fixed_automl"
            else {item["id"]: item["pipeline"]["model"] for item in manifest["portfolio"]})
        expected_engines = ({engine: engine} if run["arm"] == "fixed_automl"
            else {item["id"]: item["pipeline"].get("engine", "direct") for item in manifest["portfolio"]})
        research_by_id = {record.get("experiment_id"): record for record in receipt.get("research", [])
            if record.get("experiment_id")}
        research_ids = [record.get("experiment_id") for record in receipt.get("research", []) if record.get("experiment_id")]
        if (len(research_ids) != len(set(research_ids))
                or any(not item.get("experiment_id") or item["experiment_id"] not in research_by_id for item in queue)
                or len({item.get("experiment_id") for item in queue}) != len(queue)):
            problems.append("fixed_queue_research_mismatch")
        elif any((research := research_by_id[item["experiment_id"]]).get("status") != "completed"
                or research.get("scope") != "full" or research.get("budget_mode") != "full_comparison"
                or not isinstance(research.get("budget_seconds"), (int, float, np.number))
                or isinstance(research.get("budget_seconds"), (bool, np.bool_))
                or not np.isfinite(research["budget_seconds"]) or research["budget_seconds"] <= 0
                or research.get("engine") != expected_engines.get(item.get("id"))
                or research.get("model") != expected_models.get(item.get("id")) for item in queue):
            problems.append("fixed_queue_research_mismatch")
        searches = [record for record in receipt.get("research", []) if record["engine"] == engine and record["scope"] == "full"]
        if not searches:
            problems.append("full_automl_search_missing")
        for search in searches:
            budget = search.get("budget") or {}
            if search["status"] != "completed" or search["budget_mode"] != "full_comparison" or search["n_trials"] is not None or search["budget_seconds"] <= 1200 or budget.get("n_trials", 0) is not None:
                problems.append("full_automl_policy_not_executed")
            targets = budget.get("targets", {})
            if not targets:
                problems.append("full_automl_stop_evidence_missing")
            for target in targets.values():
                if target.get("max_iter_argument_present") is not False or target.get("stop_reason") not in manifest["automl_full"]["normal_stops"] or engine == "flaml" and target.get("early_stop") is not False:
                    problems.append("full_automl_stop_policy_invalid")
    return sorted(set(problems))


def score_csv(path, actual, task):
    predicted = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(predicted) != task.output.columns or len(predicted) != len(actual) or predicted.isna().any().any() or any(predicted[c].tolist() != actual[c].tolist() for c in task.id_columns):
        raise ValueError("CSV columns, row identities or row count differ from the frozen task")
    values = {target: predicted[target].to_numpy() for target in task.targets}
    if task.classes:
        if any(not set(v) <= set(task.classes[t]) for t, v in values.items()):
            raise ValueError("CSV contains a label outside the confirmed classes")
    else:
        values = {target: value.astype(float) for target, value in values.items()}
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError("CSV contains nonfinite predictions")
    score = score_predictions(task, actual[task.targets], values, task.output.prediction_type)
    if not np.isfinite(score):
        raise ValueError("The final CSV metric is not finite")
    return score


def evaluate(output):
    manifest = read_json(output / "MANIFEST.json")
    if file_hash(output / "DATA_MANIFEST.json") != manifest["data_manifest_sha256"]:
        raise ValueError("Frozen data manifest changed")
    registry = read_json(output / "REGISTRY.json")
    if {r["id"] for r in registry} != {r["id"] for r in manifest["runs"]} or len(registry) != len(manifest["runs"]):
        raise ValueError("Registry does not contain every planned run exactly once")
    original = {run["id"]: run for run in manifest["runs"]}
    if any(any(run.get(key) != value for key, value in original[run["id"]].items()) for run in registry):
        raise ValueError("A registered run identity differs from the frozen manifest")
    if any(r["status"] in {"not_started", "running"} for r in registry):
        raise ValueError("Complete or explicitly terminate every planned run before opening outer labels")
    data = read_json(output / "DATA_MANIFEST.json")
    scored = []
    for run in registry:
        task = TaskSpec.model_validate(data["tasks"][run["kind"]]["task"])
        labels_path = output / "sealed_evaluation" / run["kind"] / "labels.csv"
        if file_hash(labels_path) != data["tasks"][run["kind"]]["outer_labels_sha256"]:
            raise ValueError("Outer labels changed")
        row = {**run, "outer_score": None, "csv_valid": False}
        receipt_path = output / "receipts" / (run["id"] + ".json")
        receipt = read_json(receipt_path) if receipt_path.is_file() else {}
        row["acceptance_issues"] = receipt_issues(receipt, run, manifest, data["tasks"][run["kind"]])
        path = output / "receipts" / (run["id"] + ".csv")
        if path.is_file():
            try:
                if file_hash(path) != receipt.get("csv_sha256"):
                    row["acceptance_issues"].append("receipt_csv_hash_mismatch")
                actual = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
                row["outer_score"] = score_csv(path, actual, task)
                row["csv_valid"] = True
            except (ValueError, TypeError, KeyError, OSError) as error:
                row["csv_error"] = str(error)
        scored.append(row)
    comparisons = {}
    for kind, entry in data["tasks"].items():
        task = TaskSpec.model_validate(entry["task"])
        policy = selection_for(task)
        def scores(arm):
            return [next((r["outer_score"] for r in scored if r["kind"] == kind and r["arm"] == arm and r["seed"] == seed and r["repeat"] == 0), None) for seed in (17, 42, 73)]
        comparisons[kind] = {arm: paired_win(scores("agent"), scores(arm), policy) for arm in ARMS if arm != "agent"}
    report = {"runs": scored, "comparisons": comparisons, "planned": len(registry),
        "local_quality_goal_achieved": all(r["accepted"] for tasks in comparisons.values() for r in tasks.values()) and all(r["status"] == "completed" and r["csv_valid"] and not r["acceptance_issues"] for r in scored),
        "official_threshold_exceeded": "unknown", "transferability": "not_established",
        "manifest_sha256": file_hash(output / "MANIFEST.json"), "metrics_scope": "development_public_data"}
    immutable(output / "OUTER_EVALUATION.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare"); p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("select"); p.add_argument("--task", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("freeze"); p.add_argument("--output", type=Path, required=True); p.add_argument("--selected", type=Path, required=True); p.add_argument("--candidate", type=Path, required=True)
    p = commands.add_parser("run-one"); p.add_argument("--task", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--arm", choices=[x for x in ARMS if x != "v008"], required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--seconds", type=float, default=10500); p.add_argument("--engine", choices=["flaml", "autogluon"])
    p = commands.add_parser("evaluate"); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": result = prepare(args.source, args.output)
    elif args.command == "select": result = select_automl(args.task, args.output)
    elif args.command == "freeze": result = freeze(args.output, read_json(args.selected), args.candidate)
    elif args.command == "run-one":
        if args.arm == "fixed_automl" and args.engine is None:
            parser.error("run-one --arm fixed_automl requires --engine flaml or --engine autogluon")
        if args.arm != "fixed_automl" and args.engine is not None:
            parser.error("run-one --engine is only valid with --arm fixed_automl")
        result = run_one(args.task, args.arm, args.seed, args.seconds, args.output, args.engine)
    else: result = evaluate(args.output)
    print(json.dumps({"command": args.command, "status": "completed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
