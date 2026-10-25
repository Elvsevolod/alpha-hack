"""Offline resource pilots on supplied CSVs; no models or external requests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASKS = {
    "regression": ("Регрессия", "regression/candidate/knn/request.json"),
    "classification": ("Многоклассовая классификация", "classification/candidate/autogluon_portfolio/request.json"),
}


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)


def worker(args):
    import pandas as pd
    from automl_agent.contracts import TaskSpec
    from automl_agent.data import Dataset
    from automl_agent.profiling import correlations
    from automl_agent.executor import stage_frame
    from automl_agent.util import canonical
    folder, spec = TASKS[args.task]
    directory = ROOT / "data_for_participants" / folder
    task = TaskSpec.model_validate(json.loads((ROOT / "outputs/plan_v081_implementation/development_cycle_04" / spec).read_text())["task"])
    frame = pd.read_csv(directory / "train.csv")
    base = peak_mib()
    started = time.monotonic()
    result = {"task": args.task, "mechanism": args.mechanism, "variant": args.variant, "shape": list(frame.shape)}
    if args.mechanism == "profile":
        dataset = object.__new__(Dataset)
        dataset.train, dataset.test, dataset.task = frame, pd.read_csv(directory / "test.csv"), task
        dataset.source_hash, dataset.profile_mode = "supplied_immutable_csv:" + args.task, args.variant
        indices = list(range(len(frame)))
        started = time.monotonic()
        profile = dataset.profile(indices, protocol_hash="fixed_resource_pilot")
        result["wall_seconds"] = time.monotonic() - started
        result["peak_rss_mib"] = peak_mib()
        started = time.monotonic()
        repeated = dataset.profile(indices, protocol_hash="fixed_resource_pilot")
        result["repeat_seconds"] = time.monotonic() - started
        assert profile == repeated and profile.status == "available"
        fields = ("dtype", "missing", "missing_fraction", "test_missing_fraction", "unique", "numeric", "min", "max", "median")
        facts = {name: {key: value[key] for key in fields if key in value} for name, value in profile.columns.items()}
        result.update(basic_facts_sha256=hashlib.sha256(canonical(facts).encode()).hexdigest(),
            artifact_bytes=len(canonical(profile.model_dump(mode="json")).encode()), findings=len(profile.findings),
            cache=getattr(dataset, "profile_cache_stats", {}), target_scope=profile.scope)
    elif args.mechanism == "stage":
        path = args.output / "frame.json"
        stage_frame(path, frame, streaming=args.variant == "streaming", deadline=time.time() + 120, memory_mb=2048)
        result.update(wall_seconds=time.monotonic() - started, peak_rss_mib=peak_mib(), artifact_bytes=path.stat().st_size)
        # Verify separately from the measured writing phase.
        from automl_agent.worker import read_frame
        restored = read_frame(path)
        pd.testing.assert_frame_equal(restored, frame.set_axis(list(map(str, frame.index))), check_dtype=False)
        result["roundtrip_equal"] = True
        path.unlink()
    else:
        columns = [name for name in task.feature_columns if pd.api.types.is_numeric_dtype(frame[name])][:int(args.variant)]
        selected = frame[columns].sample(n=min(20_000, len(frame)), random_state=42)
        started = time.monotonic()
        report = correlations(selected)
        result.update(wall_seconds=time.monotonic() - started, peak_rss_mib=peak_mib(), rows=len(selected), columns=len(columns),
            artifact_bytes=len(canonical(report).encode()), insufficient_pairs=sum(value == "insufficient_support"
                for row in report["pair_status"].values() for value in row.values()), targets_used=False)
    result["peak_rss_before_mib"] = base
    print(json.dumps(result))


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [ROOT / "data_for_participants" / folder / name for folder, _ in TASKS.values() for name in ("train.csv", "test.csv")]
    inputs += [ROOT / "outputs/plan_v081_implementation/development_cycle_04" / spec for _, spec in TASKS.values()]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    protocol = {"profile_wall_limit_seconds": 20, "profile_extra_rss_limit_mib": 256,
        "profile_cache_speedup_min": 10, "streaming_time_ratio_max": 1.05,
        "streaming_requires_lower_peak_rss": True, "staging_repetitions": 3, "correlation_wall_limit_seconds": 20,
        "source_branch_commit": "29197532431a0a395e3de07b8bed1f8bc7637ebc", "qwen_calls": 0,
        "scope": "All provided train rows for development-only resource profiling; no test labels or quality estimates.",
        "rss_scope": "Process high-water RSS includes CSV loading; read-back validation excluded from the staging measurement.",
        "host_other_workload_controlled": False, "inputs_sha256": hashes,
        "source_files_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("automl_agent/profiling.py", "automl_agent/data.py", "automl_agent/executor.py", "tools/eda_transfer_benchmark.py")}}
    (args.output / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2))
    rows = []
    for task in TASKS:
        cases = [("profile", mode) for mode in ("legacy", "unified")]
        for repeat in range(3):
            cases += [("stage", mode) for mode in (("bulk", "streaming") if repeat % 2 == 0 else ("streaming", "bulk"))]
        cases += [("correlations", str(columns)) for columns in (10, 30)]
        for index, (mechanism, variant) in enumerate(cases):
            directory = args.output / f"{task}_{index}_{mechanism}_{variant}"
            directory.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--task", task,
                "--mechanism", mechanism, "--variant", variant, "--output", str(directory)]
            env = {**os.environ, "OPENAI_API_KEY": "offline-test", "OPENAI_BASE_URL": "http://127.0.0.1:1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
            (directory / "worker.log").write_text(process.stdout + process.stderr)
            if process.returncode:
                raise RuntimeError(f"Resource pilot failed: {directory}")
            result = json.loads(process.stdout.splitlines()[-1])
            (directory / "RESULT.json").write_text(json.dumps(result, indent=2))
            rows.append(result)
            print(json.dumps(result), flush=True)
    decisions = {}
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task]
        old, new = [next(row for row in selected if row["mechanism"] == "profile" and row["variant"] == mode) for mode in ("legacy", "unified")]
        profile_ok = (old["basic_facts_sha256"] == new["basic_facts_sha256"] and new["wall_seconds"] <= 20
            and new["peak_rss_mib"] <= old["peak_rss_mib"] + 256
            and new["repeat_seconds"] * 10 < new["wall_seconds"] and new["cache"]["hits"] == 1)
        bulk, stream = [[row for row in selected if row["mechanism"] == "stage" and row["variant"] == mode] for mode in ("bulk", "streaming")]
        ratio = statistics.median(row["wall_seconds"] for row in stream) / statistics.median(row["wall_seconds"] for row in bulk)
        rss_ratio = statistics.median(row["peak_rss_mib"] for row in stream) / statistics.median(row["peak_rss_mib"] for row in bulk)
        decisions[task] = {"profile_within_resource_gate": profile_ok, "streaming_time_ratio": ratio,
            "streaming_peak_rss_ratio": rss_ratio, "streaming_gate_passed": ratio <= 1.05 and rss_ratio < 1,
            "correlations_within_budget": all(row["wall_seconds"] <= 20 for row in selected if row["mechanism"] == "correlations")}
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha for name, sha in hashes.items()), "Pilot inputs changed"
    report = {"protocol": protocol, "results": rows, "decisions": decisions,
        "enable_unified_profile_for_new_runs": all(row["profile_within_resource_gate"] for row in decisions.values()),
        "streaming_eligible_for_default": all(row["streaming_gate_passed"] for row in decisions.values()),
        "predictive_quality": "unestablished"}
    (args.output / "REPORT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"decisions": decisions, "profile_default_eligible": report["enable_unified_profile_for_new_runs"],
        "streaming_default_eligible": report["streaming_eligible_for_default"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--mechanism", choices=("profile", "stage", "correlations"))
    parser.add_argument("--variant")
    options = parser.parse_args()
    worker(options) if options.worker else main(options)
