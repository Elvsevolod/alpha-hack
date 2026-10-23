"""Offline evidence runner. Usage: python -m context_compressor.verify --benchmark.

Runtime code has no third-party dependencies. Verification optionally uses the
organizer's jsonschema (schema oracle) and psutil (sampled resident memory).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import threading
import time
import tracemalloc
import unittest

from . import ContextCompressor, ContextStore, ProfileCache, ProfileKey
from .contracts import BUILDER_VERSION, canonical, digest
from .tests.helpers import emit, seed_history, setup, transport
from .tests.scenarios import four_attempt_fixture, pressure_scenario


def host():
    return {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
            "available_logical_cpus": os.cpu_count(), "cpu_limit_enforced": False, "ram_limit_enforced": False,
            "required_subsystem_profile": {"cpu": 4, "ram_gb": 10},
            "resource_profile_acceptance": "unconfirmed_without_a_fixed_resource_limited_host"}


def _measure(build):
    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        process = None
    baseline_rss = process.memory_info().rss if process else None
    peak_rss = [baseline_rss or 0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            stop.wait(0.001)

    sampler = threading.Thread(target=sample, daemon=True) if process else None
    tracemalloc.start()
    if sampler:
        sampler.start()
    try:
        started = time.perf_counter()
        prepared = build()
        elapsed = time.perf_counter() - started
        if process:
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        _, python_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        stop.set()
        if sampler:
            sampler.join(timeout=1)
    return {"seconds": elapsed, "python_peak_bytes": python_peak,
            "sampled_additional_rss_bytes": None if baseline_rss is None else max(0, peak_rss[0] - baseline_rss),
            "attempts": prepared.attempts, "stage": prepared.stage, "wire_bytes": prepared.wire_bytes,
            "request_sha256": prepared.request_hash}


def benchmarks(samples=30):
    if samples < 30:
        raise ValueError("The plan requires at least 30 warm measurements per case")
    cases = []
    with tempfile.TemporaryDirectory(prefix="context-benchmark-") as temporary:
        root = Path(temporary)
        for count in (100, 1000, 10000):
            with ContextStore(root / f"history-{count}.sqlite") as store:
                setup(store, cpu=4, memory_mib=9536)
                seed_history(store, count, detail_bytes=1024)
                emit(store, content="A permitted undelivered reference must remain literal.")
                compressor = ContextCompressor(store, transport(real=True))
                for mode in ("native", "json"):
                    for _ in range(3):
                        compressor.prepare(compressor.snapshot(), mode)
                    measurements = [_measure(lambda: compressor.prepare(compressor.snapshot(), mode)) for _ in range(samples)]
                    durations = sorted(item["seconds"] for item in measurements)
                    p95 = durations[math.ceil(0.95 * len(durations)) - 1]
                    python_peak = max(item["python_peak_bytes"] for item in measurements)
                    rss = [item["sampled_additional_rss_bytes"] for item in measurements if item["sampled_additional_rss_bytes"] is not None]
                    cases.append({"case": "history_growth", "experiment_records": count, "total_index_records": store.db.execute("SELECT count(*) FROM cc_records").fetchone()[0],
                                  "transport": mode, "warm_samples": samples,
                                  "median_seconds": statistics.median(durations), "p95_seconds": p95,
                                  "maximum_seconds": max(durations), "python_peak_bytes": python_peak,
                                  "sampled_additional_rss_bytes": max(rss) if rss else None,
                                  "observed_time_thresholds_pass": p95 <= 0.5 and max(durations) <= 2,
                                  "observed_memory_thresholds_pass": python_peak <= 64 * 1024 ** 2 and bool(rss) and max(rss) <= 64 * 1024 ** 2,
                                  "measurements": measurements})
        with ContextStore(root / "four-attempts.sqlite") as store:
            compressor, profile_calls = four_attempt_fixture(store)
            for _ in range(3):
                compressor.prepare(compressor.snapshot(), "native")
            measurements = [_measure(lambda: compressor.prepare(compressor.snapshot(), "native")) for _ in range(samples)]
            durations = sorted(item["seconds"] for item in measurements)
            p95 = durations[math.ceil(0.95 * samples) - 1]
            rss = [item["sampled_additional_rss_bytes"] for item in measurements if item["sampled_additional_rss_bytes"] is not None]
            python_peak = max(item["python_peak_bytes"] for item in measurements)
            cases.append({"case": "four_attempt_pressure", "experiment_records": 40,
                          "transport": "native", "warm_samples": samples,
                          "p95_seconds": p95, "maximum_seconds": max(durations),
                          "python_peak_bytes": python_peak, "sampled_additional_rss_bytes": max(rss) if rss else None,
                          "observed_time_thresholds_pass": p95 <= 0.5 and max(durations) <= 2,
                          "observed_memory_thresholds_pass": python_peak <= 64 * 1024 ** 2 and bool(rss) and max(rss) <= 64 * 1024 ** 2,
                          "all_builds_four_attempts": all(item["attempts"] == 4 for item in measurements),
                          "profile_calculator_calls": len(profile_calls), "measurements": measurements})
        with ContextStore(root / "profile.sqlite") as store:
            # Actual 256 x 10000 permitted input fixture, prepared before profiling.
            rows = [[None if (r + c) % 23 == 0 else (r * 17 + c) % 29 for c in range(10000)] for r in range(256)]
            calls = []

            def calculate():
                calls.append(1)
                return {"rows": {"fit": len(rows)}, "columns": {
                    f"feature_{index}": {"dtype": "int64", "missing_count": column.count(None)}
                    for index, column in enumerate(zip(*rows))}}

            cache = ProfileCache(store)
            key = ProfileKey("wide-256x10000-v1", digest(list(range(256))))
            started = time.perf_counter()
            reference = cache.get_or_create(key, calculate)
            cold_seconds = time.perf_counter() - started
            for _ in range(4):
                assert cache.get_or_create(key, calculate) == reference
            profile = {"rows": 256, "features": 10000, "cold_seconds_including_durable_cache": cold_seconds,
                       "calculator_calls": len(calls), "cache_hits": 4,
                       "observed_threshold_pass": cold_seconds <= 30 and len(calls) == 1,
                       "limitation": "reference fixture calculator; integrated dataset profiler must be measured separately"}
    return {"status": "preliminary_current_host_measurements", "host": host(), "cases": cases, "cold_profile": profile,
            "rss_measurement": "1 ms sampling plus Python tracemalloc; sampled RSS is not a hard peak guarantee",
            "warm_build_fraction_of_complete_run": None,
            "two_percent_full_run_gate": "pending_integrated_run_including_cold_preparation_search_and_extra_reads"}


def source_manifest():
    root = Path(__file__).parent
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()
            and "__pycache__" not in path.parts and "reports" not in path.parts and "release" not in path.parts}


def main():
    parser = argparse.ArgumentParser(description="Verify the independent compressor offline")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "reports" / "latest")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"), top_level_dir=str(Path(__file__).parent.parent))
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (args.output / "unit-tests.txt").write_text(log.getvalue(), encoding="utf-8")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "builder_version": BUILDER_VERSION,
              "scope": "standalone_offline_not_integrated_acceptance", "host": host(),
              "tests": {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                        "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
                        "successful": result.wasSuccessful()},
              "live_api_calls": 0, "server_token_savings": "unconfirmed",
              "final_B0_B1_N_comparison": "pending_finished_common_B0_and_integration",
              "integrated_6cpu_5gib_parallel_three_hour_rehearsal": "pending_integration",
              "source_sha256": source_manifest()}
    if result.wasSuccessful():
        # Fresh directories keep independent verifier invocations reproducible.
        with tempfile.TemporaryDirectory(prefix="context-pressure-") as temporary:
            report["pressure"] = [pressure_scenario(Path(temporary) / mode, mode) for mode in ("native", "json")]
        with tempfile.TemporaryDirectory(prefix="context-preflight-") as temporary:
            with ContextStore(Path(temporary) / "state.sqlite") as store:
                from .tests.helpers import task
                setup(store, task=task(wide=True))
                emit(store, content="диагностика " + "я" * 32768, kind="diagnostic", scope="code")
                compressor = ContextCompressor(store, transport(real=True))
                report["wide_preflight"] = compressor.preflight(compressor.snapshot())
        if args.benchmark:
            report["performance"] = benchmarks()
    (args.output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({"tests": report["tests"], "report": str((args.output / "verification.json").resolve()),
                     "performance_cases": len(report.get("performance", {}).get("cases", [])), "live_api_calls": 0}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
