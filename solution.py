"""Run: OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... python solution.py task.md"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main():
    started = globals().get("_AGENT_ENTRY_STARTED_AT", time.time())
    from dotenv import load_dotenv
    # Runner-provided values take precedence over the optional local configuration.
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    parser = argparse.ArgumentParser(description="Qwen autonomous CSV ML agent")
    parser.add_argument("task", nargs="?", default="task.md", type=Path)
    parser.add_argument("--time-limit", type=float, default=170 * 60)
    parser.add_argument("--memory-mb", type=float, default=28000)
    parser.add_argument("--cpu", type=int, default=7)
    parser.add_argument("--max-candidates", type=int, help="Optional development limit; platform runs use the shared time budget")
    parser.add_argument("--max-calls", type=int, help="Optional development limit; platform runs use the shared time budget")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transport", choices=["auto", "native", "json"], default="auto")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--clustering-contract", type=Path, help="Explicit target-free numeric clustering contract; requires --no-llm")
    parser.add_argument("--research-focus", help="Optional development objective; never changes task/metric or evaluation permissions")
    args = parser.parse_args()
    if args.time_limit <= 0 or args.memory_mb < 128 or args.cpu < 1 or any(value is not None and value < 0 for value in (args.max_candidates, args.max_calls)):
        parser.error("Invalid resource limit")
    from automl_agent.budget import Limits
    from automl_agent.controller import Controller
    limits = Limits(seconds=min(args.time_limit, 180 * 60), memory_mb=min(args.memory_mb, 28000),
        cpu=min(args.cpu, 7), max_candidates=args.max_candidates, max_calls=args.max_calls)
    if args.clustering_contract:
        if not args.no_llm:
            parser.error("The explicit clustering route requires --no-llm")
        from automl_agent.clustering import execute
        report = execute(args.task.parent.resolve(), args.clustering_contract, limits, started, args.run_id)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["state"] == "completed" else 1
    if not args.no_llm:
        from automl_agent.clustering_route import resolve_and_run
        try:
            clustering = resolve_and_run(args.task.resolve(), limits, started, args.run_id)
        except Exception as error:  # fail open: the supervised controller still owns a valid task
            print(json.dumps({"state": "clustering_route_failed", "error": type(error).__name__}, ensure_ascii=False))
            clustering = None
        if clustering is not None:
            print(json.dumps(clustering, ensure_ascii=False))
            return 0 if clustering["state"] == "completed" else 1
    controller = Controller(args.task, limits, started, args.run_id, args.transport, args.research_focus, args.seed)
    try:
        report = controller.run(args.no_llm)
        print(json.dumps({"state": report["state"], "output": str(controller.publisher.output) if controller.publisher else None,
            "control": report["control_status"], "refit": report["refit_status"], "failure": report["failure"]}, ensure_ascii=False))
        return 0 if report["state"] == "completed" else 1
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
