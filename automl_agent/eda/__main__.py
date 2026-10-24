"""Run only offline data reconnaissance: python -m automl_agent.eda --help."""
import argparse
import json
from pathlib import Path

from .runner import run


def main():
    parser = argparse.ArgumentParser(description="Data reconnaissance: offline reports, or bounded autonomous organizer Qwen exploration with --agent")
    parser.add_argument("task", type=Path)
    parser.add_argument("--agent", action="store_true", help="Autonomous organizer Qwen mode; <=7 generations, <=2 fits")
    parser.add_argument("--policy", choices=["free","evidence","alternatives"], help="Agent decision context; default evidence, fixed on resume")
    parser.add_argument("--stop-units", type=float, default=5, help="Gateway stop threshold between requests; not a hard tariff reservation")
    parser.add_argument("--task-spec", type=Path, help="Explicit task contract; defaults to task_spec.json beside task.md")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--correlations", nargs="+", help="At most 30 explicit columns, including optional numeric targets")
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--memory-mb", type=float, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.agent:
            if args.task_spec or args.correlations:
                raise ValueError("--agent resolves its contract and chooses analyses; --task-spec/--correlations are offline-only")
            from tools.qwen_eda_live import run_single
            result = run_single(args.task, args.output, cap=args.stop_units, policy=args.policy or "evidence", seconds=args.time_limit if args.time_limit is not None else 600)
        else:
            if args.policy:
                raise ValueError("--policy requires --agent")
            result = run(args.task, args.output, args.task_spec, args.correlations,
                args.time_limit if args.time_limit is not None else 120, args.cpu, args.memory_mb, args.seed)
    except (ValueError, OSError) as error:
        parser.exit(2, f"EDA failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
