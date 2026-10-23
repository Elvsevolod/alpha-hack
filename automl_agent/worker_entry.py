"""Trusted subprocess entry point; the environment carries no API credentials."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(os, "sched_setaffinity"):
    try:
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[:max(1, int(os.environ.get("AGENT_WORKER_CPUS", "1")))])
    except OSError:
        pass
from automl_agent.worker import main

main(sys.argv[1])
