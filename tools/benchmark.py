"""Entry point for the frozen v010 comparative protocol.

The previous development runner is preserved in outputs/v010_implementation/before/tools/benchmark.py.
Use prepare, select, freeze, run-one and evaluate; outer labels open only after
all registered runs finish or receive explicit failure statuses.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.comparison_protocol import main

if __name__ == "__main__":
    main()
