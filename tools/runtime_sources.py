"""Source boundary for the offline competition bundle, separate from development clients."""
from pathlib import Path

DEVELOPMENT_ONLY = frozenset({'eda/agent.py', 'eda/deepseek.py', 'eda/__main__.py', 'eda/reporting.py'})


def production_agent_sources(root):
    package=Path(root)/'automl_agent'
    return [path for path in sorted(package.rglob('*.py')) if path.relative_to(package).as_posix() not in DEVELOPMENT_ONLY]
