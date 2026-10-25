"""Prevent generated Python from importing or directly using unlisted packages."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

from .util import atomic_json

# Import namespaces of the unchanged 37 organizer distributions. Internal library
# calls may use transitive dependencies; the candidate may not obtain their modules.
PERMITTED_NAMESPACES = {
    "numpy", "scipy", "pandas", "polars", "pyarrow", "joblib", "threadpoolctl", "sklearn", "catboost",
    "lightgbm", "xgboost", "imblearn", "category_encoders", "feature_engine", "statsmodels", "optuna",
    "flaml", "lightautoml", "openai", "httpx", "tenacity", "langchain", "langchain_openai", "langgraph",
    "pydantic", "dotenv", "yaml", "orjson", "jsonschema", "matplotlib", "seaborn", "tqdm", "psutil", "cloudpickle", "setuptools",
}


def permitted_module(name):
    if name == "sklearn.neural_network" or name.startswith("sklearn.neural_network."):
        return False
    if name == "candidate_module":
        return True
    if name.startswith("automl_agent."):
        return name in {"automl_agent.api", "automl_agent.contracts", "automl_agent.pipelines", "automl_agent.preprocessing"}
    if name == "autogluon" or name.startswith("autogluon."):
        return name == "autogluon.tabular" or name.startswith("autogluon.tabular.")
    top = name.split(".")[0]
    return top in sys.stdlib_module_names or top in PERMITTED_NAMESPACES or top == "builtins"


def install_dependency_guard(candidate_file: Path, directory: Path):
    filename = str(candidate_file.resolve())
    result_file = directory / "worker_result.json"

    def deny(module, reason):
        # A profile-hook exception can disable profiling and be caught by a
        # candidate. End this attempt after durably recording the policy failure.
        atomic_json(result_file, {"status": "unsupported_mode", "error": f"Candidate dependency policy: {reason}: {module}", "seconds": 0})
        os._exit(78)

    def profile(frame, event, argument):
        caller = frame.f_back
        candidate_caller = caller is not None and caller.f_code.co_filename == filename
        if event == "call" and candidate_caller:
            module = frame.f_globals.get("__name__", "")
            if module and not permitted_module(module):
                deny(module, "direct call into an unlisted dependency")
        elif event == "return" and candidate_caller and isinstance(argument, ModuleType):
            if not permitted_module(argument.__name__):
                deny(argument.__name__, "dynamic loader returned an unlisted module")
        elif event == "c_call" and frame.f_code.co_filename == filename:
            module = getattr(argument, "__module__", None)
            if module and not permitted_module(module):
                deny(module, "direct native call into an unlisted dependency")

    sys.setprofile(profile)
