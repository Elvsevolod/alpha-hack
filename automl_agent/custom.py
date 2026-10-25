from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pandas as pd

from .contracts import TrainingSummary
from .pipelines import DirectPipeline
from .util import digest

ALLOWED_IMPORTS = {
    "numpy", "pandas", "scipy", "sklearn", "catboost", "lightgbm", "xgboost", "feature_engine",
    "imblearn", "category_encoders", "statsmodels", "polars", "pyarrow",
    "math", "statistics", "collections", "itertools", "functools", "operator", "typing", "dataclasses",
    "re", "json", "datetime", "time", "warnings", "copy", "numbers", "random",
}
ALLOWED_AGENT_MODULES = {"automl_agent.api", "automl_agent.contracts", "automl_agent.pipelines", "automl_agent.preprocessing"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "input", "__import__", "globals", "locals", "vars", "breakpoint", "getattr", "setattr", "delattr"}
FORBIDDEN_MEMBERS = {"datasets", "hub", "ctypes", "ctypeslib", "ffi", "cdll", "CDLL", "LoadLibrary", "load_library",
    "from_pretrained", "load_state_dict", "read_csv", "read_parquet", "read_pickle", "to_pickle", "to_csv",
    "download", "download_url_to_file", "system", "popen", "exec", "compile",
    "import_module", "import_optional_dependency", "importlib", "modules", "load_module", "exec_module",
    "spec_from_file_location", "module_from_spec", "find_spec", "reload", "attrgetter", "methodcaller", "setprofile", "settrace", "addaudithook"}


def validate_code(code: str, component: str) -> None:
    try:
        tree = ast.parse(code, filename="candidate.py")
        compile(tree, "candidate.py", "exec")
    except SyntaxError as error:
        raise ValueError(f"Candidate Python syntax: {error.msg} at line {error.lineno}, column {error.offset}") from None
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for module in modules:
                if module == "sklearn.neural_network" or module.startswith("sklearn.neural_network."):
                    raise ValueError("Neural models are outside classical ML scope")
                if any(part in {"datasets", "hub", "cpp_extension"} for part in module.split(".")):
                    raise ValueError("Bundled datasets, model hubs and extension builders are outside the data policy")
                if any(part.startswith("_") for part in module.split(".")):
                    raise ValueError("Generated code must use public library interfaces")
                if module.startswith("automl_agent"):
                    if module not in ALLOWED_AGENT_MODULES:
                        raise ValueError("Generated code may import only the public candidate interface")
                elif module.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ValueError(f"Import is outside the candidate policy: {module}")
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("Relative imports are unsupported in standalone candidate modules")
            if isinstance(node, ast.ImportFrom) and any(alias.name in FORBIDDEN_MEMBERS or alias.name.startswith("import_") or alias.name.startswith("_") or alias.name in {"load", "loads"} and node.module != "json" for alias in node.names):
                raise ValueError("Candidate import exposes a disallowed data-loading or execution operation")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise ValueError(f"Forbidden candidate operation: {node.func.id}")
        if isinstance(node, ast.ImportFrom) and node.module == "sklearn" and any(a.name == "neural_network" for a in node.names):
            raise ValueError("Neural models are outside classical ML scope")
        if isinstance(node, ast.Attribute):
            if node.attr == "neural_network":
                raise ValueError("Neural models are outside classical ML scope")
            init = node.attr == "__init__" and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "super"
            json_decode = node.attr == "loads" and isinstance(node.value, ast.Name) and node.value.id == "json"
            private_library = node.attr.startswith("_") and not (isinstance(node.value, ast.Name) and node.value.id == "self") and not init
            if node.attr == "eval":
                parent = parents.get(node)
                if not (isinstance(parent, ast.Call) and parent.func is node and not parent.args and not parent.keywords):
                    raise ValueError("Only a no-argument evaluation-mode switch is allowed; dynamic expression evaluation is disabled")
            if private_library or (node.attr.startswith("__") and not init) or node.attr in FORBIDDEN_MEMBERS or node.attr.startswith(("fetch_", "import_")) or node.attr in {"load", "loads"} and not json_decode:
                raise ValueError(f"Forbidden candidate attribute: {node.attr}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and any(token in node.value.lower() for token in ("https://", "http://", "api_key", ".env", "/proc/", "../")):
            raise ValueError("Candidate contains a network, credential, or filesystem reference")
    entry = {"pipeline": "build_pipeline", "transformer": "build_transformer", "analysis": "analyze"}[component]
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry for node in tree.body):
        raise ValueError(f"Candidate must define {entry}")


def semantic_code_hash(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:]
    return digest(ast.dump(tree, include_attributes=False))


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TransformerPipeline:
    def __init__(self, transformer, spec, task):
        self.transformer, self.spec, self.task = transformer, spec, task

    def _check(self, X, transformed):
        if not isinstance(transformed, pd.DataFrame):
            raise ValueError("A custom transformer must return a DataFrame with original row indices")
        if not transformed.index.equals(X.index):
            raise ValueError("A custom transformer must preserve row identity and order")
        if set(self.task.targets) & set(transformed.columns):
            raise ValueError("A custom transform cannot expose target-named features")
        return transformed

    def fit(self, X, y, context):
        import copy
        if self.spec.supervised_transform:
            from .research_evaluation import folds
            parts, fold_evidence = [], []
            for number, (a, b) in enumerate(folds(X, y, self.task, context["strategy"], self.spec.seed)):
                transform = copy.deepcopy(self.transformer)
                if hasattr(transform, "fit_transform"):
                    transform.fit_transform(X.iloc[a].copy(), y.iloc[a].copy())
                else:
                    transform.fit(X.iloc[a].copy(), y.iloc[a].copy())
                parts.append(self._check(X.iloc[b], transform.transform(X.iloc[b].copy())))
                fold_evidence.append({"fold": number, "fit_keys": X.index[a].tolist(), "heldout_keys": X.index[b].tolist()})
            transformed = pd.concat(parts).reindex(X.index)
            covered = X.index.isin(pd.concat(parts).index)
            transformed = transformed.loc[covered]
            if not transformed.columns.is_unique or not len(transformed):
                raise ValueError("Cross-fitted transformer has no valid training matrix")
            if hasattr(self.transformer, "fit_transform"):
                self.transformer.fit_transform(X.copy(), y.copy())
            else:
                self.transformer.fit(X.copy(), y.copy())
            self.crossfit = {"scope": "fit_only_external_folds", "folds": fold_evidence,
                "covered_rows": len(transformed), "uncovered_initial_temporal_rows": int((~covered).sum())}
        else:
            if hasattr(self.transformer, "fit_transform"):
                transformed = self._check(X, self.transformer.fit_transform(X.copy()))
            else:
                self.transformer.fit(X.copy())
                transformed = self._check(X, self.transformer.transform(X.copy()))
            self.crossfit = {"scope": "unsupervised_no_labels"}
        categorical = [c for c in transformed if not pd.api.types.is_numeric_dtype(transformed[c]) and c not in self.task.text_columns]
        adapted = self.task.model_copy(update={"feature_columns": list(transformed), "categorical_columns": categorical,
            "text_columns": [c for c in self.task.text_columns if c in transformed]})
        direct = self.spec.model_copy(update={"engine": "direct", "custom_module": None, "custom_hash": None})
        self.pipeline = DirectPipeline(direct, adapted)
        summary = self.pipeline.fit(transformed, y.loc[transformed.index], context)
        summary.rows = len(X)
        summary.preprocessing = {**(summary.preprocessing or {}), "custom_transformer": self.crossfit}
        return summary

    def predict(self, X, context):
        return self.pipeline.predict(self._check(X, self.transformer.transform(X.copy())), context)

    def frozen_spec(self):
        frozen = self.pipeline.frozen_spec()
        return self.spec.model_copy(update={"iteration_schedule": frozen.iteration_schedule,
            "learned_decision": frozen.learned_decision})
