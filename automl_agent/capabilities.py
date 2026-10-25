from __future__ import annotations

import importlib.metadata
import importlib.util
import re
from pathlib import Path

from .util import atomic_json, digest, read_json

MODEL_PACKAGES = {
    "ridge": "scikit-learn", "spline": "scikit-learn", "elastic_net": "scikit-learn", "logistic": "scikit-learn",
    "sgd": "scikit-learn", "knn": "scikit-learn", "constant": "scikit-learn", "random_forest": "scikit-learn", "extra_trees": "scikit-learn",
    "hist_gradient_boosting": "scikit-learn", "catboost": "catboost", "lightgbm": "lightgbm",
    "lgbm_ranker": "lightgbm", "xgboost": "xgboost", "naive": "scikit-learn", "sarimax": "statsmodels", "exponential_smoothing": "statsmodels",
}


def installed_versions():
    names = {*MODEL_PACKAGES.values(), "numpy", "pandas", "scipy", "pydantic", "psutil", "feature-engine", "flaml", "optuna", "autogluon.tabular"}
    root = Path(__file__).resolve().parent.parent
    path = root / "requirements.txt"
    if not path.is_file():path = root / "data_for_participants/requirements.txt"
    if path.is_file():
        names.update(match[1] for line in path.read_text().splitlines()
                     if (match := re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s#]+)", line.strip())))
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


class CapabilityRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.versions = installed_versions()
        root = Path(__file__).resolve().parent.parent
        requirements = root / "requirements.txt"
        if not requirements.is_file():
            requirements = root / "data_for_participants/requirements.txt"
        self.expected = {}
        if requirements.is_file():
            for line in requirements.read_text().splitlines():
                match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s#]+)", line.strip())
                if match:
                    self.expected[match[1]] = match[2]
        self.records = read_json(path) if path.is_file() else {}

    def key(self, spec, task, strategy):
        package = spec.engine if spec.engine in {"flaml", "optuna", "lightautoml"} else "autogluon.tabular" if spec.engine == "autogluon" else MODEL_PACKAGES.get(spec.model, "custom")
        return digest({"package": package, "version": self.versions.get(package), "task": task.kind,
            "environment_hash": digest(self.versions),
            "interface": {"custom": spec.custom_hash, "transformations": [op.name for op in spec.transformations],
                "resampling": spec.resampling, "early_stopping": bool(spec.early_stopping_rounds),
                "autogluon_profile": spec.autogluon_profile},
            "model": spec.model, "engine": spec.engine, "representation": spec.representation,
            "imputation": spec.imputation.model_dump(), "strategy": strategy,
            "output": task.output.prediction_type, "postprocess": spec.prediction_policy})

    def require_available(self, spec):
        package = "autogluon.tabular" if spec.engine == "autogluon" else spec.engine if spec.engine in {"flaml", "optuna", "lightautoml"} else MODEL_PACKAGES.get(spec.model)
        if package is None and spec.engine not in {"custom", "ensemble"}:
            raise ValueError(f"Unknown model: {spec.model}")
        if package and self.versions.get(package, "absent") == "absent":
            raise ImportError(f"Required package is unavailable: {package}")
        if package in self.expected and self.versions[package] != self.expected[package]:
            raise ImportError(f"Unverified package version: {package} {self.versions[package]}, expected {self.expected[package]}")

    def record(self, spec, task, strategy, status, probe_id, reason=None):
        self.records[self.key(spec, task, strategy)] = {"status": status, "probe_id": probe_id,
            "reason": reason, "fit_predict_save_load": status == "verified", "refit": "unverified",
            "model": spec.model, "engine": spec.engine, "task": task.kind, "strategy": strategy,
            "versions": self.versions}
        atomic_json(self.path, self.records)

    def verified(self, spec, task, strategy):
        return self.records.get(self.key(spec, task, strategy), {}).get("status") == "verified"

    def summary(self):
        grouped = {}
        for record in self.records.values():
            key = (record.get("model"), record.get("engine"), record.get("task"), record.get("strategy"), record.get("status"), record.get("refit"))
            group = grouped.setdefault(key, {field: record.get(field) for field in ("model", "engine", "task", "strategy", "status", "refit")})
            group["count"] = group.get("count", 0) + 1
            group["last_probe_id"] = record.get("probe_id")
        return {"versions": self.versions, "verified_modes": list(grouped.values()),
            "version_mismatches": {name: {"actual": actual, "required": self.expected[name]} for name, actual in self.versions.items() if name in self.expected and actual != self.expected[name]},
            "available_families": [name for name, package in MODEL_PACKAGES.items() if self.versions.get(package) != "absent"]}
