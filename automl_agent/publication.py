from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import align_predictions
from .util import atomic_bytes, atomic_json, file_hash, read_json


class Publisher:
    def __init__(self, dataset, plan, workdir, journal):
        self.dataset, self.plan, self.workdir, self.journal = dataset, plan, workdir, journal
        self.output = dataset.task_path.parent / "submission.csv"

    def frame(self, bundle):
        task = self.dataset.task
        if task.metric.requires_probabilities and task.output.prediction_type != "probabilities":
            raise ValueError("Probability metrics require probability output")
        predicted = align_predictions(bundle, self.dataset.test_keys, task, self.plan.hash, "test")
        result = pd.DataFrame(index=range(len(self.dataset.test)))
        for column in task.id_columns:
            result[column] = self.dataset.test[column].to_numpy()
        if task.output.prediction_type == "probabilities":
            for column, (target, label) in task.output.probability_columns.items():
                result[column] = predicted[target][:, task.classes[target].index(label)]
        else:
            for target in task.targets:
                values = predicted[target]
                if bundle.prediction_type == "probabilities":
                    values = np.asarray(task.classes[target])[np.argmax(values, axis=1)]
                result[target] = values
        if set(result) != set(task.output.columns):
            raise ValueError("Output columns disagree with confirmed schema")
        result = result[task.output.columns]
        if result.isna().any().any():
            raise ValueError("Output contains missing values")
        return result

    def prepare(self, bundle, experiment_id, scope, dependencies=(), details=None):
        frame = self.frame(bundle)
        content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        # Read the exact bytes back with identifier preservation before committing.
        restored = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
        if list(restored) != self.dataset.task.output.columns or len(restored) != len(frame):
            raise ValueError("CSV round-trip schema mismatch")
        for column in self.dataset.task.id_columns:
            if restored[column].tolist() != self.dataset.test[column].astype(str).tolist():
                raise ValueError("CSV round-trip changed identifier spelling or order")
        sha = hashlib.sha256(content).hexdigest()
        provenance = {"task_hash": self.dataset.task.hash, "data_version": self.dataset.source_hash,
            "validation_hash": self.plan.hash, "experiment_id": experiment_id, "scope": scope,
            "dependencies": list(dependencies), "csv_sha256": sha, "rows": len(frame),
            "columns": list(frame), "details": details or {}}
        artifact = self.workdir / "publications" / f"{sha}.csv"
        atomic_bytes(artifact, content)
        return {**provenance, "artifact": str(artifact)}

    def publish(self, bundle, experiment_id, scope, dependencies=(), details=None):
        pending = self.prepare(bundle, experiment_id, scope, dependencies, details)
        self.journal.put("publication_pending", pending)
        self.recover()
        return pending

    def recover(self):
        pending = self.journal.get("publication_pending")
        if pending:
            artifact = Path(pending["artifact"])
            if not artifact.is_file() or file_hash(artifact) != pending["csv_sha256"]:
                raise ValueError("Pending publication artifact is missing or corrupt")
            if not self.output.is_file() or file_hash(self.output) != pending["csv_sha256"]:
                atomic_bytes(self.output, artifact.read_bytes(), mode=0o644)
            self.journal.put_many({"publication": pending, "publication_pending": None},
                event=("publication_completed", pending))
            atomic_json(self.workdir / "submission_provenance.json", pending)
        saved = self.journal.get("publication")
        if saved:
            artifact = Path(saved["artifact"])
            if not artifact.is_file() or file_hash(artifact) != saved["csv_sha256"]:
                raise ValueError("Published artifact is missing or corrupt")
            if not self.output.is_file() or file_hash(self.output) != saved["csv_sha256"]:
                atomic_bytes(self.output, artifact.read_bytes(), mode=0o644)
            self.output.chmod(0o644)
            atomic_json(self.workdir / "submission_provenance.json", saved)


def constant_bundle(dataset, plan):
    from .api import prediction_bundle
    task = dataset.task
    X = dataset.features(test=True)
    X.index = dataset.test_keys
    y = dataset.train.iloc[plan.fit][task.targets]
    predicted = {}
    for target in task.targets:
        if task.classes:
            classes = task.classes[target]
            counts = y[target].value_counts()
            prior = np.asarray([float(counts.get(c, 0)) / max(1, len(y)) for c in classes])
            predicted[target] = np.tile(prior, (len(X), 1))
        else:
            predicted[target] = np.full(len(X), float(y[target].mean()))
    return prediction_bundle(predicted, X, {"task": task.model_dump(mode="json"), "protocol_hash": plan.hash, "split_id": "test"},
        "probabilities" if task.classes else "scores" if task.kind == "ranking" else "values")
