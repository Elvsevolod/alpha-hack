from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import MetricSpec, OutputSpec, TaskSpec


@pytest.fixture(autouse=True)
def explicit_fixture_semantic_channel(monkeypatch):
    """Existing synthetic contracts use an offline semantic response, never DNS/API.

    Interpretation acceptance tests override this method with their own proposals.
    """
    from automl_agent.qwen import QwenClient
    original = QwenClient.resolve
    def resolve(client, system, request):
        path = client.journal.directory.parent.parent / 'task_spec.json'
        if not path.is_file():
            return original(client, system, request)
        spec = json.loads(path.read_text())
        spec.update(contract_version=2, source_hash=request['source_hash'], file_roles={
            role: role for role in ('train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv')
            if role in json.loads(request['sources']['inventory']['text'])})
        return {'request_id':request['request_id'], 'source_hash':request['source_hash'],
            'status':'proposed', 'task_spec':spec,
            'evidence':{field:['task','inventory'] for field in spec}}
    monkeypatch.setattr(QwenClient, 'resolve', resolve)
    original_critique = QwenClient.critique
    def critique(client, request, reserve_seconds):
        path = client.journal.directory.parent.parent / 'task_spec.json'
        if not path.is_file():
            return original_critique(client, request, reserve_seconds)
        return {'review_id': request['review_id'], 'verdict': 'approve',
            'evaluation_scope': 'research_validation',
            'comparator_id': next((item['experiment_id'] for item in request['comparators']
                if item['status'] == 'completed' and item['scope'] != 'pilot' and item['score'] is not None), None),
            'evidence_assessment': 'Offline fixture hypothesis.',
            'comparison_assessment': 'Fixed fixture protocol.',
            'leakage_assessment': 'Existing fit-only fixture execution.',
            'cost_assessment': 'Bounded offline test.',
            'falsification_assessment': 'Fixture metric and execution checks.', 'revision_requests': []}
    monkeypatch.setattr(QwenClient, 'critique', critique)


@pytest.fixture
def tabular():
    rng = np.random.default_rng(41)
    n = 180
    x = rng.normal(size=n)
    X = pd.DataFrame({"number": x, "second": rng.normal(size=n), "category": np.where(x > 0, "A", "B"),
        "empty": np.full(n, np.nan)}, index=[f"train:{i}" for i in range(n)])
    X.loc[X.index[::7], "number"] = np.nan
    y = pd.DataFrame({"answer": 2 * x + rng.normal(scale=0.1, size=n)}, index=X.index)
    task = TaskSpec(kind="regression", targets=["answer"], id_columns=["identifier"], feature_columns=list(X),
        categorical_columns=["category"], metric=MetricSpec(name="rmse"),
        output=OutputSpec(columns=["identifier", "answer"], prediction_type="values"), source_hash="data-v1")
    return X, y, task


def write_task(root: Path, kind="regression", probability=False, temporal=False, groups=False, multilabel=False):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    n, nt = 180, 25
    a = rng.normal(size=n + nt)
    frame = pd.DataFrame({"row": [f"{i:05d}" for i in range(n + nt)], "value": a,
        "empty": np.full(n + nt, np.nan), "code": np.where(a > 0, "NA", "-")})
    frame.loc[::11, "value"] = np.nan
    frame["answer"] = np.where(a > 0, "positive", "negative") if kind == "classification" else 2 * a
    metric = MetricSpec(name="log_loss" if probability else "f1" if kind == "classification" else "rmse")
    targets = ["answer"]
    if multilabel:
        kind = "multilabel"
        frame["answer"] = (a > 0).astype(int)
        frame["other"] = (a > 0.7).astype(int)
        targets.append("other")
        metric = MetricSpec(name="f1", average="macro")
    if temporal:
        frame["when"] = np.repeat(np.arange(1, 6), [45, 45, 45, 45, nt])
    if groups:
        frame["query"] = np.arange(n + nt) // 5
    train, test = frame.iloc[:n], frame.iloc[n:].drop(columns=targets)
    train.to_csv(root / "train.csv", index=False)
    test.to_csv(root / "test.csv", index=False)
    if probability:
        columns = ["row", "p_negative", "p_positive"]
        output = OutputSpec(columns=columns, prediction_type="probabilities", probability_columns={"p_negative": ("answer", "negative"), "p_positive": ("answer", "positive")})
    else:
        columns = ["row", *targets]
        output = OutputSpec(columns=columns, prediction_type="labels" if kind in {"classification", "multilabel"} else "values")
    sample = pd.DataFrame({c: test["row"].to_numpy() if c == "row" else np.zeros(nt) for c in columns})
    sample.to_csv(root / "sample_submission.csv", index=False)
    spec = TaskSpec(kind=kind, targets=targets, id_columns=["row"], feature_columns=[c for c in test if c != "row"],
        categorical_columns=["code"], metric=metric, output=output, source_hash="placeholder",
        time_column="when" if temporal else None, group_column="query" if groups else None)
    (root / "task_spec.json").write_text(json.dumps(spec.model_dump(mode="json", exclude={"source_hash"}), ensure_ascii=False))
    (root / "task.md").write_text("Synthetic technical CSV fixture. No external training data.\n")
    return root / "task.md"
