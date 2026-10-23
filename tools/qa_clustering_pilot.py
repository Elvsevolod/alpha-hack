"""Target-free clustering pilot, isolated from the frozen A/B runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ClusteringContract(BaseModel):
    model_config = ConfigDict(extra='forbid')
    version: Literal[1] = 1
    kind: Literal['clustering'] = 'clustering'
    train_file: str
    test_file: str
    features: list[str] = Field(min_length=1)
    id_columns: list[str] = Field(min_length=1)
    n_clusters: int = Field(ge=2)
    metric: Literal['silhouette'] = 'silhouette'
    seed: int = 42

    @model_validator(mode='after')
    def roles(self):
        if set(self.features) & set(self.id_columns):
            raise ValueError('Identifiers cannot be features')
        if len(set(self.features)) != len(self.features) or len(set(self.id_columns)) != len(self.id_columns):
            raise ValueError('Duplicate column role')
        if 'cluster' in self.id_columns or self.train_file == self.test_file:
            raise ValueError('Output or file roles overlap')
        for name in (self.train_file, self.test_file):
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Input must be a task-relative path')
        return self


def load_input(root, filename, spec):
    root = root.resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Input escapes task directory')
    frame = pd.read_csv(path)
    required = spec.features + spec.id_columns
    if not set(required) <= set(frame) or frame.empty:
        raise ValueError('Missing input columns or rows')
    if frame[spec.id_columns].isna().any().any() or frame[spec.id_columns].duplicated().any():
        raise ValueError('Identifiers must be present and unique within the file')
    features = frame[spec.features].apply(pd.to_numeric, errors='raise')
    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError('Infinite feature value')
    return frame, features


def fit_model(features, spec):
    if len(features) < spec.n_clusters:
        raise ValueError('Insufficient training rows for the declared cluster count')
    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                          StandardScaler(), KMeans(n_clusters=spec.n_clusters, n_init=10,
                                                   random_state=spec.seed))
    model.fit(features)
    return model


def silhouette_diagnostic(matrix, labels):
    labels = np.asarray(labels)
    if not 1 < len(np.unique(labels)) < len(labels):
        return {'status': 'undefined_silhouette', 'score': None,
                'reason': 'requires_2_to_n_minus_1_observed_clusters'}
    # Full-vector measurement is bounded; never silently substitute a sample.
    if len(labels) > 5000:
        return {'status': 'limited', 'score': None, 'reason': 'full_silhouette_row_budget_5000'}
    return {'status': 'completed', 'score': float(silhouette_score(matrix, labels, metric='euclidean'))}


def run(root, spec):
    train, features = load_input(root, spec.train_file, spec)
    test, future = load_input(root, spec.test_file, spec)
    if (root / spec.train_file).resolve() == (root / spec.test_file).resolve():
        raise ValueError('Train and test resolve to the same source')
    rows = np.random.default_rng(spec.seed).permutation(len(train))
    boundary = int(.8 * len(rows))
    research = fit_model(features.iloc[rows[:boundary]], spec)
    validation = features.iloc[rows[boundary:]]
    if validation.empty:
        raise ValueError('No validation rows')
    transformed = research[:-1].transform(validation)
    diagnostic = silhouette_diagnostic(transformed, research.predict(validation))
    # Freeze this explicit recipe, recreate every preprocessing/model component.
    final = fit_model(features, spec)
    assert final is not research
    predicted = final.predict(future)
    output = test[spec.id_columns].copy()
    output['cluster'] = predicted.astype(int)
    temporary = root / '.submission.clustering.pending'
    with temporary.open('x', newline='') as stream:
        output.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root / 'submission.csv')
    return {'contract': spec.model_dump(), 'validation': diagnostic,
            'publication': {'scope': 'full_train_refit', 'rows': len(output),
                            'sha256': hashlib.sha256((root / 'submission.csv').read_bytes()).hexdigest()},
            'research_fit_rows': boundary, 'final_fit_rows': len(train),
            'prediction_fits': 0, 'recipe_frozen_before_refit': True}


def pilot(output):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    points = np.concatenate([rng.normal(-5, .1, (80, 2)), rng.normal(5, .1, (80, 2))])
    train = pd.DataFrame(points, columns=['x', 'z'])
    train.insert(0, 'row_id', np.arange(len(train)))
    train.to_csv(output / 'observations.csv', index=False)
    test = pd.DataFrame({'row_id': [900, 700, 800], 'x': [-5., 5., -5.1], 'z': [-5., 5., -4.9]})
    test.to_csv(output / 'future.csv', index=False)
    spec = ClusteringContract(train_file='observations.csv', test_file='future.csv',
                              features=['x', 'z'], id_columns=['row_id'], n_clusters=2)
    result = run(output, spec)
    actual = pd.read_csv(output / 'submission.csv')
    assert list(actual) == ['row_id', 'cluster'] and list(actual.row_id) == [900, 700, 800]
    assert actual.cluster.iloc[0] == actual.cluster.iloc[2] != actual.cluster.iloc[1]
    assert result['validation']['status'] == 'completed' and result['validation']['score'] > .9
    golden = silhouette_diagnostic(np.array([[0.], [1.], [10.], [11.]]), [0, 0, 1, 1])
    assert np.isclose(golden['score'], 359 / 399, atol=1e-14)
    for labels in ([0, 0, 0], [0, 1, 2]):
        assert silhouette_diagnostic(np.arange(3)[:, None], labels)['status'] == 'undefined_silhouette'
    model = fit_model(train[['x', 'z']], spec)
    before = model[-1].cluster_centers_.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError('predict attempted to fit')
    model.fit = forbidden
    model[-1].fit = forbidden
    model.predict(test[['x', 'z']])
    assert np.array_equal(before, model[-1].cluster_centers_)
    for field, value in [('targets', ['fake_target']), ('train_file', '../outside.csv'), ('features', ['row_id'])]:
        try:
            ClusteringContract.model_validate({**spec.model_dump(), field: value})
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid clustering contract accepted: ' + field)
    result.update(passed=True, arithmetic_golden=golden, organizer_requests=0,
                  status='staged_target_free_pilot_not_enabled_in_A_or_B',
                  limitation='Numeric KMeans only; no general dataset or live semantic acceptance claim')
    (output / 'REPORT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = pilot(args.output.resolve())
    print(json.dumps({'passed': report['passed'], 'silhouette': report['validation']['score']}))
