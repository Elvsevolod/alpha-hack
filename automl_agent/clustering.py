"""Minimal numeric clustering contract, separate from supervised targets."""
from __future__ import annotations

import hashlib
import json
import fcntl
import re
import shutil
import time
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
    n_clusters_candidates: list[int] = Field(default_factory=list, max_length=8)

    @model_validator(mode='after')
    def roles(self):
        if set(self.features) & set(self.id_columns):
            raise ValueError('Identifiers cannot be features')
        if len(set(self.features)) != len(self.features) or len(set(self.id_columns)) != len(self.id_columns):
            raise ValueError('Duplicate column role')
        if 'cluster' in self.id_columns or self.train_file == self.test_file:
            raise ValueError('Output or file roles overlap')
        for name in (self.train_file, self.test_file):
            if not Path(name).parts or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Input must be a task-relative path')
            if Path(name).parts[0] == 'outputs' or Path(name).as_posix() in {'submission.csv', 'request.json', 'worker_result.json', 'run.lock'}:
                raise ValueError('Input overlaps reserved output or worker paths')
        if any(value < 2 for value in self.n_clusters_candidates):
            raise ValueError('Cluster counts must be at least two')
        return self


def load_input(root, filename, spec):
    root = root.resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Input escapes task directory')
    frame = pd.read_csv(path, dtype={column: str for column in spec.id_columns}, keep_default_na=False)
    required = spec.features + spec.id_columns
    if not set(required) <= set(frame) or frame.empty:
        raise ValueError('Missing input columns or rows')
    if (frame[spec.id_columns].isna().any().any() or frame[spec.id_columns].eq('').any().any()) or frame[spec.id_columns].duplicated().any():
        raise ValueError('Identifiers must be present and unique within the file')
    features = frame[spec.features].apply(lambda column: pd.to_numeric(column.mask(column.eq('')), errors='raise'))
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
    if (root / spec.train_file).samefile(root / spec.test_file):
        raise ValueError('Train and test resolve to the same source')
    rows = np.random.default_rng(spec.seed).permutation(len(train))
    boundary = int(.8 * len(rows))
    validation = features.iloc[rows[boundary:]]
    if validation.empty:
        raise ValueError('No validation rows')
    candidates=[]
    for count in spec.n_clusters_candidates or [spec.n_clusters]:
        candidate=spec.model_copy(update={'n_clusters':count,'n_clusters_candidates':[]})
        research=fit_model(features.iloc[rows[:boundary]],candidate)
        diagnostic=silhouette_diagnostic(research[:-1].transform(validation),research.predict(validation))
        candidates.append((candidate,diagnostic))
    available=[item for item in candidates if item[1]['score'] is not None]
    spec,diagnostic=max(available,key=lambda item:item[1]['score']) if available else candidates[0]
    # Freeze this explicit recipe, recreate every preprocessing/model component.
    final = fit_model(features, spec)
    assert final is not research
    import cloudpickle
    model_path=root/'clustering_model.pkl'
    model_path.write_bytes(cloudpickle.dumps(final))
    restored=cloudpickle.loads(model_path.read_bytes())
    predicted = final.predict(future)
    if not np.array_equal(predicted,restored.predict(future)):
        raise ValueError('Clustering save/load changed predictions')
    output = test[spec.id_columns].copy()
    output['cluster'] = predicted.astype(int)
    from .util import atomic_bytes
    atomic_bytes(root / 'submission.csv', output.to_csv(index=False).encode('utf-8'))
    return {'contract': spec.model_dump(), 'validation': diagnostic,
            'publication': {'scope': 'full_train_refit', 'rows': len(output),
                            'sha256': hashlib.sha256((root / 'submission.csv').read_bytes()).hexdigest()},
            'candidate_diagnostics': [{'clusters':s.n_clusters,**d} for s,d in candidates], 'save_load_verified':True,
            'research_fit_rows': boundary, 'final_fit_rows': len(train),
            'prediction_fits': 0, 'recipe_frozen_before_refit': True}


def execute(root, contract_path, limits, started_at, run_id=None):
    """Explicit offline route: one bounded isolated worker, durable CSV recovery."""
    from .executor import Executor
    from .journal import Journal
    from .util import atomic_bytes, atomic_json, digest, file_hash, read_json

    root = root.resolve()
    spec = ClusteringContract.model_validate_json(contract_path.read_text())
    if (root / spec.train_file).samefile(root / spec.test_file):
        raise ValueError('Train and test resolve to the same source')
    # Validate paths and schema before staging; no supervised labels are inferred.
    for filename in (spec.train_file, spec.test_file):
        load_input(root, filename, spec)
    source = {name: file_hash(root / name) for name in (spec.train_file, spec.test_file)}
    binding = digest({'contract': spec.model_dump(), 'sources': source})
    run_id = run_id or binding[:16]
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id):
        raise ValueError('Invalid clustering run ID')
    work = root / 'outputs' / ('clustering-' + run_id)
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = Journal(work)
        try:
            previous = journal.get('binding')
            if previous not in (None, binding):
                raise ValueError('Saved clustering run belongs to another source or contract')
            journal.put('binding', binding)
            deadline = journal.get('deadline', started_at + limits.seconds)
            journal.put('deadline', deadline)
            executor = Executor(limits.cpu, limits.memory_mb, min(deadline, started_at + limits.seconds), journal)
            executor.recover_worker()
            attempt = work / 'attempt'
            result = journal.get('result')
            if result is None and (attempt / 'worker_result.json').is_file():
                result = read_json(attempt / 'worker_result.json')
            if result is None:
                if journal.get('attempt_started'):
                    return {'state': 'failed', 'error_code': 'outcome_unknown',
                        'reason': 'Interrupted attempt has no verified result; use a new explicit run ID to retry'}
                if time.time() >= executor.deadline:
                    return {'state': 'failed', 'error_code': 'time_budget_exhausted'}
                attempt.mkdir(exist_ok=False)
                for filename in source:
                    destination = attempt / filename
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(root / filename, destination)
                    if file_hash(destination) != source[filename]:
                        raise ValueError('Clustering source changed while staging')
                journal.put('attempt_started', True)
                result = executor.run(attempt, {'mode': 'clustering', 'contract': spec.model_dump()},
                                      max(.1, executor.deadline - time.time()))
            if result.get('status') != 'completed':
                report = {'state': 'failed', 'worker': result}
                journal.put('result', result)
                atomic_json(work / 'run_report.json', report)
                return report
            expected = result.get('hashes', {}).get('submission.csv')
            output = attempt / 'submission.csv'
            if not expected or not output.is_file() or file_hash(output) != expected:
                raise ValueError('Unverified clustering worker result')
            if any(file_hash(root / name) != expected_hash for name, expected_hash in source.items()):
                raise ValueError('Clustering input changed before publication')
            restored = pd.read_csv(output, dtype={column: str for column in spec.id_columns}, keep_default_na=False)
            test, _ = load_input(root, spec.test_file, spec)
            clusters = pd.to_numeric(restored['cluster'], errors='raise')
            selected_count = result['report']['contract']['n_clusters']
            if selected_count not in (spec.n_clusters_candidates or [spec.n_clusters]):
                raise ValueError('Cluster count differs from the frozen candidate set')
            if (list(restored) != spec.id_columns + ['cluster'] or len(restored) != len(test)
                    or not restored[spec.id_columns].equals(test[spec.id_columns])
                    or not np.isfinite(clusters).all() or not clusters.between(0, selected_count - 1).all()
                    or not (clusters == np.floor(clusters)).all()):
                raise ValueError('Clustering CSV failed schema, identifier or cluster validation')
            report = {'state': 'completed', **result['report'], 'source_hashes': source,
                      'contract_hash': binding, 'output': str(root / 'submission.csv'),
                      'limitations': ['Numeric KMeans and silhouette only; task evidence must identify clustering semantics']}
            # Save the verified result before publishing; resume never retrains it.
            journal.put_many({'result': result, 'publication_pending': report})
            atomic_bytes(root / 'submission.csv', output.read_bytes(), mode=0o644)
            atomic_json(work / 'submission_provenance.json', report['publication'])
            journal.put_many({'publication_pending': None, 'completed_report': report})
            atomic_json(work / 'run_report.json', report)
            return report
        except Exception as error:
            import traceback
            from .util import redact
            trace = work / 'error.txt'
            atomic_bytes(trace, redact(traceback.format_exc()).encode('utf-8'))
            atomic_json(work / 'error.json', {'error_code': 'clustering:' + type(error).__name__, 'traceback_ref': trace.name})
            raise
        finally:
            journal.close()

