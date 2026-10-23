"""One preregistered offline C2 experiment. Development tooling, never delivered."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


SPACE = {'learning_rate': {'type': 'float', 'low': .015, 'high': .2, 'log': True},
    'num_leaves': {'type': 'int', 'low': 8, 'high': 64},
    'min_child_samples': {'type': 'int', 'low': 10, 'high': 80},
    'reg_lambda': {'type': 'float', 'low': .0001, 'high': 30., 'log': True},
    'colsample_bytree': {'type': 'float', 'low': .6, 'high': 1.}}


def case(args):
    import optuna
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from automl_agent.contracts import PipelineSpec
    from automl_agent.data import Dataset
    from automl_agent.evaluation import align_predictions, score_predictions
    from automl_agent.search import SearchPipeline
    from automl_agent.util import digest
    from automl_agent.validation import choose_validation
    started = time.monotonic()
    source = 'Регрессия' if args.kind == 'regression' else 'Многоклассовая классификация'
    data = Dataset(args.data / source / 'task.md')
    task, questions = data.draft()
    assert task and not questions
    data.activate(task)
    plan = choose_validation(data, seed=42)
    rows = plan.fit
    if args.kind == 'classification' and len(rows) > 12000:
        rows, _ = train_test_split(rows, train_size=12000, random_state=2026,
            stratify=data.train.iloc[rows][task.targets[0]])
        rows = sorted(rows)
    X = data.features(rows)
    X.index = [data.row_keys[i] for i in rows]
    y = data.train.iloc[rows][task.targets].copy(); y.index = X.index
    validation = data.features(plan.validation)
    validation.index = [data.row_keys[i] for i in plan.validation]
    actual = data.train.iloc[plan.validation][task.targets].reset_index(drop=True)
    spec = PipelineSpec(engine='optuna', model='lightgbm', representation='native', seed=args.seed,
        n_trials=8, hyperparameters={'n_estimators': 200}, search_space=SPACE)
    model = SearchPipeline(spec, task)
    context = {'cpu': 1, 'memory_mb': 3500, 'task': task.model_dump(mode='json'),
        'strategy': plan.strategy, 'protocol_hash': plan.hash, 'split_id': 'validation',
        'workdir': str(args.output), 'budget_seconds': max(.1, 175 - (time.monotonic() - started)),
        'phase_estimates': {'selected_fit_seconds': 3, 'prediction_seconds': 2, 'publication_seconds': 1}}
    factory = optuna.samplers.TPESampler
    with patch.object(optuna.samplers, 'TPESampler', lambda **kw: factory(**kw, n_startup_trials=args.startup)):
        summary = model.fit(X, y, context)
    bundle = model.predict(validation, context)
    predictions = align_predictions(bundle, validation.index.tolist(), task, plan.hash, 'validation')
    score = score_predictions(task, actual, predictions, bundle.prediction_type)
    values = {target: predictions[target].reshape(-1).tolist() for target in task.targets}
    pd.DataFrame(values).to_csv(args.output / 'dev_predictions.csv', index=False)
    trials = summary.inner_scores['trials']
    diagnostic = [json.loads(line) for line in (args.output / 'diagnostic_trials.jsonl').read_text().splitlines()]
    frozen = model.frozen_spec().model_dump(mode='json')
    save(args.output / 'result.json', {'kind': args.kind, 'seed': args.seed, 'startup': args.startup,
        'status': 'completed', 'dev_score': score, 'metric': task.metric.model_dump(),
        'seconds': time.monotonic() - started, 'fit_rows': len(rows), 'validation_rows': len(validation),
        'fit_rows_hash': digest(rows), 'protocol_hash': plan.hash, 'source_hash': data.source_hash,
        'control_opened': False, 'trials_completed': len(trials),
        'trials_attempted': sum(row['kind'] == 'trial_started' for row in diagnostic),
        'trials_failed': sum(row.get('status') == 'failed' for row in diagnostic),
        'adaptive_trials': max(0, len(trials) - args.startup), 'trials': trials,
        'selected_recipe': frozen, 'budget': summary.budget})


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    order = [(kind, seed, startup) for kind in ('regression', 'classification') for seed in (42, 43)
        for startup in ((10, 3) if seed == 42 else (3, 10))]
    save(args.output / 'PREREGISTERED.json', {'base': 'dist/solution_v043.zip',
        'base_sha256': '7c5169d525bc9503e5c7cda6091b12a46dfacfb5382997d8bb7a2efa1c202ee7',
        'order': order, 'space': SPACE, 'model': 'lightgbm', 'n_estimators': 200, 'n_trials': 8,
        'case_deadline_seconds': 180, 'series_deadline_seconds': 1800, 'cpu': 1,
        'classification_sample': '12000 fit-only stratified rows; fixed seed 2026',
        'regression_rows': 'all research fit rows; temporal inner split',
        'promotion_rule': 'At least one task mean gain exceeds 1% RMSE or .005 F1; every seed within that regression tolerance. All runs completed 8 trials. Otherwise retain startup10.',
        'control_labels_used': False, 'seeds_or_startup_retry_allowed': False})
    results = []
    started = time.monotonic()
    for kind, seed, startup in order:
        directory = args.output / f'{kind}_{seed}_{startup}'; directory.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), '--case', '--kind', kind,
            '--seed', str(seed), '--startup', str(startup), '--output', str(directory), '--data', str(args.data)]
        environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        try:
            with (directory / 'run.log').open('w') as log:
                completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    timeout=min(180, 1800 - (time.monotonic() - started)))
            value = json.loads((directory / 'result.json').read_text()) if completed.returncode == 0 else {'status': 'failed', 'returncode': completed.returncode}
        except subprocess.TimeoutExpired:
            value = {'status': 'timeout'}
        row = {'kind': kind, 'seed': seed, 'startup': startup, **value}
        results.append(row); save(args.output / 'RESULTS.json', results)
        print(json.dumps({k: row[k] for k in ('kind', 'seed', 'startup', 'status')} | {k: row[k] for k in ('dev_score', 'seconds', 'trials_completed') if k in row}), flush=True)
    comparisons = []
    for kind in ('regression', 'classification'):
        for seed in (42, 43):
            a, b = [next(row for row in results if (row['kind'], row['seed'], row['startup']) == (kind, seed, startup)) for startup in (10, 3)]
            if any(row['status'] != 'completed' for row in (a, b)):
                comparisons.append({'kind': kind, 'seed': seed, 'complete': False}); continue
            gain = (a['dev_score'] - b['dev_score']) if kind == 'regression' else (b['dev_score'] - a['dev_score'])
            tolerance = .01 * abs(a['dev_score']) if kind == 'regression' else .005
            comparisons.append({'kind': kind, 'seed': seed, 'complete': a['trials_completed'] == b['trials_completed'] == 8,
                'gain': gain, 'tolerance': tolerance, 'relative_gain': gain / tolerance})
    eligible = all(row['complete'] and row['relative_gain'] >= -1 for row in comparisons)
    signal = any(sum(row.get('relative_gain', 0) for row in comparisons if row['kind'] == kind) / 2 > 1 for kind in ('regression', 'classification'))
    save(args.output / 'DECISION.json', {'promote_startup3': eligible and signal, 'comparisons': comparisons,
        'series_seconds': time.monotonic() - started, 'scope': 'public development; private transfer not established'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'data_for_participants'); p.add_argument('--case', action='store_true')
    p.add_argument('--kind'); p.add_argument('--seed', type=int); p.add_argument('--startup', type=int)
    args = p.parse_args(); case(args) if args.case else run(args)
