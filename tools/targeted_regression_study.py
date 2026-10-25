"""Frozen refit-rule comparisons using shared reference workers and executors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from tools.literature_reference_study import reference_splits
from tools.solve_participant_examples import ROOT, run_job, sha, write, predict_saved
from tools.targeted_decisions import refit_iterations


def study(output, seconds=900):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    directory = ROOT / 'data_for_participants/Регрессия'
    frame, test = pd.read_csv(directory / 'train.csv'), pd.read_csv(directory / 'test.csv')
    y = frame.site_eui.to_numpy()
    specs = {
        'lgb': dict(model='lightgbm', hyperparameters=dict(n_estimators=2200, num_leaves=31,
            learning_rate=.03, reg_lambda=10, min_child_samples=50, colsample_bytree=.9,
            subsample=.8, subsample_freq=1), seed=42),
        'cb': dict(model='catboost', hyperparameters=dict(iterations=2200, depth=8, learning_rate=.04, l2_leaf_reg=8), seed=42)}
    baseline = {'lgb': 1035, 'cb': 979}
    rules = ['fixed_existing', 'latest_inner', 'latest_inner_sqrt_rows']
    split = reference_splits(frame, False)
    for period in [3, 4, 5, 6]:
        full = np.flatnonzero(frame.Year_Factor < period)
        latest = frame.Year_Factor.iloc[full].max()
        split[f'inner{period}'] = np.flatnonzero(frame.Year_Factor < latest)
        split[f'stop{period}'] = np.flatnonzero(frame.Year_Factor == latest)
        split[f'full{period}'] = full
        split[f'eval{period}'] = np.flatnonzero(frame.Year_Factor == period)
    # Preserve both membership AND ordering of the accepted audit's training
    # rows: stochastic learners may change when the same rows are reordered.
    split['full5'] = split['development'].copy()
    path = out / 'splits.npz'
    np.savez(path, **split)
    write(out / 'protocol.json', {'rules': rules, 'selection_periods': [3, 4], 'audit_period': 5,
        'test_period': 6, 'audit_previously_used': True, 'baseline_commit': 'ff5f93b',
        'weights': {'lgb': .5, 'cb': .5}, 'iteration_limit': 2500,
        'sqrt_scaling_is_engineering_hypothesis': True, 'api_calls': 0,
        'source_sha256': {str(p): sha(ROOT / p) for p in ['tools/targeted_regression_study.py',
            'tools/targeted_decisions.py', 'tools/solve_participant_examples.py', 'tools/reference_features.py']},
        'data_sha256': {f: sha(directory / f) for f in ['train.csv', 'test.csv', 'task.md']},
        'split_sha256': sha(path), 'frozen_at_unix': time.time()})
    deadline = time.monotonic() + seconds
    inner, values, iterations = {}, {}, {}
    base = dict(directory=str(directory), target='site_eui', split=str(path),
                building_only=True, feature_plan='energy_building', timeout_seconds=180)
    def fit(name, spec, train, evaluate, *, fixed=False, final=False):
        q = dict(base, spec=spec, train=train, evaluate=evaluate, fixed_iterations=fixed, final=final)
        result = run_job(out, name, q, deadline)
        if result is None:
            raise RuntimeError(f'Refit experiment failed: {name}')
        return result
    def specified(spec, count):
        key = 'iterations' if spec['model'] == 'catboost' else 'n_estimators'
        return dict(spec, hyperparameters=dict(spec['hyperparameters'], **{key: count}))
    def counts(period):
        if period not in inner:
            inner[period] = {}
            for model, spec in specs.items():
                r = fit(f'{model}_inner{period}', spec, f'inner{period}', f'stop{period}')
                inner[period][model] = r['iterations']
        return inner[period]
    for period in [3, 4]:
        found = counts(period)
        for rule in rules:
            p, numbers = [], {}
            for model, spec in specs.items():
                number = baseline[model] if rule == 'fixed_existing' else refit_iterations(rule, found[model], len(split[f'inner{period}']), len(split[f'full{period}']))
                numbers[model] = number
                name = f'{model}_{rule}_period{period}'
                fit(name, specified(spec, number), f'full{period}', f'eval{period}', fixed=True)
                p.append(np.load(out / name / 'prediction.npy'))
            combined = sum(p) / 2
            values[rule, period] = float(np.sqrt(np.mean((combined - y[split[f'eval{period}']]) ** 2)))
            iterations[rule, period] = numbers
            print('RULE', rule, period, values[rule, period], numbers, flush=True)
    scores = {rule: float(np.mean([values[rule, period] for period in [3, 4]])) for rule in rules}
    chosen = min(rules, key=lambda name: scores[name])
    write(out / 'selection.json', {'rule': chosen, 'scores': scores,
        'folds': {rule: {str(p): {'rmse': values[rule, p], 'iterations': iterations[rule, p]} for p in [3, 4]} for rule in rules},
        'sealed_before_reused_audit': True, 'at_unix': time.time()})
    predictions, final_predictions, manifest = [], [], []
    for period, final in [(5, False), (6, True)]:
        found = counts(period) if chosen != 'fixed_existing' else None
        for model, spec in specs.items():
            number = baseline[model] if found is None else refit_iterations(chosen, found[model], len(split[f'inner{period}']), len(split[f'full{period}']))
            name = f'{model}_accepted_period{period}'
            fit(name, specified(spec, number), f'full{period}', 'control', fixed=True, final=final)
            p = np.load(out / name / 'prediction.npy')
            (final_predictions if final else predictions).append(p)
            if final:
                manifest.append({'model': f'{name}/model.joblib', 'weight': .5})
    p = sum(predictions) / 2
    final = sum(final_predictions) / 2
    score = float(np.sqrt(np.mean((p - y[split['control']]) ** 2)))
    pd.DataFrame({'id': test.id, 'site_eui': final}).to_csv(out / 'submission.csv', index=False)
    np.save(out / 'audit_prediction.npy', p)
    write(out / 'inference.json', {'target': 'site_eui', 'classes': None, 'members': manifest})
    write(out / 'summary.json', {'rule': chosen, 'selection_scores': scores, 'audit_rmse': score,
        'baseline_audit_rmse': 39.03614722027966, 'audit_reused': True, 'api_calls': 0,
        'rows': len(test), 'submission_sha256': sha(out / 'submission.csv'), 'existing_csv_not_changed': True})
    print('REFIT_AUDIT', score, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=900)
    args = p.parse_args()
    study(args.output, args.seconds)
