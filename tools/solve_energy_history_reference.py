"""Explicitly exploratory follow-up to the sealed participant reference study.

Year 5 has already been scored in the initial study: the second score is reused
control evidence, not a new blind acceptance. Candidate selection still uses
only year 4. Historical-target features use strictly earlier permitted rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from solve_participant_examples import ROOT, history_features, predict_saved, run_job, sha, write


def execute(initial, out):
    out.mkdir(parents=True, exist_ok=False)
    directory = ROOT / 'data_for_participants' / 'Регрессия'
    data = pd.read_csv(directory / 'train.csv')
    test = pd.read_csv(directory / 'test.csv')
    shutil.copyfile(initial / 'splits.npz', out / 'splits.npz')
    split = np.load(out / 'splits.npz')
    fit, tune, control = split['fit'], split['tune'], split['control']
    signature = ['State_Factor', 'facility_type', 'floor_area', 'year_built', 'ELEVATION']
    deadline = time.monotonic() + 900
    rmse = lambda y, p: float(np.sqrt(mean_squared_error(y, p)))
    candidates = [
        ('history_cb8', 'catboost', {'iterations': 1600, 'depth': 8, 'learning_rate': .05}, True),
        ('history_lgb31', 'lightgbm', {'n_estimators': 1600, 'num_leaves': 31, 'learning_rate': .04, 'reg_lambda': 5}, True),
        ('history_cb8_weather', 'catboost', {'iterations': 1600, 'depth': 8, 'learning_rate': .05}, False),
    ]
    write(out / 'protocol.json', {'hypothesis': 'Earlier target values of candidate building links may help next-year transfer.',
          'signature_is_not_verified_entity_id': True, 'signature': signature,
          'control_already_observed_in_previous_study': True,
          'selection_uses_only_year4': True, 'candidates': candidates,
          'sources': {f: sha(directory / f) for f in ['train.csv', 'test.csv', 'task.md']},
          'split_sha256': sha(out / 'splits.npz')})
    results = []
    for name, model, parameters, building in candidates:
        q = {'directory': str(directory), 'target': 'site_eui', 'split': str(out / 'splits.npz'),
             'train': 'fit', 'evaluate': 'tune', 'history_signature': signature, 'building_only': building,
             'spec': {'model': model, 'hyperparameters': parameters, 'seed': 42}}
        r = run_job(out, name, q, deadline)
        if r:
            results.append({'name': name, 'iterations': r['iterations'], 'request': q,
                            'score': rmse(data.site_eui.iloc[tune], np.load(out / name / 'prediction.npy'))})
            print('SCORE', results[-1]['name'], results[-1]['score'], flush=True)
        write(out / 'tuning_results.json', results)
    if not results:
        raise RuntimeError('No history candidate completed')
    selected = min(results, key=lambda r: r['score'])
    initial_selection = json.loads((initial / 'selection.json').read_text())
    if selected['score'] >= initial_selection['tune_score']:
        write(out / 'summary.json', {'status': 'rejected_on_tune', 'best_score': selected['score'],
              'original_score': initial_selection['tune_score'], 'control_not_recomputed': True})
        return
    write(out / 'selection.json', dict(selected, selected_before_second_control=True,
          control_already_observed_in_previous_study=True, sealed_at_unix=time.time()))
    q = dict(selected['request'])
    parameters = dict(q['spec']['hyperparameters'])
    parameters['iterations' if q['spec']['model'] == 'catboost' else 'n_estimators'] = selected['iterations']
    q['spec'] = dict(q['spec'], hyperparameters=parameters)
    q.update(train='development', evaluate='control', final=True)
    if not run_job(out, 'control_fit', q, deadline):
        raise RuntimeError('Follow-up control fit failed')
    p = predict_saved(out / 'control_fit' / 'model.joblib', data.iloc[control])
    control_rmse = rmse(data.site_eui.iloc[control], p)
    np.save(out / 'control_prediction.npy', p)
    q.update(train='all')
    if not run_job(out, 'final', q, deadline):
        raise RuntimeError('Follow-up final fit failed')
    prediction = np.load(out / 'final' / 'prediction.npy')
    assert len(prediction) == len(test) and np.isfinite(prediction).all()
    submission = pd.DataFrame({'id': test.id, 'site_eui': prediction})
    assert submission.id.is_unique and submission.id.equals(test.id)
    submission.to_csv(out / 'submission.csv', index=False)
    hf = history_features(data.iloc[fit], data.iloc[tune], signature, 'site_eui', 'Year_Factor')
    ht = history_features(data, test, signature, 'site_eui', 'Year_Factor')
    write(out / 'summary.json', {'status': 'completed', 'selected': selected['name'],
          'iterations': selected['iterations'], 'tune_rmse': selected['score'],
          'original_tune_rmse': initial_selection['tune_score'], 'reused_control_rmse': control_rmse,
          'control_is_new_blind_evidence': False, 'submission_rows': len(test),
          'submission_sha256': sha(out / 'submission.csv'), 'api_calls': 0,
          'history_match_fraction_tune': float(hf.history_last_mean.notna().mean()),
          'history_match_fraction_test': float(ht.history_last_mean.notna().mean()),
          'external_quality': 'unknown: test labels unavailable'})
    print('FOLLOWUP', control_rmse, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    execute(args.initial.resolve(), args.output.resolve())
