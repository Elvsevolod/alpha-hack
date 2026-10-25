"""Isolated semantic-feature and hierarchical-score experiments, development only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from tools.literature_reference_study import run_case, probabilities, paired_group_interval, write_report
from tools.reference_features import fit_offsets, macro_f1
from tools.solve_participant_examples import ROOT, prepare_saved, predict_saved, run_job, sha, write
from tools.targeted_decisions import fit_temperature, join_hierarchy, nll, temper, unweight


def parts(path, frame):
    bundle, x = prepare_saved(path, frame)
    model = bundle['model']
    return model.models_['gate'].predict_proba(x), model.models_['attack'].predict_proba(x)


def corrected(gate, conditional, counts, normal, settings, temperatures=(1., 1.)):
    gcounts = np.array([counts[normal], counts.sum() - counts[normal]])
    ccounts = np.delete(counts, normal)
    gate = unweight(gate, gcounts, settings['gate_correction'])
    conditional = unweight(conditional, ccounts, settings['conditional_correction'])
    return temper(gate, temperatures[0]), temper(conditional, temperatures[1])


def calibration(output):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'outputs/further_reference_20261025/classification'
    directory = ROOT / 'data_for_participants/Многоклассовая классификация'
    frame, test = pd.read_csv(directory / 'train.csv'), pd.read_csv(directory / 'test.csv')
    source_protocol = json.loads((source / 'protocol.json').read_text())
    if any(sha(directory / key) != value for key, value in source_protocol['source_hashes'].items()):
        raise ValueError('Saved estimators refer to different task inputs')
    split = np.load(source / 'splits.npz')
    classes = sorted(frame.attack_cat.unique())
    normal = classes.index('Normal')
    attack = np.array([i for i in range(len(classes)) if i != normal])
    y = pd.Categorical(frame.attack_cat, categories=classes).codes
    recipes = [dict(name='raw', gate_correction=0., conditional_correction=0., temperatures=False),
        dict(name='gate_unweighted', gate_correction=1., conditional_correction=0., temperatures=False),
        dict(name='conditional_unweighted', gate_correction=0., conditional_correction=1., temperatures=False),
        dict(name='both_unweighted', gate_correction=1., conditional_correction=1., temperatures=False),
        dict(name='half_unweighted', gate_correction=.5, conditional_correction=.5, temperatures=False),
        dict(name='raw_temperatures', gate_correction=0., conditional_correction=0., temperatures=True),
        dict(name='unweighted_temperatures', gate_correction=1., conditional_correction=1., temperatures=True)]
    write(out / 'protocol.json', {'recipes': recipes, 'fixed_mixture_weights': [.5, .5], 'fixed_tau': .5,
        'offset_fit_procedure_identical_across_recipes': True, 'temperature_bounds': [.5, 2.],
        'early_stopping_and_base_estimators_frozen': True, 'audit_previously_used': True,
        'data_source_sha256': source_protocol['source_hashes'], 'source_study': source.parent.name,
        'source_code_sha256': {p: sha(ROOT / p) for p in ['tools/targeted_decisions.py',
            'tools/targeted_classification_study.py', 'tools/solve_participant_examples.py', 'tools/reference_features.py']},
        'api_calls': 0, 'frozen_at_unix': time.time()})
    cached = []
    for fold in range(2):
        rows, cal = split[f'eval{fold}'], split[f'calibration{fold}']
        mask = np.isin(rows, cal)
        xgb = predict_saved(source / f'current_weighted_cv{fold}/model.joblib', frame.iloc[rows])
        gate, conditional = parts(source / f'hierarchy_balanced_cv{fold}/model.joblib', frame.iloc[rows])
        raw = join_hierarchy(gate, conditional, normal)
        np.testing.assert_allclose(raw, np.load(source / f'hierarchy_balanced_cv{fold}/prediction.npy'), atol=1e-12)
        cached.append((xgb, gate, conditional, np.bincount(y[split[f'fit{fold}']], minlength=len(classes)), y[rows], mask))
    results = []
    for settings in recipes:
        scores, losses, fitted, offsets = [], [], [], []
        for xgb, gate, conditional, counts, labels, mask in cached:
            g, c = corrected(gate, conditional, counts, normal, settings)
            temps = (1., 1.)
            if settings['temperatures']:
                binary_labels = (labels[mask] != normal).astype(int)
                attack_mask = mask & (labels != normal)
                temps = (fit_temperature(g[mask], binary_labels),
                         fit_temperature(c[attack_mask], np.searchsorted(attack, labels[attack_mask])))
            g, c = corrected(gate, conditional, counts, normal, settings, temps)
            hierarchy = join_hierarchy(g, c, normal)
            mixture = .5 * xgb + .5 * hierarchy
            losses.append(nll(mixture[~mask], labels[~mask]))
            p = probabilities({'xgb': xgb, 'hierarchy': hierarchy}, ['xgb', 'hierarchy'], [.5, .5], counts, .5)
            bias = fit_offsets(p[mask], labels[mask])
            scores.append(macro_f1(labels[~mask], (np.log(np.maximum(p[~mask], 1e-15)) + bias).argmax(axis=1), len(classes)))
            fitted.append(list(temps)); offsets.append(bias.tolist())
        result = dict(settings=settings, score=float(np.mean(scores)), fold_scores=scores,
                      selection_nll=losses, fitted_temperatures=fitted, offsets=offsets)
        results.append(result)
        print('CALIBRATION', settings['name'], result['score'], 'nll', np.mean(losses), flush=True)
    results.sort(key=lambda item: -item['score'])
    chosen = results[0]
    temperatures = np.exp(np.log(chosen['fitted_temperatures']).mean(axis=0)).tolist()
    frozen_offsets = np.mean(chosen['offsets'], axis=0).tolist()
    selection = dict(chosen, frozen_temperatures=temperatures, frozen_offsets=frozen_offsets,
                     sealed_before_reused_audit=True, at_unix=time.time())
    write(out / 'selection.json', selection); write(out / 'results.json', results)
    # Underlying fitted models are reused unchanged for audit and final inference.
    manifest = {'target': 'attack_cat', 'classes': classes, 'normal_index': normal,
        'settings': chosen['settings'], 'temperatures': temperatures, 'offsets': frozen_offsets,
        'source': str(source), 'full_class_counts': np.bincount(y, minlength=len(classes)).tolist()}
    write(out / 'inference.json', manifest)
    rows = split['control']
    counts = np.bincount(y[split['development']], minlength=len(classes))
    audit = calibrated_prediction(manifest, frame.iloc[rows], source, 'audit', counts)
    baseline = np.load(source / 'audit_prediction.npy')
    groups = pd.util.hash_pandas_object(frame.drop(columns=['id', 'attack_cat']).iloc[rows], index=False).to_numpy()
    write(out / 'uncertainty.json', paired_group_interval(y[rows], baseline, audit, groups, len(classes)))
    prediction = predict(out, test)
    assert prediction.id.equals(test.id) and prediction.id.is_unique and not prediction.isna().any().any()
    prediction.to_csv(out / 'submission.csv', index=False)
    np.save(out / 'audit_prediction.npy', audit)
    write(out / 'summary.json', {'selected': chosen['settings']['name'], 'audit_macro_f1': macro_f1(y[rows], audit, len(classes)),
        'baseline_audit_macro_f1': macro_f1(y[rows], baseline, len(classes)), 'audit_reused': True,
        'calibration_optimization_calls': 8, 'base_estimator_fits': 0, 'api_calls': 0,
        'rows': len(test), 'submission_sha256': sha(out / 'submission.csv'), 'previous_csv_unchanged': True})


def calibrated_prediction(manifest, frame, source, stage, counts):
    xgb = predict_saved(source / f'current_weighted_{stage}/model.joblib', frame)
    g, c = parts(source / f'hierarchy_balanced_{stage}/model.joblib', frame)
    g, c = corrected(g, c, counts, manifest['normal_index'], manifest['settings'], manifest['temperatures'])
    hierarchy = join_hierarchy(g, c, manifest['normal_index'])
    p = probabilities({'xgb': xgb, 'hierarchy': hierarchy}, ['xgb', 'hierarchy'], [.5, .5], counts, .5)
    return (np.log(np.maximum(p, 1e-15)) + np.asarray(manifest['offsets'])).argmax(axis=1)


def predict(run, frame):
    manifest = json.loads((Path(run) / 'inference.json').read_text())
    encoded = calibrated_prediction(manifest, frame, Path(manifest['source']), 'final', np.asarray(manifest['full_class_counts']))
    return pd.DataFrame({'id': frame.id, 'attack_cat': np.asarray(manifest['classes'])[encoded]})


def feature_study(output, seconds):
    from tools.further_reference_study import configurations
    base = {c['name']: c for c in configurations(True)}
    xgb = base['current_weighted']
    hierarchy = base['hierarchy_balanced']
    hierarchy = dict(hierarchy, name='current_hierarchy', fixed_iterations=True,
        alternative_component_iterations={'gate': 757, 'attack': 304})
    configs = [xgb, hierarchy, dict(xgb, name='semantic_xgb', feature_plan='network_semantic'),
               dict(hierarchy, name='semantic_hierarchy', feature_plan='network_semantic')]
    recipes = [dict(names=['current_weighted', 'current_hierarchy'], weights=[.5, .5], tau=.5, offsets=True),
               dict(names=['semantic_xgb', 'semantic_hierarchy'], weights=[.5, .5], tau=.5, offsets=True)]
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + seconds
    run_case(out, 'classification', deadline, config_override=configs,
        baseline_override=.6565257846316546, locked_recipes=recipes)
    feature_controlled_audit(out / 'classification', deadline)
    write_report(out)


def feature_controlled_audit(out, deadline):
    """Audit the fixed baseline with its own controlled CV offsets as well.

    This descriptive comparison follows recipe selection, changes no selection
    and must never be presented as a fresh holdout.
    """
    out = Path(out)
    protocol = json.loads((out / 'protocol.json').read_text())
    recipes = json.loads((out / 'recipes.json').read_text())
    baseline = next(r for r in recipes if r['names'] == ['current_weighted', 'current_hierarchy'])
    candidates = {c['name']: c for c in protocol['candidates']}
    directory = ROOT / 'data_for_participants/Многоклассовая классификация'
    frame = pd.read_csv(directory / 'train.csv')
    split = np.load(out / 'splits.npz'); rows = split['control']
    classes = sorted(frame.attack_cat.unique())
    y = pd.Categorical(frame.attack_cat, categories=classes).codes
    predictions = {}
    for name in baseline['names']:
        q = {k: v for k, v in candidates[name].items() if k != 'name'}
        q.update(directory=str(directory), target='attack_cat', split=str(out / 'splits.npz'),
            train='development', evaluate='control', final=True, timeout_seconds=300)
        job = f'{name}_controlled_audit'
        if run_job(out, job, q, deadline) is None:
            raise RuntimeError('Controlled baseline audit failed; keep existing CSV')
        predictions[name] = predict_saved(out / job / 'model.joblib', frame.iloc[rows])
    counts = np.bincount(y[split['development']], minlength=len(classes))
    p = probabilities(predictions, baseline['names'], baseline['weights'], counts, baseline['tau'])
    offsets = np.mean(baseline['learned_offsets'], axis=0)
    old = (np.log(np.maximum(p, 1e-15)) + offsets).argmax(axis=1)
    candidate = np.load(out / 'audit_prediction.npy')
    np.save(out / 'controlled_baseline_prediction.npy', old)
    groups = pd.util.hash_pandas_object(frame.drop(columns=['id', 'attack_cat']).iloc[rows], index=False).to_numpy()
    write(out / 'controlled_audit.json', {'baseline_macro_f1': macro_f1(y[rows], old, len(classes)),
        'selected_macro_f1': macro_f1(y[rows], candidate, len(classes)),
        'uncertainty': paired_group_interval(y[rows], old, candidate, groups, len(classes)),
        'baseline_recipe': baseline, 'audit_previously_used': True,
        'descriptive_only_no_reselection': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['features', 'calibration'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=900)
    args = p.parse_args()
    calibration(args.output) if args.mode == 'calibration' else feature_study(args.output, args.seconds)
