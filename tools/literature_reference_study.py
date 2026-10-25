"""Offline source-informed reference study. Never part of contest runtime.

Frozen candidates, two forward regression origins; separate early-stop/policy
calibration and selection groups for classification. Previous control is a
reused audit, never described as fresh blind evidence.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from tools.reference_features import fit_offsets, macro_f1
from tools.solve_participant_examples import ROOT, predict_saved, run_job, sha, write


def candidates(classification):
    def item(name, model, parameters, **options):
        return dict(name=name, spec={'model': model, 'hyperparameters': parameters, 'seed': 42}, **options)
    if classification:
        return [
            item('anchor31', 'lightgbm', dict(n_estimators=173, num_leaves=31, learning_rate=.05, reg_lambda=3), fixed_iterations=True),
            item('anchor63', 'lightgbm', dict(n_estimators=112, num_leaves=63, learning_rate=.05, reg_lambda=3), fixed_iterations=True),
            item('macro31', 'lightgbm', dict(n_estimators=1200, num_leaves=31, learning_rate=.04, reg_lambda=5), feature_plan='network', macro_stopping=True),
            item('macro63', 'lightgbm', dict(n_estimators=1200, num_leaves=63, learning_rate=.04, reg_lambda=5), feature_plan='network', macro_stopping=True),
            item('xgb5', 'xgboost', dict(n_estimators=900, max_depth=5, learning_rate=.06, reg_lambda=5, min_child_weight=2, subsample=.85, colsample_bytree=.85)),
            item('xgb5_weighted', 'xgboost', dict(n_estimators=900, max_depth=5, learning_rate=.06, reg_lambda=5, min_child_weight=2, subsample=.85, colsample_bytree=.85), feature_plan='network', class_weight_exponent=.5),
            item('forest', 'random_forest', dict(n_estimators=250, min_samples_leaf=2, max_features=.7, class_weight='balanced_subsample'), feature_plan='network'),
            item('mlp', 'mlp', dict(max_iter=60, hidden_layer_sizes=[128, 64], alpha=.1, batch_size=256), feature_plan='network'),
        ]
    return [
        item('anchor_cb', 'catboost', dict(iterations=1199, depth=8, learning_rate=.05), building_only=True, fixed_iterations=True),
        item('anchor_lgb', 'lightgbm', dict(n_estimators=460, num_leaves=31, learning_rate=.04, reg_lambda=5), fixed_iterations=True),
        item('engineered_cb', 'catboost', dict(iterations=1800, depth=8, learning_rate=.04, l2_leaf_reg=8), building_only=True, feature_plan='energy_building'),
        item('imputed_cb', 'catboost', dict(iterations=1800, depth=8, learning_rate=.04, l2_leaf_reg=8), building_only=True, feature_plan='energy_building', impute_rating=True),
        item('regularized_lgb', 'lightgbm', dict(n_estimators=2000, num_leaves=31, learning_rate=.03, reg_lambda=10, min_child_samples=50, colsample_bytree=.9, subsample=.8, subsample_freq=1), building_only=True, feature_plan='energy_building'),
        item('xgb6', 'xgboost', dict(n_estimators=1400, max_depth=6, learning_rate=.04, reg_lambda=10, min_child_weight=5, subsample=.85, colsample_bytree=.85), feature_plan='energy_all'),
        item('xgb_building', 'xgboost', dict(n_estimators=1400, max_depth=5, learning_rate=.04, reg_lambda=10, min_child_weight=5, subsample=.85, colsample_bytree=.85), building_only=True, feature_plan='energy_building'),
        item('extra_trees', 'extra_trees', dict(n_estimators=400, min_samples_leaf=3, max_features=1.), building_only=True, feature_plan='energy_building'),
        item('mlp', 'mlp', dict(max_iter=80, hidden_layer_sizes=[128, 64], alpha=.1, batch_size=256), building_only=True, feature_plan='energy_building'),
    ]


def probabilities(predictions, names, weights, counts, tau):
    values = []
    factors = (counts.max() / counts) ** tau
    for name, weight in zip(names, weights):
        p = predictions[name] * factors
        values.append(weight * p / p.sum(axis=1, keepdims=True))
    return sum(values)


def reference_splits(frame, classification):
    """Reproduce the initial fixed partitions without ignored prior artifacts."""
    if classification:
        features = frame.drop(columns=['id', 'attack_cat'])
        groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
        development, control = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(features, frame.attack_cat, groups))
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=43).split(features.iloc[development], frame.attack_cat.iloc[development], groups[development]))
        fit, tune = development[a], development[b]
    else:
        fit, tune, control = [np.flatnonzero(mask) for mask in [frame.Year_Factor < 4, frame.Year_Factor == 4, frame.Year_Factor == 5]]
        development = np.r_[fit, tune]
    return dict(fit=fit, tune=tune, control=control, development=development, all=np.arange(len(frame)))


def validate_locked_recipes(recipes, completed):
    """Fail closed when a controlled comparison did not complete its models."""
    if not recipes:
        raise ValueError('Controlled comparison requires at least one recipe')
    for recipe in recipes:
        names, weights = recipe['names'], recipe['weights']
        if not names or len(names) != len(weights) or len(set(names)) != len(names):
            raise ValueError('Invalid controlled recipe members')
        if not all(n in completed for n in names):
            raise ValueError('Controlled comparison has incomplete candidates')
        if not np.isfinite(weights).all() or min(weights) < 0 or not np.isclose(sum(weights), 1):
            raise ValueError('Controlled recipe weights must sum to one')


def diverse_pairs(ordered_singles, completed):
    """Retain model-family diversity before testing blends on development data."""
    unique = list(dict.fromkeys(r['names'][0] for r in ordered_singles))
    families = {}
    for name in unique:
        config = completed[name]['config']
        family = (config['spec']['model'], config.get('alternative'))
        families.setdefault(family, name)
    pool = list(dict.fromkeys(unique[:3] + list(families.values())))
    return [(pool[a], pool[b]) for a in range(len(pool)) for b in range(a + 1, len(pool))]


def run_case(root, kind, deadline, *, config_override=None, reuse_root=None, baseline_override=None, reference_pair=None, locked_recipes=None):
    classification = kind == 'classification'
    directory = ROOT / 'data_for_participants' / ('Многоклассовая классификация' if classification else 'Регрессия')
    target = 'attack_cat' if classification else 'site_eui'
    frame, test = pd.read_csv(directory / 'train.csv'), pd.read_csv(directory / 'test.csv')
    evidence = json.loads((ROOT / 'docs/reports/examples/2026-10-25-reference-evidence.json').read_text())['cases'][kind]
    input_hashes = {f: sha(directory / f) for f in ['train.csv', 'test.csv', 'task.md']}
    if evidence['protocol']['source_hashes'] != input_hashes:
        raise ValueError('Historical baseline scores refer to different input data')
    previous = evidence['summary'] if baseline_override is None else {'control': {'selected_score': baseline_override}}
    out = root / kind
    out.mkdir()
    original = ROOT / 'outputs/participant_reference_20261025' / kind
    old_split = reference_splits(frame, classification)
    if (original / 'splits.npz').exists():
        persisted = np.load(original / 'splits.npz')
        if any(not np.array_equal(values, persisted[key]) for key, values in old_split.items()):
            raise ValueError('Initial partition reproduction differs from persisted reference')
    control, development = old_split['control'], old_split['development']
    classes = sorted(frame[target].unique()) if classification else None
    y = pd.Categorical(frame[target], categories=classes).codes if classification else frame[target].to_numpy()
    split = {'control': control, 'development': development, 'all': np.arange(len(frame))}
    validation = []
    if classification:
        fit, tune = old_split['fit'], old_split['tune']
        groups = pd.util.hash_pandas_object(frame.drop(columns=['id', target]), index=False).to_numpy()
        a, b = next(StratifiedGroupKFold(2, shuffle=True, random_state=2026).split(frame.iloc[tune], y[tune], groups[tune]))
        for i, (calibration, selection) in enumerate([(a, b), (b, a)]):
            split.update({f'fit{i}': fit, f'eval{i}': tune, f'calibration{i}': tune[calibration]})
            assert not set(groups[tune[calibration]]) & set(groups[tune[selection]])
            assert not set(groups[fit]) & set(groups[tune])
            validation.append(dict(labels=y[tune], calibration=calibration, selection=selection,
                                   counts=np.bincount(y[fit], minlength=len(classes))))
    else:
        for i, year in enumerate([3, 4]):
            fit, evaluate = np.flatnonzero(frame.Year_Factor < year), np.flatnonzero(frame.Year_Factor == year)
            split.update({f'fit{i}': fit, f'eval{i}': evaluate})
            validation.append(dict(labels=y[evaluate], year=year))
    split_path = out / 'splits.npz'
    np.savez(split_path, **split)
    configs = candidates(classification) if config_override is None else config_override
    worker_files = ['tools/solve_participant_examples.py', 'tools/reference_features.py', 'automl_agent/pipelines.py']
    if any(c.get('alternative') for c in configs):
        worker_files.append('tools/reference_alternatives.py')
    worker_sources = {name: sha(ROOT / name) for name in worker_files}
    dependencies = {name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'catboost', 'lightgbm', 'xgboost']}
    dependencies['python'] = sys.version.split()[0]
    if any(c.get('alternative') == 'tabm_mini' for c in configs):
        dependencies['torch'] = version('torch')
    write(out / 'protocol.json', {'candidate_freeze_unix': time.time(), 'candidates': configs,
        'worker_source_sha256': worker_sources,
        'dependencies': dependencies,
        'source_hashes': input_hashes,
        'split_sha256': sha(split_path), 'previous_control_reused': True,
        'baseline_audit_score': previous['control']['selected_score'],
        'ensemble_pair_policy': 'top three single recipes plus best recipe per (model, alternative) family',
        'locked_recipes': locked_recipes,
        'selection': 'two complementary calibration/selection group partitions, calibration only for stopping and offsets' if classification else 'mean RMSE on next-period validation years 3 and 4',
        'policy_search': {'tau': [0, .5, 1.], 'offset_bound': 1., 'support_shrinkage': 50., 'coordinate_passes': 2},
        'resources': {'cpu_per_worker': 4, 'timeout_per_worker_seconds': 180},
        'no_external_data_or_models': True, 'api_calls': 0})
    predictions = [{}, {}]
    completed = {}
    reusable = {}
    reuse = Path(reuse_root) / kind if reuse_root else None
    if reuse is not None:
        old_protocol = json.loads((reuse / 'protocol.json').read_text())
        new_protocol = json.loads((out / 'protocol.json').read_text())
        if old_protocol['source_hashes'] != new_protocol['source_hashes']:
            raise ValueError('Cannot reuse predictions for changed input files')
        if old_protocol.get('worker_source_sha256') != worker_sources:
            raise ValueError('Cannot reuse predictions for unverified or changed worker source')
        if old_protocol.get('dependencies') != dependencies:
            raise ValueError('Cannot reuse predictions for unverified or changed dependencies')
        previous_split = np.load(reuse / 'splits.npz')
        if any(key not in previous_split or not np.array_equal(values, previous_split[key]) for key, values in split.items()):
            raise ValueError('Cannot reuse predictions for changed splits')
        reusable = json.loads((reuse / 'completed.json').read_text())
    for config in configs:
        name = config['name']
        if name in reusable and reusable[name]['config'] == config:
            completed[name] = reusable[name]
            for fold in range(2):
                predictions[fold][name] = np.load(reuse / f'{name}_cv{fold}' / 'prediction.npy')
            write(out / 'completed.json', completed)
            print('REUSE', name, flush=True)
            continue
        records = []
        for fold in range(2):
            q = {k: v for k, v in config.items() if k != 'name'}
            q.update(directory=str(directory), target=target, split=str(split_path),
                     train=f'fit{fold}', evaluate=f'eval{fold}', timeout_seconds=180)
            if classification:
                q['early_stop_evaluate'] = f'calibration{fold}'
            result = run_job(out, f'{name}_cv{fold}', q, deadline)
            if result is None:
                break
            records.append(result)
            predictions[fold][name] = np.load(out / f'{name}_cv{fold}' / 'prediction.npy')
        if len(records) == 2:
            completed[name] = {'config': config, 'folds': records}
        write(out / 'completed.json', completed)
    if not completed:
        raise RuntimeError('No candidate completed both comparisons')
    recipes = []
    def assess(names, weights, tau=0, offsets=False):
        scores, learned = [], []
        for i, valid in enumerate(validation):
            if classification:
                p = probabilities(predictions[i], names, weights, valid['counts'], tau)
                bias = fit_offsets(p[valid['calibration']], valid['labels'][valid['calibration']]) if offsets else np.zeros(len(classes))
                selection = valid['selection']
                score = macro_f1(valid['labels'][selection], (np.log(np.maximum(p[selection], 1e-15)) + bias).argmax(axis=1), len(classes))
                learned.append(bias.tolist())
            else:
                p = sum(weight * predictions[i][name] for name, weight in zip(names, weights))
                score = float(np.sqrt(np.mean((valid['labels'] - p) ** 2)))
            scores.append(score)
        item = dict(names=names, weights=weights, tau=tau, offsets=offsets, fold_scores=scores,
                    score=float(np.mean(scores)), learned_offsets=learned)
        recipes.append(item)
        return item
    direction = -1 if classification else 1
    if locked_recipes is not None:
        validate_locked_recipes(locked_recipes, completed)
        for settings in locked_recipes:
            assess(**settings)
    else:
        for name in completed:
            if classification:
                for tau in [0, .5, 1.]:
                    assess([name], [1.], tau, False)
                    assess([name], [1.], tau, True)
            else:
                assess([name], [1.])
        best_single = sorted(recipes, key=lambda r: direction * r['score'])
        pairs = diverse_pairs(best_single, completed)
        anchors = tuple(reference_pair) if reference_pair else ('anchor31', 'anchor63') if classification else ('anchor_cb', 'anchor_lgb')
        if all(n in completed for n in anchors) and anchors not in pairs:
            pairs.append(anchors)
        for a, b in pairs:
            for weight in [.25, .5, .75]:
                for tau in ([0, .5, 1.] if classification else [0]):
                    assess([a, b], [weight, 1 - weight], tau, False)
                    if classification:
                        assess([a, b], [weight, 1 - weight], tau, True)
    recipes.sort(key=lambda r: direction * r['score'])
    chosen = recipes[0]
    selection = dict(chosen, sealed_before_reused_audit=True, at_unix=time.time())
    write(out / 'recipes.json', recipes)
    write(out / 'selection.json', selection)
    print('SELECTED', kind, json.dumps(chosen), flush=True)
    audit_predictions, final_predictions, iterations = {}, {}, {}
    for name in chosen['names']:
        entry = completed[name]
        q = {k: v for k, v in entry['config'].items() if k != 'name'}
        iteration = int(np.median([r['iterations'] for r in entry['folds']]))
        iterations[name] = iteration
        key = 'iterations' if q['spec']['model'] == 'catboost' else 'max_iter' if q['spec']['model'] == 'mlp' else 'n_estimators'
        q['spec'] = dict(q['spec'], hyperparameters=dict(q['spec']['hyperparameters'], **{key: iteration}))
        if q.get('alternative') == 'hierarchical':
            q['alternative_component_iterations'] = {stage: int(np.median([r['component_iterations'][stage] for r in entry['folds']])) for stage in ['gate', 'attack']}
        q.update(directory=str(directory), target=target, split=str(split_path), train='development', evaluate='control', final=True, timeout_seconds=300)
        if run_job(out, f'{name}_audit', q, deadline) is None:
            raise RuntimeError('Audit fit failed; sealed selection saved')
        audit_predictions[name] = predict_saved(out / f'{name}_audit' / 'model.joblib', frame.iloc[control])
        q['train'] = 'all'
        if run_job(out, f'{name}_final', q, deadline) is None:
            raise RuntimeError('Final fit failed; prior solution not changed')
        final_predictions[name] = np.load(out / f'{name}_final' / 'prediction.npy')
    frozen_offsets = np.mean(chosen['learned_offsets'], axis=0) if classification else None
    def combine(preds, rows):
        if classification:
            counts = np.bincount(y[rows], minlength=len(classes))
            p = probabilities(preds, chosen['names'], chosen['weights'], counts, chosen['tau'])
            return (np.log(np.maximum(p, 1e-15)) + frozen_offsets).argmax(axis=1)
        return sum(weight * preds[name] for name, weight in zip(chosen['names'], chosen['weights']))
    audit = combine(audit_predictions, development)
    audit_score = macro_f1(y[control], audit, len(classes)) if classification else float(np.sqrt(np.mean((y[control] - audit) ** 2)))
    prediction = combine(final_predictions, np.arange(len(frame)))
    if classification:
        prediction = np.asarray(classes)[prediction]
    else:
        assert np.isfinite(prediction).all()
    submission = pd.DataFrame({'id': test.id, target: prediction})
    assert len(submission) == len(test) and submission.id.equals(test.id) and submission.id.is_unique
    assert not submission.isna().any().any()
    submission.to_csv(out / 'submission.csv', index=False)
    np.save(out / 'audit_prediction.npy', audit)
    manifest = {'target': target, 'classes': classes, 'tau': chosen['tau'],
                'offsets': frozen_offsets.tolist() if classification else None,
                'class_counts': np.bincount(y, minlength=len(classes)).tolist() if classification else None,
                'members': [{'model': f'{n}_final/model.joblib', 'weight': w} for n, w in zip(chosen['names'], chosen['weights'])]}
    write(out / 'inference.json', manifest)
    summary = {'status': 'completed', 'selection': selection, 'iterations': iterations, 'audit_score': audit_score,
        'audit_is_reused': True, 'previous_audit_score': previous['control']['selected_score'],
        'submission_rows': len(submission), 'submission_sha256': sha(out / 'submission.csv'),
        'api_calls': 0, 'external_quality': 'unknown; no labels accessed', 'previous_submission_not_modified': True}
    write(out / 'summary.json', summary)
    print('AUDIT', kind, audit_score, 'previous', previous['control']['selected_score'], flush=True)


def predict(run, frame):
    run = Path(run)
    manifest = json.loads((run / 'inference.json').read_text())
    ps = [predict_saved(run / m['model'], frame) for m in manifest['members']]
    if manifest['classes']:
        counts = np.asarray(manifest['class_counts'])
        factors = (counts.max() / counts) ** manifest['tau']
        ps = [p * factors / (p * factors).sum(axis=1, keepdims=True) for p in ps]
    p = sum(m['weight'] * v for m, v in zip(manifest['members'], ps))
    if manifest['classes']:
        p = np.asarray(manifest['classes'])[(np.log(np.maximum(p, 1e-15)) + np.asarray(manifest['offsets'])).argmax(axis=1)]
    return pd.DataFrame({'id': frame.id, manifest['target']: p})


def write_report(root):
    root = Path(root)
    lines = ['# Source-informed reference study', '',
        'Offline experiments; development research is separate from contest runtime.',
        'All audit splits were inspected previously; these are reused audits, not fresh blind tests.', '']
    for kind in ['regression', 'classification']:
        out = root / kind
        if not (out / 'summary.json').exists():
            continue
        summary = json.loads((out / 'summary.json').read_text())
        recipes = json.loads((out / 'recipes.json').read_text())
        completed = json.loads((out / 'completed.json').read_text())
        single = {}
        for r in recipes:
            if len(r['names']) == 1:
                single.setdefault(r['names'][0], r)
        lines += [f'## {kind}', '', '| Candidate | Best selection score | Fold scores | Tau | Offsets |',
                  '|---|---:|---|---:|---|']
        for name, r in single.items():
            lines.append(f"| {name} | {r['score']:.6f} | {r['fold_scores']} | {r['tau']} | {r['offsets']} |")
        lines += ['', f"Frozen choice: {summary['selection']}.",
                  f"Reused audit: {summary['audit_score']:.6f}; previous recipe: {summary['previous_audit_score']:.6f}.",
                  f"Iterations: {summary['iterations']}.",
                  f"Submission: {summary['submission_rows']} rows; SHA-256 `{summary['submission_sha256']}`.",
                  'External-test quality remains unknown. This does not evaluate autonomous Qwen decisions.', '']
        lines.append(f"Completed paired recipes: {len(completed)}. All attempted fits and optional auxiliary fits are recorded in job artifacts.")
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def paired_group_interval(labels, baseline, candidate, groups, classes, resamples=500):
    """Conditional paired cluster bootstrap, not a claim about unseen domains.

    Identical-feature clusters are the resampling units. Other dependencies and
    previous adaptive use of this audit are not corrected by this interval.
    """
    if not 10 <= resamples <= 1000:
        raise ValueError('Choose 10..1000 resamples for this diagnostic')
    _, inverse = np.unique(groups, return_inverse=True)
    count = int(inverse.max() + 1)
    matrices = []
    for prediction in [baseline, candidate]:
        grouped = np.zeros((count, classes, classes), dtype=float)
        np.add.at(grouped, (inverse, labels, prediction), 1)
        matrices.append(grouped.reshape(count, classes * classes))
    rng = np.random.default_rng(42)
    weights = rng.multinomial(count, np.full(count, 1 / count), size=resamples).astype(float)
    values = []
    for grouped in matrices:
        confusion = (weights @ grouped).reshape(resamples, classes, classes)
        denominator = confusion.sum(axis=1) + confusion.sum(axis=2)
        tp = np.diagonal(confusion, axis1=1, axis2=2)
        values.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0).mean(axis=1))
    delta = values[1] - values[0]
    return {'delta_macro_f1': macro_f1(labels, candidate, classes) - macro_f1(labels, baseline, classes),
        'conditional_percentile_95': np.quantile(delta, [.025, .975]).tolist(),
        'resamples': resamples, 'seed': 42, 'groups': count,
        'limitations': ['assumes independent identical-feature clusters; this does not establish session independence',
                       'conditions on reused audit; does not correct prior adaptive model selection',
                       'does not establish external-domain quality']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', choices=['regression', 'classification', 'both'], default='both')
    parser.add_argument('--classification-ablation', action='store_true')
    parser.add_argument('--reuse-root', type=Path)
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--predict-run', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--prediction', type=Path)
    args = parser.parse_args()
    if args.predict_run:
        if not args.data or not args.prediction:
            parser.error('--data and --prediction are required for inference')
        predict(args.predict_run, pd.read_csv(args.data)).to_csv(args.prediction, index=False)
        raise SystemExit(0)
    if args.output is None:
        parser.error('--output is required for experiments')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.seconds
    if args.classification_ablation:
        base = {c['name']: c for c in candidates(True)}
        raw_weighted = dict(base['xgb5_weighted'], name='xgb_raw_weighted')
        raw_weighted.pop('feature_plan')
        engineered = dict(base['xgb5'], name='xgb_engineered', feature_plan='network')
        configs = [base[n] for n in ['anchor31', 'anchor63', 'xgb5', 'xgb5_weighted']] + [raw_weighted, engineered]
        run_case(out, 'classification', deadline, config_override=configs, reuse_root=args.reuse_root)
        write_report(out)
        raise SystemExit(0)
    for kind in (['regression', 'classification'] if args.case == 'both' else [args.case]):
        run_case(out, kind, deadline)
    write_report(out)
