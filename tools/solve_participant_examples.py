"""Development reference experiments, not the autonomous contest entry point.

No network, external data, or test labels. Reuses the production estimator factory.
Run: .venv/bin/python tools/solve_participant_examples.py --output outputs/reference
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history_features(reference, query, signature, target, period):
    """Candidate entity links, always using strictly earlier permitted rows.

    A signature is not an asserted building identity. Keep collision/support
    information and never join current-year or future target values.
    """
    import numpy as np
    import pandas as pd

    reference = reference.copy()
    reference['_group'] = pd.util.hash_pandas_object(reference[signature], index=False).to_numpy()
    keys = pd.util.hash_pandas_object(query[signature], index=False)
    result = pd.DataFrame(np.nan, index=query.index, columns=[
        'history_last_mean', 'history_mean', 'history_std', 'history_count', 'history_gap'])
    for value in sorted(query[period].dropna().unique()):
        permitted = reference.loc[reference[period] < value]
        mask = query[period] == value
        if permitted.empty:
            continue
        summary = permitted.groupby('_group')[target].agg(['mean', 'std', 'count'])
        latest = permitted.groupby('_group')[period].max()
        current = permitted.join(latest.rename('_latest'), on='_group')
        last = current.loc[current[period] == current['_latest']].groupby('_group')[target].mean()
        for name, values in [('history_last_mean', last), ('history_mean', summary['mean']),
                             ('history_std', summary['std']), ('history_count', summary['count'])]:
            result.loc[mask, name] = keys.loc[mask].map(values).to_numpy()
        result.loc[mask, 'history_gap'] = value - keys.loc[mask].map(latest).to_numpy()
    return result


def worker(request):
    import joblib
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from automl_agent.contracts import PipelineSpec
    from automl_agent.pipelines import make_estimator
    from tools.reference_features import engineer, macro_f1, RatingImputer

    q = json.loads(Path(request).read_text())
    directory = Path(q['directory'])
    split = np.load(q['split'])
    train, evaluate = split[q['train']], split[q['evaluate']]
    data = pd.read_csv(directory / 'train.csv')
    feature_imputer = None
    if q.get('impute_rating'):
        feature_imputer = RatingImputer().fit(data.iloc[train])
        data = feature_imputer.transform(data)
    data = engineer(data, q.get('feature_plan'))
    target = q['target']
    classification = target == 'attack_cat'
    spec = PipelineSpec.model_validate(q['spec'])
    if q.get('alternative') == 'tabm_mini':
        from tools.reference_alternatives import MiniEnsembleEstimator
        model = MiniEnsembleEstimator(classification=classification,
            epochs=spec.hyperparameters.get('max_iter', 40), seed=spec.seed)
    elif q.get('alternative') == 'hierarchical':
        from tools.reference_alternatives import HierarchicalLGBM
        normal = sorted(data[target].unique()).index(q.get('normal_class', 'Normal'))
        model = HierarchicalLGBM(q['spec'], normal, q.get('alternative_component_iterations'))
    else:
        model = make_estimator(spec, classification, {'cpu': 4})
    if not classification and q.get('regression_eval_metric'):
        model.set_params(metric=q['regression_eval_metric'])
    columns = [c for c in data if c not in [target, 'id']]
    if q.get('building_only'):
        columns = ['State_Factor', 'building_class', 'facility_type', 'floor_area',
                   'year_built', 'energy_star_rating', 'ELEVATION'] + [c for c in data if c.startswith('ref_')]
    categorical = list(data[columns].select_dtypes('object').columns)
    x = data[columns].copy()
    history = q.get('history_signature')
    if history:
        x = x.join(history_features(data.iloc[train], data, history, target, 'Year_Factor'))
        columns = list(x.columns)
    # Vocabulary is learned from training rows only, never evaluation or test.
    vocabulary = {}
    for c in categorical:
        x[c] = x[c].fillna('__MISSING__').astype(str)
        vocabulary[c] = sorted(x.iloc[train][c].unique().tolist())
        if spec.model == 'lightgbm':
            x[c] = pd.Categorical(x[c], categories=vocabulary[c])
    y = data[target]
    if classification:
        classes = sorted(y.unique())
        encoded = pd.Series(pd.Categorical(y, categories=classes).codes, index=y.index)
    else:
        classes, encoded = None, y
    transformed = np.log1p(encoded) if spec.target_transform == 'log1p' else encoded
    target_mean, target_scale = 0., 1.
    if spec.model == 'mlp' and not classification:
        target_mean, target_scale = float(transformed.iloc[train].mean()), float(transformed.iloc[train].std())
        transformed = (transformed - target_mean) / target_scale
    preprocessor = None
    if spec.model not in {'catboost', 'lightgbm'}:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        numeric = [c for c in columns if c not in categorical]
        steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True))]
        if spec.model == 'mlp':
            steps.append(('quantile', QuantileTransformer(n_quantiles=min(500, len(train)), output_distribution='normal', random_state=42)))
        preprocessor = ColumnTransformer([('numeric', Pipeline(steps), numeric),
            ('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)], sparse_threshold=0)
        fitted_x = preprocessor.fit_transform(x.iloc[train])
        evaluated_x = preprocessor.transform(x.iloc[evaluate])
    else:
        fitted_x, evaluated_x = x.iloc[train], x.iloc[evaluate]
    stop_rows = split[q['early_stop_evaluate']] if q.get('early_stop_evaluate') else evaluate
    stopping_x = preprocessor.transform(x.iloc[stop_rows]) if preprocessor is not None else x.iloc[stop_rows]
    started = time.monotonic()
    fit_args = {}
    if classification and q.get('class_weight_exponent'):
        frequency = np.bincount(encoded.iloc[train], minlength=len(classes))
        fit_args['sample_weight'] = (frequency.max() / frequency)[encoded.iloc[train]] ** q['class_weight_exponent']
    if spec.model == 'catboost':
        fit_args['cat_features'] = categorical
    if q.get('alternative') == 'tabm_mini' and not q.get('final'):
        fit_args['eval_set'] = (stopping_x, transformed.iloc[stop_rows])
    if not q.get('final') and not q.get('fixed_iterations') and spec.model in {'catboost', 'lightgbm', 'xgboost'}:
        fit_args['eval_set'] = (stopping_x, transformed.iloc[stop_rows])
        if spec.model == 'catboost':
            fit_args.update(early_stopping_rounds=80, use_best_model=True)
        elif spec.model == 'lightgbm':
            if classification and q.get('macro_stopping'):
                frequency = np.bincount(encoded.iloc[train], minlength=len(classes))
                factors = (frequency.max() / frequency) ** .5
                def evaluate_macro(y, p):
                    return 'adjusted_macro_f1', macro_f1(y, (p * factors).argmax(axis=1), len(classes)), True
                model.set_params(metric='None')
                fit_args['eval_metric'] = evaluate_macro
            fit_args.update(callbacks=[lgb.early_stopping(80, verbose=False)])
        else:
            model.set_params(early_stopping_rounds=80)
            fit_args.update(eval_set=[fit_args['eval_set']], verbose=False)
    model.fit(fitted_x, transformed.iloc[train], **fit_args)
    if spec.model == 'catboost':
        iterations = int(model.tree_count_)
    elif spec.model == 'lightgbm':
        iterations = int(model.best_iteration_ or spec.hyperparameters['n_estimators'])
    elif spec.model == 'xgboost':
        iterations = int(model.best_iteration + 1) if not q.get('final') and not q.get('fixed_iterations') else int(spec.hyperparameters['n_estimators'])
    elif spec.model == 'mlp':
        iterations = int(model.n_iter_)
    else:
        iterations = int(spec.hyperparameters['n_estimators'])
    if q.get('final'):
        test = pd.read_csv(directory / 'test.csv')
        if feature_imputer is not None:
            test = feature_imputer.transform(test)
        test = engineer(test, q.get('feature_plan'))
        predict_x = test[[c for c in columns if c in test]].copy()
        if history:
            predict_x = predict_x.join(history_features(data.iloc[train], test, history, target, 'Year_Factor'))
        predict_x = predict_x[columns]
        for c in categorical:
            predict_x[c] = predict_x[c].fillna('__MISSING__').astype(str)
            if spec.model == 'lightgbm':
                predict_x[c] = pd.Categorical(predict_x[c], categories=vocabulary[c])
    else:
        predict_x = x.iloc[evaluate]
    if preprocessor is not None:
        predict_x = preprocessor.transform(predict_x)
    prediction = model.predict_proba(predict_x) if classification else model.predict(predict_x)
    if not classification:
        prediction = prediction * target_scale + target_mean
    if spec.target_transform == 'log1p':
        prediction = np.expm1(prediction)
    assert np.isfinite(prediction).all()
    np.save(q['prediction'], prediction)
    joblib.dump({'model': model, 'spec': q['spec'], 'columns': columns,
                 'categorical': categorical, 'vocabulary': vocabulary, 'classes': classes,
                 'history_signature': history,
                 'feature_plan': q.get('feature_plan'), 'preprocessor': preprocessor,
                 'feature_imputer': feature_imputer,
                 'target_mean': target_mean, 'target_scale': target_scale,
                 'history_reference': data.iloc[train][history + [target, 'Year_Factor']] if history else None}, q['model'])
    importance = dict(zip(columns, map(float, model.feature_importances_))) if preprocessor is None and hasattr(model, 'feature_importances_') else {}
    write(q['result'], {'status': 'completed', 'iterations': iterations,
          'seconds': time.monotonic() - started, 'classes': classes,
          'auxiliary_fits': int(feature_imputer is not None) + int(q.get('alternative') == 'hierarchical'),
          'alternative': q.get('alternative'),
          'component_iterations': getattr(model, 'component_iterations_', None),
          'epochs_run': getattr(model, 'epochs_run_', None),
          'stop_reason': getattr(model, 'stop_reason_', None),
          'feature_importance': sorted(importance.items(), key=lambda z: -z[1])[:15]})


def run_job(out, name, base, deadline):
    job = out / name
    job.mkdir()
    q = dict(base, prediction=str(job / 'prediction.npy'), model=str(job / 'model.joblib'),
             result=str(job / 'result.json'))
    write(job / 'request.json', q)
    print('START', name, flush=True)
    environment = {k: v for k, v in os.environ.items() if k in {'PATH', 'HOME', 'TMPDIR', 'SYSTEMROOT'}}
    environment.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    remaining = deadline - time.monotonic()
    if remaining <= 1:
        raise TimeoutError('Reference experiment deadline reached')
    with (job / 'worker.log').open('w') as log:
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(job / 'request.json')],
                                    stdout=log, stderr=subprocess.STDOUT, env=environment,
                                    timeout=min(base.get('timeout_seconds', 360), remaining))
        except subprocess.TimeoutExpired:
            write(job / 'failure.json', {'status': 'timeout'})
            print('TIMEOUT', name, flush=True)
            return None
    if result.returncode:
        write(job / 'failure.json', {'status': 'technical_failure', 'exit_code': result.returncode,
              'reason': 'terminated_by_signal' if result.returncode < 0 else 'process_exit'})
        print('FAILED', name, flush=True)
        return None
    r = json.loads((job / 'result.json').read_text())
    print('DONE', name, 'iterations', r['iterations'], 'seconds', round(r['seconds'], 1), flush=True)
    return r


def solve(out, kind, deadline):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, mean_squared_error, classification_report
    from sklearn.model_selection import StratifiedGroupKFold

    classification = kind == 'classification'
    directory = ROOT / 'data_for_participants' / ('Многоклассовая классификация' if classification else 'Регрессия')
    target = 'attack_cat' if classification else 'site_eui'
    data = pd.read_csv(directory / 'train.csv')
    test = pd.read_csv(directory / 'test.csv')
    out.mkdir()
    features = data.drop(columns=[target, 'id'])
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    if classification:
        development, control = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(features, data[target], groups))
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=43).split(features.iloc[development], data[target].iloc[development], groups[development]))
        fit, tune = development[a], development[b]
        assert not set(groups[fit]) & set(groups[tune])
        assert not set(groups[development]) & set(groups[control])
        classes = sorted(data[target].unique())
        for rows in (fit, tune, control):
            assert set(data[target].iloc[rows]) == set(classes)
        y_tune = pd.Categorical(data[target].iloc[tune], categories=classes).codes
        y_control = pd.Categorical(data[target].iloc[control], categories=classes).codes
        score = lambda y, p: float(f1_score(y, np.argmax(p, axis=1), average='macro', labels=list(range(len(classes))), zero_division=0))
        weights = data[target].iloc[fit].value_counts().reindex(classes).to_numpy()
        prior = weights.max() / weights
        candidates = [('lgb31', 'lightgbm', {'n_estimators': 800, 'num_leaves': 31, 'learning_rate': .05, 'reg_lambda': 3}),
                      ('lgb31_balanced', 'lightgbm', {'n_estimators': 800, 'num_leaves': 31, 'learning_rate': .05, 'reg_lambda': 3, 'class_weight': 'balanced'}),
                      ('lgb63', 'lightgbm', {'n_estimators': 800, 'num_leaves': 63, 'learning_rate': .05, 'reg_lambda': 3}),
                      ('cb6', 'catboost', {'iterations': 200, 'depth': 6, 'learning_rate': .1}),
                      ('cb6_sqrt', 'catboost', {'iterations': 200, 'depth': 6, 'learning_rate': .1, 'auto_class_weights': 'SqrtBalanced'})]
        default = np.zeros((len(tune), len(classes))); default[:, int(np.argmax(weights))] = 1
        null_score = score(y_tune, default)
    else:
        fit = np.flatnonzero(data.Year_Factor.to_numpy() < 4)
        tune = np.flatnonzero(data.Year_Factor.to_numpy() == 4)
        control = np.flatnonzero(data.Year_Factor.to_numpy() == 5)
        development = np.r_[fit, tune]
        classes, prior = None, None
        y_tune, y_control = data[target].iloc[tune].to_numpy(), data[target].iloc[control].to_numpy()
        score = lambda y, p: float(np.sqrt(mean_squared_error(y, p)))
        null_score = score(y_tune, np.full(len(tune), data[target].iloc[fit].mean()))
        candidates = [('cb6', 'catboost', {'iterations': 1200, 'depth': 6, 'learning_rate': .05}),
                      ('cb8', 'catboost', {'iterations': 1200, 'depth': 8, 'learning_rate': .05}),
                      ('lgb31', 'lightgbm', {'n_estimators': 1400, 'num_leaves': 31, 'learning_rate': .04, 'reg_lambda': 5}),
                      ('lgb63', 'lightgbm', {'n_estimators': 1400, 'num_leaves': 63, 'learning_rate': .04, 'reg_lambda': 5}),
                      ('cb8_building', 'catboost', {'iterations': 1200, 'depth': 8, 'learning_rate': .05}),
                      ('cb6_log', 'catboost', {'iterations': 1200, 'depth': 6, 'learning_rate': .05})]
    split = out / 'splits.npz'
    np.savez(split, fit=fit, tune=tune, control=control, development=development, all=np.arange(len(data)))
    conflicts = data.groupby(groups)[target].nunique() if classification else None
    write(out / 'protocol.json', {'kind': kind, 'development_reference_not_autonomous_agent': True,
          'source_hashes': {f: sha(directory / f) for f in ['train.csv', 'test.csv', 'task.md']},
          'split_hash': sha(split), 'counts': {'fit': len(fit), 'tune': len(tune), 'control': len(control)},
          'split': 'stratified identical-feature groups' if classification else 'fit years 1-3; tune year 4; control year 5; external year 6',
          'duplicate_feature_rows': int(features.duplicated().sum()),
          'conflicting_feature_groups': int((conflicts > 1).sum()) if classification else None,
          'dummy_tune_score': null_score, 'candidates': candidates,
          'control_policy': 'Select and seal on tune only; score selected recipe and first baseline once. No selection on control.',
          'limitations': 'No external labels; internal group split does not reproduce official capture-domain shift.' if classification else 'Only one untouched forward control origin; yearly transfer uncertainty remains.'})
    base = {'directory': str(directory), 'target': target, 'split': str(split), 'train': 'fit', 'evaluate': 'tune',
            'timeout_seconds': 180 if classification else 360}
    results, predictions = [], {}
    for name, model, parameters in candidates:
        spec = {'model': model, 'hyperparameters': parameters, 'seed': 42,
                'target_transform': 'log1p' if name.endswith('_log') else 'none'}
        q = dict(base, spec=spec, building_only=name.endswith('_building'))
        r = run_job(out, name, q, deadline)
        if r is None:
            continue
        p = np.load(out / name / 'prediction.npy')
        predictions[name] = p
        alphas = [0, .25, .5] if classification else [0]
        for alpha in alphas:
            calibrated = p * prior ** alpha if classification else p
            value = score(y_tune, calibrated)
            entry = {'name': name, 'alpha': alpha, 'score': value, 'iterations': r['iterations'],
                     'request': q, 'seconds': r['seconds']}
            results.append(entry)
            print('SCORE', name, alpha, value, flush=True)
        write(out / 'tuning_results.json', results)
    if not results:
        raise RuntimeError('No candidate completed')
    results.sort(key=lambda r: -r['score'] if classification else r['score'])
    selected = results[0]
    members = [selected]
    # A single prespecified equal blend of the two best distinct model recipes.
    alternative = next((r for r in results if r['name'] != selected['name']), None)
    if alternative:
        def tune_prediction(r):
            p = predictions[r['name']].copy()
            if classification:
                p *= prior ** r['alpha']
                p /= p.sum(axis=1, keepdims=True)
            return p
        blend_score = score(y_tune, (tune_prediction(selected) + tune_prediction(alternative)) / 2)
        if (blend_score > selected['score']) if classification else (blend_score < selected['score']):
            members = [selected, alternative]
    selection = {'members': members, 'weights': [1 / len(members)] * len(members),
                 'selected_before_control': True, 'sealed_at_unix': time.time(),
                 'tune_score': blend_score if len(members) == 2 else selected['score']}
    write(out / 'selection.json', selection)
    print('SEALED', kind, [(r['name'], r['alpha']) for r in members], selection['tune_score'], flush=True)
    # Control fitting never passes control labels to early stopping.
    control_predictions = {}
    baseline = next((r for r in results if r['name'] == candidates[0][0] and r['alpha'] == 0),
                    next(r for r in results if r['alpha'] == 0))
    for r in {r['name']: r for r in members + [baseline]}.values():
        q = dict(r['request'])
        parameters = dict(q['spec']['hyperparameters'])
        parameters['iterations' if q['spec']['model'] == 'catboost' else 'n_estimators'] = r['iterations']
        q['spec'] = dict(q['spec'], hyperparameters=parameters)
        q.update(train='development', evaluate='control', final=True)
        # Fixed iterations, predict test here; save control predictions separately using fitted model below.
        fit_result = run_job(out, r['name'] + '_control_fit', q, deadline)
        if fit_result is None:
            raise RuntimeError('Control fit failed; sealed selection preserved')
        control_predictions[r['name']] = predict_saved(out / (r['name'] + '_control_fit') / 'model.joblib', data.iloc[control])
    def combine(preds, counts):
        ps = []
        local_prior = counts.max() / counts if classification else None
        for r in members:
            p = preds[r['name']].copy()
            if classification:
                p *= local_prior ** r['alpha']
                p /= p.sum(axis=1, keepdims=True)
            ps.append(p)
        return sum(ps) / len(ps)
    counts = data[target].iloc[development].value_counts().reindex(classes).to_numpy() if classification else None
    cp = combine(control_predictions, counts)
    control_score = score(y_control, cp)
    baseline_score = score(y_control, control_predictions[baseline['name']])
    control_report = {'selected_score': control_score, 'baseline_score': baseline_score, 'baseline_name': baseline['name'],
                      'selection_changed_after_control': False}
    if classification:
        control_report['per_class'] = classification_report(y_control, cp.argmax(axis=1), labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
    else:
        error = cp - y_control
        control_report.update(bias=float(error.mean()), mae=float(np.abs(error).mean()))
    write(out / 'control_report.json', control_report)
    print('CONTROL', kind, control_report['selected_score'], 'baseline', baseline_score, flush=True)
    # Final training on every available labeled row, with the already frozen recipe.
    final = {}
    for r in members:
        q = json.loads((out / (r['name'] + '_control_fit') / 'request.json').read_text())
        q.update(train='all', evaluate='control', final=True)
        fitted = run_job(out, r['name'] + '_final', q, deadline)
        if fitted is None:
            raise RuntimeError('Final fit failed')
        final[r['name']] = np.load(out / (r['name'] + '_final') / 'prediction.npy')
    counts = data[target].value_counts().reindex(classes).to_numpy() if classification else None
    prediction = combine(final, counts)
    values = np.asarray(classes)[prediction.argmax(axis=1)] if classification else prediction
    submission = pd.DataFrame({'id': test.id, target: values})
    assert submission.columns.tolist() == ['id', target]
    assert len(submission) == len(test) and submission.id.is_unique
    assert submission.id.equals(test.id) and not submission.isna().any().any()
    if classification:
        assert set(values) <= set(classes)
    else:
        assert np.isfinite(values).all()
    submission.to_csv(out / 'submission.csv', index=False)
    write(out / 'summary.json', {'status': 'completed', 'target': target, 'metric': 'macro_f1' if classification else 'rmse',
          'tune_score': selection['tune_score'], 'control': control_report,
          'submission_rows': len(submission), 'submission_sha256': sha(out / 'submission.csv'),
          'members': [{'name': r['name'], 'iterations': r['iterations'], 'alpha': r['alpha']} for r in members],
          'external_quality': 'unknown: test labels unavailable', 'api_calls': 0})


def write_report(out):
    """Create the readable evidence alongside each complete reference series."""
    lines = ['# Participant reference experiments', '',
             'Offline development reference; this is not a test of autonomous Qwen decisions.',
             'Test labels are unavailable. All scores below are internal validation scores.', '']
    for kind in ['regression', 'classification']:
        directory = Path(out) / kind
        summary = json.loads((directory / 'summary.json').read_text())
        protocol = json.loads((directory / 'protocol.json').read_text())
        results = json.loads((directory / 'tuning_results.json').read_text())
        export_inference(directory)
        lines += [f'## {kind}', '', f"Split: {protocol['split']}.",
                  f"Counts: {protocol['counts']}.", '', '| Candidate | Class correction exponent | Tuning score | Iterations |',
                  '|---|---:|---:|---:|']
        lines += [f"| {r['name']} | {r['alpha']} | {r['score']:.6f} | {r['iterations']} |" for r in results]
        lines += ['', f"Selected: {summary['members']}.",
                  f"Control score: {summary['control']['selected_score']:.6f}; first baseline: {summary['control']['baseline_score']:.6f}.",
                  f"Output: {summary['submission_rows']} rows; SHA-256 `{summary['submission_sha256']}`.",
                  f"Limitation: {protocol['limitations']}", '']
    lines += ['## Verification boundaries', '',
              '- Input and split hashes, random seeds, frozen recipes and fit logs are saved.',
              '- Identical feature groups are disjoint in classification; temporal ordering is preserved in regression.',
              '- Control labels are not used for early stopping or recipe selection in this initial study.',
              '- CSV row count, identifier order, finite values and allowed labels are checked.',
              '- No API requests, downloaded data, weights or packages are used by the experiment.',
              '- This does not establish contest container readiness, autonomous decision quality, or external-test quality.', '']
    (Path(out) / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def export_inference(run):
    """Freeze all inference inputs so prediction needs no training labels."""
    import pandas as pd

    run = Path(run)
    selection = json.loads((run / 'selection.json').read_text())
    target = json.loads((run / 'summary.json').read_text())['target']
    classification = target == 'attack_cat'
    training = pd.read_csv(Path(selection['members'][0]['request']['directory']) / 'train.csv', usecols=[target])
    classes = sorted(training[target].unique()) if classification else None
    counts = training[target].value_counts().reindex(classes).to_numpy() if classification else None
    manifest = {'target': target, 'classes': classes, 'members': []}
    for r in selection['members']:
        manifest['members'].append({'model': f"{r['name']}_final/model.joblib",
            'class_factors': ((counts.max() / counts) ** r['alpha']).tolist() if classification else None})
    write(run / 'inference.json', manifest)


def predict_solution(run, frame):
    import numpy as np
    import pandas as pd

    run = Path(run)
    manifest = json.loads((run / 'inference.json').read_text())
    predictions = []
    for member in manifest['members']:
        p = predict_saved(run / member['model'], frame)
        if manifest['classes']:
            p *= np.asarray(member['class_factors'])
            p /= p.sum(axis=1, keepdims=True)
        predictions.append(p)
    p = sum(predictions) / len(predictions)
    if manifest['classes']:
        p = np.asarray(manifest['classes'])[p.argmax(axis=1)]
    assert len(p) == len(frame) and frame.id.is_unique
    result = pd.DataFrame({'id': frame.id, manifest['target']: p})
    assert not result.isna().any().any()
    if not manifest['classes']:
        assert np.isfinite(p).all()
    return result


def prepare_saved(path, frame):
    import joblib
    import numpy as np
    import pandas as pd
    from tools.reference_features import engineer
    bundle = joblib.load(path)
    if bundle.get('feature_imputer') is not None:
        frame = bundle['feature_imputer'].transform(frame)
    frame = engineer(frame, bundle.get('feature_plan'))
    x = frame[[c for c in bundle['columns'] if c in frame]].copy()
    if bundle.get('history_signature'):
        x = x.join(history_features(bundle['history_reference'], frame,
            bundle['history_signature'], 'site_eui', 'Year_Factor'))
    x = x[bundle['columns']]
    for c in bundle['categorical']:
        x[c] = x[c].fillna('__MISSING__').astype(str)
        if bundle['spec']['model'] == 'lightgbm':
            x[c] = pd.Categorical(x[c], categories=bundle['vocabulary'][c])
    if bundle.get('preprocessor') is not None:
        x = bundle['preprocessor'].transform(x)
    return bundle, x


def predict_saved(path, frame):
    import numpy as np
    bundle, x = prepare_saved(path, frame)
    model = bundle['model']
    p = model.predict_proba(x) if bundle['classes'] else model.predict(x)
    if not bundle['classes']:
        p = p * bundle.get('target_scale', 1.) + bundle.get('target_mean', 0.)
    return np.expm1(p) if bundle['spec'].get('target_transform') == 'log1p' else p


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=int, default=3000)
    parser.add_argument('--predict-run', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--prediction', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.predict_run:
        import pandas as pd
        if not args.data or not args.prediction:
            parser.error('--data and --prediction are required for inference')
        predict_solution(args.predict_run, pd.read_csv(args.data)).to_csv(args.prediction, index=False)
    else:
        if not args.output:
            parser.error('--output required')
        out = args.output.resolve()
        out.mkdir(parents=True, exist_ok=False)
        deadline = time.monotonic() + args.seconds
        for kind in ['regression', 'classification']:
            solve(out / kind, kind, deadline)
        write_report(out)
