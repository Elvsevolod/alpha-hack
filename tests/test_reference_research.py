import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from tools.reference_features import engineer, fit_offsets, macro_f1, RatingImputer
from tools.solve_participant_examples import predict_saved, worker


def test_macro_f1_includes_absent_classes():
    y, prediction = np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])
    assert macro_f1(y, prediction, 3) == pytest.approx(f1_score(y, prediction, labels=[0, 1, 2], average='macro', zero_division=0))


def test_offsets_bounded_and_deterministic():
    p = np.array([[.6, .4], [.7, .3], [.8, .2], [.9, .1]])
    y = np.array([1, 1, 0, 0])
    a = fit_offsets(p, y)
    np.testing.assert_array_equal(a, fit_offsets(p, y))
    assert (np.abs(a) <= 1).all()
    assert macro_f1(y, (np.log(p) + a).argmax(axis=1), 2) >= macro_f1(y, p.argmax(axis=1), 2)


@pytest.mark.parametrize('classification,model', [(False, 'xgboost'), (True, 'xgboost'), (False, 'mlp'), (True, 'lightgbm'), (False, 'extra_trees')])
def test_research_worker_saved_predictions(classification, model, tmp_path):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({'id': np.arange(150), 'number': rng.normal(size=150), 'category': np.tile(['a', 'b', 'c'], 50)})
    target = 'attack_cat' if classification else 'site_eui'
    frame[target] = np.tile(['A', 'B', 'C'], 50) if classification else 3 * frame.number + 20
    frame.to_csv(tmp_path / 'train.csv', index=False)
    frame.drop(columns=target).iloc[-20:].to_csv(tmp_path / 'test.csv', index=False)
    np.savez(tmp_path / 'split.npz', fit=np.arange(100), tune=np.arange(100, 150))
    parameters = {'max_iter': 5, 'hidden_layer_sizes': [8]} if model == 'mlp' else {'n_estimators': 5}
    q = {'directory': str(tmp_path), 'target': target, 'split': str(tmp_path / 'split.npz'),
         'train': 'fit', 'evaluate': 'tune', 'spec': {'model': model, 'hyperparameters': parameters},
         'macro_stopping': True, 'prediction': str(tmp_path / 'prediction.npy'),
         'model': str(tmp_path / 'model.joblib'), 'result': str(tmp_path / 'result.json')}
    request = tmp_path / 'request.json'
    request.write_text(json.dumps(q))
    worker(request)
    np.testing.assert_allclose(predict_saved(tmp_path / 'model.joblib', frame.iloc[100:]), np.load(tmp_path / 'prediction.npy'))
    # Changing supplied query labels cannot change any computed prediction.
    altered = frame.iloc[100:].copy()
    altered[target] = 'unseen' if classification else -1e9
    np.testing.assert_allclose(predict_saved(tmp_path / 'model.joblib', altered), np.load(tmp_path / 'prediction.npy'))


def test_energy_features_are_target_free():
    frame = pd.DataFrame({'State_Factor': ['S'], 'facility_type': ['F'], 'floor_area': [100.],
                          'year_built': [0.], 'energy_star_rating': [90.], 'site_eui': [80.]})
    first = engineer(frame, 'energy_building').drop(columns='site_eui')
    frame.site_eui = -1000
    pd.testing.assert_frame_equal(first, engineer(frame, 'energy_building').drop(columns='site_eui'))
    assert pd.isna(first.year_built.iloc[0])


def test_rating_imputer_preserves_observed_values_and_ignores_ml_target():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({'State_Factor': ['S'] * 80, 'building_class': ['C'] * 80,
        'facility_type': ['F'] * 80, 'floor_area': rng.uniform(10, 100, 80),
        'year_built': [2000.] * 80, 'ELEVATION': [50.] * 80,
        'energy_star_rating': rng.uniform(1, 100, 80), 'site_eui': rng.uniform(1, 1000, 80)})
    frame.loc[::5, 'energy_star_rating'] = np.nan
    model = RatingImputer().fit(frame)
    changed = frame.copy()
    changed['site_eui'] = -1e9
    first, second = model.transform(frame), model.transform(changed)
    pd.testing.assert_series_equal(first.energy_star_rating, second.energy_star_rating)
    mask = frame.energy_star_rating.notna()
    pd.testing.assert_series_equal(first.energy_star_rating[mask], frame.energy_star_rating[mask])
    assert first.energy_star_rating.notna().all()
    assert first.ref_rating_was_missing.sum() == 16


def test_separate_early_stopping_subset_reaches_estimator(monkeypatch, tmp_path):
    import automl_agent.pipelines
    captured = []
    original = automl_agent.pipelines.make_estimator
    def make(spec, classification, context):
        model = original(spec, classification, context)
        fit = model.fit
        def record(x, y, **kwargs):
            captured.append(kwargs['eval_set'][0].index.tolist())
            return fit(x, y, **kwargs)
        model.fit = record
        return model
    monkeypatch.setattr(automl_agent.pipelines, 'make_estimator', make)
    # Don't serialize the local spy; this test inspects the actual fit boundary.
    monkeypatch.setattr('joblib.dump', lambda *a, **k: None)
    frame = pd.DataFrame({'id': np.arange(90), 'x': np.arange(90), 'attack_cat': np.tile(['A', 'B', 'C'], 30)})
    frame.to_csv(tmp_path / 'train.csv', index=False)
    np.savez(tmp_path / 'split.npz', fit=np.arange(60), tune=np.arange(60, 90), calibration=np.arange(60, 70))
    q = {'directory': str(tmp_path), 'target': 'attack_cat', 'split': str(tmp_path / 'split.npz'),
         'train': 'fit', 'evaluate': 'tune', 'early_stop_evaluate': 'calibration',
         'spec': {'model': 'lightgbm', 'hyperparameters': {'n_estimators': 5}},
         'macro_stopping': True, 'prediction': str(tmp_path / 'prediction.npy'),
         'model': str(tmp_path / 'model.joblib'), 'result': str(tmp_path / 'result.json')}
    request = tmp_path / 'request.json'
    request.write_text(json.dumps(q))
    worker(request)
    assert captured == [list(range(60, 70))]


def test_paired_cluster_interval_identical_predictions_has_zero_delta():
    from tools.literature_reference_study import paired_group_interval
    y = np.array([0, 0, 1, 1, 0, 1])
    prediction = np.array([0, 0, 1, 0, 1, 1])
    report = paired_group_interval(y, prediction, prediction, np.array([0, 0, 1, 2, 3, 4]), 2, resamples=20)
    assert report['groups'] == 5
    assert report['delta_macro_f1'] == 0
    assert report['conditional_percentile_95'] == [0, 0]


@pytest.mark.parametrize('changed', ['worker_source_sha256', 'dependencies'])
def test_prediction_reuse_rejects_changed_runtime_before_any_fit(monkeypatch, tmp_path, changed):
    import sys
    import time
    import tools.literature_reference_study as study

    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'sha', lambda path: 'fixed')
    monkeypatch.setattr(study, 'version', lambda name: 'test-version')
    directory = tmp_path / 'data_for_participants' / 'Многоклассовая классификация'
    directory.mkdir(parents=True)
    frame = pd.DataFrame({'id': np.arange(60), 'feature': np.arange(60), 'attack_cat': np.tile(['A', 'B', 'C'], 20)})
    frame.to_csv(directory / 'train.csv', index=False)
    frame.drop(columns='attack_cat').iloc[:3].to_csv(directory / 'test.csv', index=False)
    original = tmp_path / 'outputs/participant_reference_20261025/classification'
    original.mkdir(parents=True)
    np.savez(original / 'splits.npz', **study.reference_splits(frame, True))
    evidence = tmp_path / 'docs/reports/examples/2026-10-25-reference-evidence.json'
    evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps({'cases': {'classification': {
        'protocol': {'source_hashes': {name: 'fixed' for name in ['train.csv', 'test.csv', 'task.md']}},
        'summary': {}}}}))
    prior = tmp_path / 'prior/classification'
    prior.mkdir(parents=True)
    workers = {name: 'fixed' for name in ['tools/solve_participant_examples.py', 'tools/reference_features.py', 'automl_agent/pipelines.py']}
    packages = {name: 'test-version' for name in ['numpy', 'pandas', 'scikit-learn', 'catboost', 'lightgbm', 'xgboost']}
    packages['python'] = sys.version.split()[0]
    protocol = {'source_hashes': {name: 'fixed' for name in ['train.csv', 'test.csv', 'task.md']},
                'worker_source_sha256': workers, 'dependencies': packages}
    protocol[changed] = {'changed': True}
    (prior / 'protocol.json').write_text(json.dumps(protocol))
    output = tmp_path / 'new'
    output.mkdir()
    with pytest.raises(ValueError, match='worker source|dependencies'):
        study.run_case(output, 'classification', time.monotonic() + 10, config_override=[], reuse_root=tmp_path / 'prior')
    assert not list(output.rglob('request.json'))


def test_recreated_temporal_partitions_preserve_forward_boundary():
    from tools.literature_reference_study import reference_splits
    result = reference_splits(pd.DataFrame({'Year_Factor': [1, 2, 3, 4, 5]}), False)
    np.testing.assert_array_equal(result['fit'], [0, 1, 2])
    np.testing.assert_array_equal(result['tune'], [3])
    np.testing.assert_array_equal(result['control'], [4])
    np.testing.assert_array_equal(result['development'], [0, 1, 2, 3])
