import joblib
import numpy as np
import pandas as pd
import pytest

from tools.reference_specialists import PatternLGBM
from tools.reference_features import engineer
from tools.reduction_reference_study import candidates
from automl_agent.contracts import PipelineSpec
from automl_agent.pipelines import make_estimator


def test_pattern_routes_on_inputs_and_falls_back_with_small_support(tmp_path):
    x = pd.DataFrame({'ref_rating_missing': [0.] * 160 + [1.] * 80,
                      'number': np.arange(240) % 13})
    y = np.where(x.ref_rating_missing == 0, x.number, 100 + x.number)
    spec = dict(model='lightgbm', hyperparameters=dict(n_estimators=5, num_leaves=7), seed=42)
    model = PatternLGBM(spec).fit(x, y)
    assert model.fit_count_ == 2
    q = pd.DataFrame({'ref_rating_missing': [0., 1., np.nan], 'number': [3, 3, 3]})
    p = model.predict(q)
    assert p[0] == pytest.approx(model.models_[0.].predict(q.iloc[[0]])[0])
    np.testing.assert_allclose(p[1:], model.global_.predict(q.iloc[1:]))
    joblib.dump(model, tmp_path / 'model.joblib')
    np.testing.assert_allclose(joblib.load(tmp_path / 'model.joblib').predict(q), p)
    with pytest.raises(ValueError):
        model.fit(x, y, eval_set=(x, y))


def test_energy_interactions_preserve_old_features_and_ignore_labels():
    frame = pd.DataFrame({'State_Factor': ['S', 'T'], 'facility_type': ['F', 'G'],
        'floor_area': [100., 10000.], 'year_built': [0., 1985.],
        'energy_star_rating': [np.nan, 80.], 'site_eui': [20., 100.], 'id': [1, 2]})
    old, new = engineer(frame, 'energy_building'), engineer(frame, 'energy_interactions')
    pd.testing.assert_frame_equal(new[old.columns], old)
    altered = engineer(frame.assign(site_eui=[9999., -1.], id=[555, 666]), 'energy_interactions')
    cols = [c for c in new if c.startswith('ref_')]
    pd.testing.assert_frame_equal(new[cols], altered[cols])
    assert new.ref_facility_rating_bin.tolist() == ['F|<NA>', 'G|8']
    assert new.ref_facility_vintage.tolist() == ['F|<NA>', 'G|99']


def test_ova_is_supported_and_returns_all_classes():
    config = next(c for c in candidates(True) if c['name'] == 'ova')
    spec = PipelineSpec.model_validate(config['spec'])
    spec = spec.model_copy(update={'hyperparameters': dict(spec.hyperparameters, n_estimators=5)})
    x = pd.DataFrame({'x': np.arange(120) % 7})
    model = make_estimator(spec, True, {'cpu': 2}).fit(x, np.arange(120) % 3)
    p = model.predict_proba(x)
    assert p.shape == (120, 3) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
