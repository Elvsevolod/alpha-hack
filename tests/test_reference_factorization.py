import joblib
import json
import numpy as np
import pandas as pd
import pytest

from tools.reference_factorization import rating_factor, RatingLGBM
from tools.solve_participant_examples import worker, predict_saved


def test_rating_factor_is_target_free_decreasing_and_missing_safe():
    factor = rating_factor([0., 1., 25., 75., 100., np.nan], 4.)
    assert np.isfinite(factor).all() and (factor > 0).all()
    assert (np.diff(factor[:5]) <= 0).all()
    assert (np.diff(factor[1:5]) < 0).all()
    assert factor[-1] == 1.
    with pytest.raises(ValueError):
        rating_factor([50], 0.)
    with pytest.raises(ValueError):
        rating_factor([np.inf], 4.)


def test_factorization_squared_error_matches_original_units():
    factor = rating_factor([5., 50., 95., np.nan], 16.)
    y = np.array([200., 80., 30., 120.])
    base = np.array([100., 70., 60., 90.])
    np.testing.assert_allclose(factor**2 * (y/factor-base)**2, (y-factor*base)**2)


def test_factorized_model_roundtrip_and_no_validation_labels(tmp_path):
    x = pd.DataFrame({'energy_star_rating': np.tile([np.nan, 5., 50., 95.], 50),
                      'area': np.arange(200) % 7})
    spec = dict(model='lightgbm', hyperparameters=dict(n_estimators=5, num_leaves=7), seed=42)
    model = RatingLGBM(spec, 4.)
    model.fit(x, 10 + 100 * rating_factor(x.energy_star_rating, 4.))
    p = model.predict(x)
    assert np.isfinite(p).all()
    joblib.dump(model, tmp_path/'model.joblib')
    np.testing.assert_allclose(joblib.load(tmp_path/'model.joblib').predict(x), p)
    with pytest.raises(ValueError):
        model.fit(x, p, eval_set=(x, p))


def test_factorization_shared_worker_roundtrip_and_label_boundary(tmp_path):
    frame = pd.DataFrame({'id': np.arange(160), 'State_Factor': ['S']*160,
        'building_class': ['Commercial']*160, 'facility_type': ['Office']*160,
        'floor_area': 100 + np.arange(160), 'year_built': [1980.]*160,
        'ELEVATION': [10.]*160, 'energy_star_rating': np.tile([np.nan, 5., 50., 95.], 40)})
    frame['site_eui'] = 100 * rating_factor(frame.energy_star_rating, 4.) + frame.floor_area / 10
    frame.to_csv(tmp_path/'train.csv', index=False)
    frame.drop(columns='site_eui').iloc[-20:].to_csv(tmp_path/'test.csv', index=False)
    np.savez(tmp_path/'split.npz', fit=np.arange(120), tune=np.arange(120, 160))
    q = dict(directory=str(tmp_path), target='site_eui', split=str(tmp_path/'split.npz'),
        train='fit', evaluate='tune', spec=dict(model='lightgbm', hyperparameters=dict(n_estimators=5)),
        alternative='rating', rating_shape=4., feature_plan='energy_building', building_only=True,
        fixed_iterations=True, prediction=str(tmp_path/'prediction.npy'), model=str(tmp_path/'model.joblib'),
        result=str(tmp_path/'result.json'))
    request = tmp_path/'request.json'; request.write_text(json.dumps(q))
    worker(request)
    np.testing.assert_allclose(predict_saved(tmp_path/'model.joblib', frame.iloc[120:]), np.load(tmp_path/'prediction.npy'))
    altered = frame.iloc[120:].assign(site_eui=-1e9)
    np.testing.assert_allclose(predict_saved(tmp_path/'model.joblib', altered), np.load(tmp_path/'prediction.npy'))
