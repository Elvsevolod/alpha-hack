import numpy as np
import pandas as pd
import pytest

from tools.reference_features import engineer
from tools.targeted_decisions import (normalize, unweight, temper, nll,
    fit_temperature, join_hierarchy, refit_iterations)
from tools.literature_reference_study import validate_locked_recipes
from tools.targeted_error_report import regression_segments


def test_semantic_network_features_are_target_and_id_independent():
    frame = pd.DataFrame({c: [0., 4.] for c in ['sbytes', 'spkts', 'dbytes',
        'dpkts', 'synack', 'tcprtt', 'sttl', 'dttl', 'ackdat', 'swin', 'dwin', 'dur', 'rate']})
    frame['proto'], frame['service'], frame['state'] = ['tcp', 'udp'], ['-', 'dns'], ['FIN', 'INT']
    frame['id'], frame['attack_cat'] = [1, 2], ['Normal', 'Worms']
    legacy = engineer(frame, 'network')
    first = engineer(frame, 'network_semantic')
    pd.testing.assert_frame_equal(first[legacy.columns], legacy)
    altered = frame.assign(id=[99, 88], attack_cat=['DoS', 'Generic'])
    features = [c for c in first if c.startswith('ref_')]
    pd.testing.assert_frame_equal(first[features], engineer(altered, 'network_semantic')[features])
    assert first.ref_no_reply.tolist() == [1., 0.]
    assert first.ref_service_unspecified.tolist() == [1., 0.]
    assert first.ref_proto_service.tolist() == ['tcp|-', 'udp|dns']
    assert not np.isinf(first[features].select_dtypes('number')).any().any()
    assert np.isnan(first.ref_byte_ratio.iloc[0])


def test_unweight_recovers_known_weighted_posterior():
    true = np.array([[.6, .3, .1], [.1, .2, .7]])
    counts = np.array([100, 20, 10])
    weighted = normalize(true / counts)
    np.testing.assert_allclose(unweight(weighted, counts, 1), true)
    with pytest.raises(ValueError):
        unweight(true, [1, 0, 3], 1)


def test_hierarchy_preserves_gate_mass_and_class_position():
    p = join_hierarchy(np.array([[.7, .3]]), np.array([[.2, .8]]), 1)
    np.testing.assert_allclose(p, [[.06, .7, .24]])
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_temperature_optimizes_only_given_calibration_labels():
    p = np.tile([.99, .01], (10, 1))
    y = np.array([0] * 7 + [1] * 3)
    original = p.copy()
    t = fit_temperature(p, y)
    assert .5 <= t <= 2
    assert nll(temper(p, t), y) < nll(p, y)
    assert fit_temperature(p, y) == t
    np.testing.assert_array_equal(p, original)
    with pytest.raises(ValueError):
        temper(p, 0)


def test_refit_rule_uses_only_earlier_sample_counts_and_is_capped():
    assert refit_iterations('latest_inner', 100, 100, 400) == 100
    assert refit_iterations('latest_inner_sqrt_rows', 100, 100, 400) == 200
    assert refit_iterations('latest_inner_sqrt_rows', 2000, 100, 400) == 2500
    with pytest.raises(ValueError):
        refit_iterations('latest_inner', 100, 400, 400)
    with pytest.raises(ValueError):
        refit_iterations('unknown', 100, 100, 400)


@pytest.mark.parametrize('p', [[[np.nan, 1]], [[-1, 2]], [[0, 0]]])
def test_invalid_probabilities_fail_explicitly(p):
    with pytest.raises(ValueError):
        normalize(p)


def test_controlled_comparison_fails_closed_for_missing_models():
    recipe = dict(names=['old', 'new'], weights=[.5, .5])
    validate_locked_recipes([recipe], {'old': {}, 'new': {}})
    with pytest.raises(ValueError, match='incomplete'):
        validate_locked_recipes([recipe], {'old': {}})
    with pytest.raises(ValueError, match='sum to one'):
        validate_locked_recipes([dict(recipe, weights=[.6, .6])], {'old': {}, 'new': {}})
    with pytest.raises(ValueError):
        validate_locked_recipes([], {})


def test_error_segment_arithmetic_and_unknown_small_segments():
    frame = pd.DataFrame({'energy_star_rating': [np.nan, 80],
                          'building_class': ['A', 'B'], 'facility_type': ['X', 'Y']})
    segments = {s['segment']: s for s in regression_segments(frame, [10., 100.], [12., 90.], 50.)}
    assert segments['all']['rmse'] == pytest.approx(np.sqrt(52))
    assert segments['all']['bias'] == -4
    assert segments['rating_missing']['count'] == 1
    assert segments['target_above_earlier_q95']['squared_error_share'] == pytest.approx(100/104)
