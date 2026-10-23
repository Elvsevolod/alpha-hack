"""Keep complete-fit I/O distinct from sampled model work; preserve safety scaling."""
import pytest

from automl_agent.completion import measured_cost


def summary():
    return {'rows': 400, 'execution_phases': {'loaded_fit_rows': 4000,
        'fit_seconds': 2., 'save_load_seconds': .2, 'predict_and_verify_seconds': .3,
        'load_and_construct_seconds': .5, 'controller_staging_seconds': .4}}


def test_sampled_model_work_scales_but_complete_fit_io_uses_its_actual_rows():
    # wall5 = fit2 + save/predict.5 + full-fit I/O.9 + startup1.6.
    assert measured_cost(summary(), 5., 4000, 4000) == pytest.approx((2.5 * 10 + .9 + 1.6) * 1.25 + .2)
    assert measured_cost(summary(), 5., 8000, 4000) == pytest.approx((2.5 * 20 + .9 * 2 + 1.6) * 1.25 + .2)


def test_save_and_prediction_work_keep_the_previous_conservative_row_multiplier():
    a = summary(); b = summary()
    a['execution_phases']['predict_and_verify_seconds'] = .1
    b['execution_phases']['predict_and_verify_seconds'] = 1.1
    # An extra second of measured prediction is NOT fixed when model rows grow100x.
    assert measured_cost(b, 6., 40000, 400) - measured_cost(a, 5., 40000, 400) == pytest.approx(100 * 1.25)
    a = summary(); b = summary()
    b['execution_phases']['save_load_seconds'] += 1
    assert measured_cost(b, 6., 40000, 400) - measured_cost(a, 5., 40000, 400) == pytest.approx(100 * 1.25)


def test_selected_automl_fit_excludes_search_and_uses_selected_rows():
    s = summary(); s.update(seconds=100, automatic_pilot_seconds=3,
        budget={'selected_fit_actual_seconds': 2., 'selected_fit_rows': 400})
    assert measured_cost(s, 105, 4000, 4000) == pytest.approx((2.5 * 10 + .9 + .6) * 1.25 + .2)


def test_multioutput_selected_fit_sums_targets():
    s = summary(); s.update(seconds=100, budget={'targets': {
        'a': {'selected_fit_actual_seconds': 2.}, 'b': {'selected_fit_actual_seconds': 3.}}})
    assert measured_cost(s, 105, 4000, 4000) == pytest.approx((5.5 * 10 + .9 + 3.6) * 1.25 + .2)


def test_completed_fit_does_not_charge_automatic_pilot_again():
    s = summary(); s['automatic_pilot_seconds'] = 3
    assert measured_cost(s, 8, 4000, 4000) == measured_cost(summary(), 5, 4000, 4000)


@pytest.mark.parametrize('s,seconds', [({}, 4), ({'rows': 400}, 2),
    ({'rows': 0}, .1), ({'rows': 400, 'execution_phases': {'fit_seconds': .2}}, 2)])
def test_no_complete_phase_measurement_preserves_legacy_fallback(s, seconds):
    assert measured_cost(s, seconds, 4000, 400) == pytest.approx(
        max(.2, seconds) * max(1, 4000 / max(1, s.get('rows', 400))) * 1.25 + .2)


def test_legacy_selected_fit_without_row_basis_preserves_previous_formula():
    s = {'rows': 400, 'seconds': 100, 'budget': {'selected_fit_actual_seconds': 2.}}
    assert measured_cost(s, 105, 4000, 4000) == pytest.approx(7 * 10 * 1.25 + .2)


def test_smaller_refit_does_not_discount_observed_model_or_io_work():
    assert measured_cost(summary(), 5, 10, 400) == pytest.approx(5 * 1.25 + .2)
