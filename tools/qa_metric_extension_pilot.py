"""Staged section-4 metric contract; frozen A/B archives do not import this file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn import metrics


class MetricContractError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


POLICIES = {
    'mape': {
        'maximize': False, 'scale': 'fraction',
        'zero_denominator': 'max(abs(y_true), float64 epsilon)',
        'negative_targets': 'absolute denominator; original target scale',
        'aggregation': 'mean rows per target, then uniform or declared target weights',
    },
    'balanced_accuracy': {
        'maximize': True, 'scale': 'fraction', 'adjusted': False,
        'classes': 'mean recall over classes observed in y_true; reject undeclared labels',
        'missing_declared_class': 'excluded from that target mean, as in sklearn 1.5.2',
        'aggregation': 'uniform or declared target weights; not pooled across targets',
    },
}


def score_extension(name, actual, predicted, *, target_weights=None, classes=None):
    """Trusted evaluator only: labels are not passed to generated LLM scorers."""
    if name not in POLICIES:
        raise MetricContractError('unsupported_metric', f'Unsupported metric: {name}')
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim == 1:
        actual = actual[:, None]
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    if actual.ndim != 2 or actual.shape != predicted.shape or not all(actual.shape):
        raise MetricContractError('invalid_metric_input', 'Expected equal nonempty row/target arrays')
    weights = None if target_weights is None else np.asarray(target_weights, dtype=float)
    if weights is not None and (weights.shape != (actual.shape[1],) or
            not np.isfinite(weights).all() or (weights < 0).any() or
            not np.isfinite(weights.sum()) or weights.sum() <= 0):
        raise MetricContractError('invalid_metric_input', 'Invalid target weights')
    if name == 'balanced_accuracy' and (classes is None or len(classes) != actual.shape[1]):
        raise MetricContractError('invalid_metric_input', 'Declare the label space for every target')
    values = []
    for index in range(actual.shape[1]):
        truth, estimate = actual[:, index], predicted[:, index]
        if name == 'mape':
            value = metrics.mean_absolute_percentage_error(
                truth.astype(np.float64), estimate.astype(np.float64))
        else:
            allowed = set(classes[index])
            if not allowed or not set(truth) <= allowed or not set(estimate) <= allowed:
                raise MetricContractError('invalid_metric_input', 'Undeclared class label')
            value = metrics.balanced_accuracy_score(truth, estimate, adjusted=False)
        if not np.isfinite(value):
            raise MetricContractError('undefined_metric', 'Metric is not finite on this split')
        values.append(float(value))
    return float(np.average(values, weights=weights))


def pilot():
    """Independent arithmetic goldens, including zero targets and target weighting."""
    cases = []

    def golden(name, expected, *args, **kwargs):
        measured = score_extension(*args, **kwargs)
        assert np.isclose(measured, expected, rtol=1e-13, atol=1e-15), (name, measured, expected)
        cases.append({'name': name, 'passed': True, 'value': measured, 'expected': expected})

    def rejected(name, code, *args, **kwargs):
        try:
            score_extension(*args, **kwargs)
        except MetricContractError as error:
            assert error.code == code
            cases.append({'name': name, 'passed': True, 'code': error.code})
        else:
            raise AssertionError(name + ': invalid input accepted')

    golden('mape_fraction_not_percent', .25, 'mape', [2, 4], [1, 4])
    golden('mape_negative_target', .5, 'mape', [-2, -4], [-1, -6])
    golden('mape_zero_exact', 0., 'mape', [0, 0], [0, 0])
    golden('mape_zero_epsilon', 1 / (2 * np.finfo(np.float64).eps), 'mape', [0, 1], [1, 1])
    golden('mape_near_zero_epsilon', 1e-20 / np.finfo(np.float64).eps,
           'mape', [1e-20], [2e-20])
    golden('mape_multioutput_uniform', .75, 'mape', [[2, 4], [4, 8]], [[1, 0], [2, 0]])
    golden('mape_multioutput_weighted', .875, 'mape', [[2, 4], [4, 8]],
           [[1, 0], [2, 0]], target_weights=[1, 3])
    golden('balanced_binary', .75, 'balanced_accuracy', [0, 0, 1, 1],
           [0, 1, 1, 1], classes=[[0, 1]])
    golden('balanced_imbalance_not_accuracy', .5, 'balanced_accuracy', [0, 0, 0, 1],
           [0, 0, 0, 0], classes=[[0, 1]])
    golden('balanced_multiclass', 2 / 3, 'balanced_accuracy', ['a', 'a', 'b', 'c'],
           ['a', 'a', 'a', 'c'], classes=[['a', 'b', 'c']])
    golden('balanced_missing_declared', .5, 'balanced_accuracy', [0, 0],
           [0, 1], classes=[[0, 1, 2]])
    golden('balanced_multioutput_weighted', .375, 'balanced_accuracy',
           [[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 1], [1, 1], [1, 1], [1, 0]],
           classes=[[0, 1], [0, 1]], target_weights=[1, 3])
    rejected('unknown_metric', 'unsupported_metric', 'made_up', [1], [1])
    rejected('shape_mismatch', 'invalid_metric_input', 'mape', [1, 2], [1])
    rejected('empty', 'invalid_metric_input', 'mape', [], [])
    rejected('negative_weight', 'invalid_metric_input', 'mape', [[1, 2]], [[1, 2]], target_weights=[-1, 2])
    rejected('unknown_class', 'invalid_metric_input', 'balanced_accuracy', [0, 1], [0, 7], classes=[[0, 1]])
    return {'passed': True, 'cases': cases, 'policies': POLICIES,
            'status': 'staged_pilot_not_enabled_in_frozen_A_or_B', 'organizer_requests': 0,
            'remaining': 'Integrate into contracts, semantic interpretation, scoring and exact-archive model routes after A/B comparison'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists()
    result = pilot()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'cases': len(result['cases'])}))
