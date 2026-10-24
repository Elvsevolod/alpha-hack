"""Feature-only EDA window policy and bounded paired temporal error comparison."""
import math

import numpy as np

from ..validation import choose_validation, ordered_periods

PROTOCOL_VERSION = 3


def choose_eda_validation(dataset, seed):
    # Production partition defaults preserved; adaptive evaluation is nested within fit.
    return choose_validation(dataset, seed)


def evaluation_contract(dataset, plan, design=None):
    result = {'version': PROTOCOL_VERSION, 'protocol_hash': plan.hash, 'strategy': plan.strategy,
        'temporal_window': None, 'design': design.model_dump(mode='json') if design else None,
        'method':design.method if design else 'descriptive',
        'claim_limits': ['Aggregate bias does not locate errors in the target tails.',
            'Duplicate features do not establish their cause or whether they are harmless.',
            'No universal effect threshold or uncertainty estimator is valid for every task.']}
    if plan.strategy not in {'temporal', 'temporal_group'}:
        if design and (design.temporal_window_periods or design.gap_periods or design.method=='moving_block'):
            raise ValueError('Temporal design requires an established temporal transfer strategy')
        return result
    column = dataset.task.time_column
    periods = int(ordered_periods(dataset.train.iloc[plan.fit][column]).nunique())
    external = int(ordered_periods(dataset.test[column]).nunique())
    declared = dataset.task.horizon if dataset.task.kind == 'forecasting' else None
    window = design.temporal_window_periods if design else declared
    if window is not None and window + (design.gap_periods if design else 0) >= periods:
        raise ValueError('Requested evaluation window/gap leaves no earlier fit history')
    if declared and window is not None and window < declared:
        raise ValueError('Evaluation cannot shorten the declared forecast horizon')
    result.update(temporal_window=window, available_fit_periods=periods, external_periods=external,
        horizon_covered=window >= external if window is not None else None,
        gap_periods=design.gap_periods if design else 0,
        design_status='agent_selected' if design else 'unselected',
        limitations=['Single historical origin; regime and seasonal changes may invalidate transfer.',
            'Observed periods are not elapsed duration; irregular calendars require separate interpretation.',
            'Window, effect criterion and uncertainty assumptions require task-specific justification.'])
    return result


def period_errors(time, actual, predicted):
    time = ordered_periods(time).reset_index(drop=True)
    error = np.asarray(predicted, dtype=float).reshape(-1) - np.asarray(actual, dtype=float).reshape(-1)
    result = []
    for period in sorted(time.unique()):
        selected = (time == period).to_numpy()
        e = error[selected]
        result.append({'period': str(period), 'rows': len(e), 'squared_error': float((e*e).sum()),
            'absolute_error': float(np.abs(e).sum()), 'error': float(e.sum())})
    return result


def paired_comparison(baseline, candidate):
    result = {'status': 'incompatible', 'verdict': 'inconclusive',
        'limitations': ['One historical origin; any interval is conditional on the selected dependence assumptions.',
            'Repeated adaptive selection and external regime shifts are not covered by this comparison.']}
    keys = ('metric', 'fit_keys_hash', 'evaluation_keys_hash', 'evaluation_protocol_hash')
    if not all(baseline.get(k) is not None and baseline[k] == candidate.get(k) for k in keys):
        return result
    metric = candidate['metric']
    a, b = baseline.get('period_errors', []), candidate.get('period_errors', [])
    if metric not in {'rmse','mse','mae'} or not a or [(r['period'],r['rows']) for r in a] != [(r['period'],r['rows']) for r in b]:
        return result
    contract=candidate.get('evaluation_contract',{})
    design=contract.get('design') or {}
    n = len(a)
    result.update(status='descriptive', periods=n, baseline_score=baseline['score'], candidate_score=candidate['score'],
        score_delta=candidate['score']-baseline['score'], method=contract.get('method','descriptive'))
    if contract.get('method') != 'moving_block':
        result['reason']='Descriptive comparison selected; no uncertainty interval or automatic acceptance.'
        return result
    block=design.get('block_periods')
    if not block or block >= n or n//block < 2:
        result.update(status='limited',reason='Selected block size leaves fewer than two blocks; interval unavailable.')
        return result
    resamples=design['resamples']
    if resamples*n > 2_000_000:
        result.update(status='budget_too_small',reason='Selected resampling exceeds bounded comparison memory; interval unknown.')
        return result
    rng = np.random.default_rng(42)
    starts = rng.integers(0, n-block+1, size=(resamples, math.ceil(n/block)))
    indices = (starts[:,:,None] + np.arange(block)).reshape(resamples, -1)[:,:n]
    counts = np.asarray([r['rows'] for r in a])
    field = 'absolute_error' if metric == 'mae' else 'squared_error'
    def scores(rows):
        losses = np.asarray([r[field] for r in rows])
        values = losses[indices].sum(axis=1) / counts[indices].sum(axis=1)
        return np.sqrt(values) if metric == 'rmse' else values
    alpha=(1-design['confidence'])/2
    low, high = map(float, np.quantile(scores(b)-scores(a), [alpha,1-alpha]))
    floors=[v for v in [design.get('minimum_effect_abs'),
        abs(baseline['score'])*design['minimum_effect_rel'] if design.get('minimum_effect_rel') is not None else None] if v is not None]
    threshold=max(floors) if floors else None
    result.update(status='available', method='moving_block_percentile', resamples=resamples, block_periods=block,
        delta_interval=[low,high], confidence=design['confidence'], minimum_absolute_gain=threshold,
        horizon_covered=contract.get('horizon_covered'),
        verdict='supported_on_this_origin' if threshold is not None and high < -threshold else
            'below_selected_effect_floor' if threshold is not None and high < 0 else
            'direction_only_on_this_origin' if high < 0 else 'worse_on_this_origin' if low > 0 else 'inconclusive')
    return result
