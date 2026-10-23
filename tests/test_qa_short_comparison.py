import json

import pytest

from tools import diagnostic_3h as d
from tools.qa_compare_short import compare


@pytest.fixture
def receipts(tmp_path, monkeypatch):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    protocol = {'status': 'authorized_preregistered_before_live_results',
        'archive_sha256': {'A': 'archive_A', 'B': 'archive_B'},
        'effective_time_limit_seconds_per_task': 1200, 'cpu_per_task': 5,
        'memory_gib_per_task': 5, 'api_budget_cap_per_series': 100,
        'basis': {'inputs': {'regression': {}, 'classification': {}}, 'training_seed': 42,
            'model': 'test_model', 'temperature': 1, 'reasoning_efforts': ['medium', 'xhigh'],
            'metrics': {'regression': {'name': 'rmse', 'maximize': False, 'min_delta_abs': .001, 'min_delta_rel': 0},
                        'classification': {'name': 'f1', 'average': 'macro', 'maximize': True, 'min_delta_abs': .001, 'min_delta_rel': 0}}}}
    path = tmp_path / 'protocol.json'
    write(path, protocol)
    for arm in ('A', 'B'):
        root = tmp_path / arm
        write(root / 'MANIFEST.json', {'archive_sha256': 'archive_' + arm, 'duration_seconds': 1200,
            'cpu_per_task': 5, 'memory_gib_per_task': 5, 'api_budget_cap': 100,
            'supervisor_sources': {'supervisor': 'same'},
            'jobs': {kind: {'input_hashes': {}} for kind in ('regression', 'classification')}})
        results = []
        for kind in ('regression', 'classification'):
            analysis = root / kind / 'analysis'
            write(analysis / 'metadata.json', {'validation': {'strategy': 'random', 'fit': [1, 2], 'validation': [3], 'control': [4], 'seed': 42},
                'task': {'targets': ['y'], 'id_columns': ['id'], 'output': {'prediction_type': 'values' if kind == 'regression' else 'labels'},
                         'metric': {'name': 'rmse' if kind == 'regression' else 'f1', 'average': 'macro'}},
                'qwen_configuration': {'model': 'test_model'}})
            write(analysis / 'experiments.json', [{'spec': {'pipeline': {'seed': 42}}}])
            score = (1.0 if arm == 'A' else .8) if kind == 'regression' else (.5 if arm == 'A' else .6)
            results.append({'kind': kind, 'report': {'control_winner': 'baseline', 'control_scores': {'baseline': score}}})
        write(root / 'REPORT.json', {'full_run_passed': True, 'results': results, 'reconciliation': {'interval_increase': 10}})
    monkeypatch.setattr(d, 'events', lambda _: [{'kind': 'request_queued', 'task_id': kind, 'temperature': 1, 'effort': 'medium'}
                                              for kind in ('regression', 'classification')])
    return tmp_path / 'A', tmp_path / 'B', path


def mutate(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_valid_comparison_preserves_unrounded_direction(receipts):
    result = compare(*receipts)
    assert result['comparison_valid'] and result['measured_api_spend_total'] == 20
    assert result['tasks']['regression']['observed_scores']['directed_gain_B_minus_A'] == 1.0 - .8
    assert result['tasks']['classification']['observed_scores']['directed_gain_B_minus_A'] == .6 - .5


@pytest.mark.parametrize('field,new_value', [('control', [5]), ('seed', 17)])
def test_equal_strategy_is_insufficient_when_rows_or_seed_differ(receipts, field, new_value):
    _, b, _ = receipts
    mutate(b / 'regression/analysis/metadata.json', lambda value: value['validation'].update({field: new_value}))
    result = compare(*receipts)
    assert not result['comparison_valid'] and result['tasks']['regression']['rollback_required'] is None


def test_macro_micro_mismatch_is_not_a_quality_gain(receipts):
    _, b, _ = receipts
    mutate(b / 'classification/analysis/metadata.json', lambda value: value['task']['metric'].update(average='micro'))
    assert not compare(*receipts)['tasks']['classification']['comparable']


def test_missing_contract_is_reported_as_incomparable(receipts):
    _, b, _ = receipts
    mutate(b / 'regression/analysis/metadata.json', lambda value: value.pop('task'))
    assert not compare(*receipts)['tasks']['regression']['comparable']


def test_technical_failure_keeps_scores_without_accepting_quality(receipts):
    _, b, _ = receipts
    mutate(b / 'REPORT.json', lambda value: value.update(full_run_passed=False))
    result = compare(*receipts)
    assert not result['comparison_valid']
    assert result['tasks']['regression']['observed_scores'] is not None
    assert result['tasks']['regression']['rollback_required'] is None


def test_missing_control_scores_does_not_abort_failure_report(receipts):
    _, b, _ = receipts
    mutate(b / 'REPORT.json', lambda value: value['results'][0]['report'].update(control_scores=None,control_winner=None))
    result=compare(*receipts)
    assert not result['comparison_valid'] and result['tasks']['regression']['observed_scores'] is None
    assert result['tasks']['classification']['observed_scores'] is not None
