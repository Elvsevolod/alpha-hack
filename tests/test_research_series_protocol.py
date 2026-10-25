import json
from tools.research_series_preflight import inspect_metadata, proof_issues, registration
from tools.research_memory_series import TASKS, FILES


def test_price_metadata_does_not_export_credentials_or_assume_usd_is_gateway_units():
    raw = {'data': [{'id': 'qwen/qwen3.8-27b', 'api_key': 'PRIVATE_SENTINEL',
        'pricing': {'currency': 'USD', 'prompt': '0.000001', 'completion': '0.000002'}}]}
    rows = inspect_metadata(raw)
    assert 'PRIVATE_SENTINEL' not in json.dumps(rows)
    assert 'gateway_unit_mapping_unconfirmed' in proof_issues(rows[0])
    assert proof_issues({'billing_unit': 'gateway_unit', 'input_cost_per_token': 1e-6,
        'output_cost_per_token': 2e-6, 'fixed_request_cost': 0,
        'completion_includes_reasoning': True}) == []


def test_registration_is_balanced_and_excludes_sealed_labels_from_inputs():
    # The earlier capped preflight remains a historical record; the actual live
    # runner records the later human override with monetary_series_cap=None.
    rows = registration()['runs']
    assert [(r['task'], r['arm']) for r in rows] == [
        ('adult', 'A'), ('adult', 'B'), ('bank', 'B'), ('bank', 'A'),
        ('bike', 'A'), ('bike', 'B'), ('har', 'B'), ('har', 'A')]
    assert tuple(dict.fromkeys(r['task'] for r in rows)) == TASKS
    assert set(FILES) == {'task.md', 'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv'}
    assert not any(row['outer_labels_opened'] for row in rows)
