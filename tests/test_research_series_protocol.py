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


def test_exhausted_account_snapshot_is_known_not_a_network_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools import research_memory_series
    path = tmp_path / 'after.json'
    value = {'remaining_budget': 0, 'budget_exhausted': True, 'limits': {'blocked': True},
        'comparison': {'increase': 2.5}}
    path.write_text(json.dumps(value))
    monkeypatch.setattr(research_memory_series.subprocess, 'run',
        lambda *a, **kw: SimpleNamespace(returncode=3))
    assert research_memory_series.snapshot(path, 'after') == value
    monkeypatch.setattr(research_memory_series.subprocess, 'run',
        lambda *a, **kw: SimpleNamespace(returncode=2))
    assert research_memory_series.snapshot(path, 'after') is None


def test_unavailable_automl_package_is_rejected_before_execution(tmp_path, monkeypatch):
    import pytest
    from automl_agent.capabilities import CapabilityRegistry
    from automl_agent.contracts import PipelineSpec
    registry = CapabilityRegistry(tmp_path / 'capabilities.json')
    monkeypatch.setitem(registry.versions, 'autogluon.tabular', 'absent')
    with pytest.raises(ImportError, match='autogluon.tabular'):
        registry.require_available(PipelineSpec(engine='autogluon', model='automl'))


def test_changed_series_harness_is_rejected_without_an_api_request(tmp_path, monkeypatch):
    import pytest
    from tools import research_memory_series
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture-no-paid-requests')
    (tmp_path / 'protocol.json').write_text(json.dumps({'runs': [], 'harness_sha256': 'changed'}))
    monkeypatch.setattr(research_memory_series, 'snapshot', lambda *_a, **_kw: pytest.fail('API called'))
    with pytest.raises(ValueError, match='harness changed'):
        research_memory_series.live(tmp_path)
