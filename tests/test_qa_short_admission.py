import time

import pytest

from tools import diagnostic_3h as d
from tools.test_diagnostic_3h import gateway, post


@pytest.mark.parametrize('reason', ['budget_threshold_or_blocked', 'upstream_outcome_unknown',
    'upstream_outcome_unknown_at_deadline', 'api_period_admission_closed',
    'previous_request_billing_unknown', 'period_changed', 'incomparable_budget_counters',
    'account_unavailable', 'shared_300_admission_threshold', 'development_request_size_limit',
    'concurrent_or_duplicate_paid_admission', 'billing_request_identity_mismatch',
    'response_billing_unknown', 'observed_charge_exceeds_safety_margin',
    'request_still_inflight_at_release'])
def test_model_admission_stop_preserves_local_finalization(reason):
    assert not d.infrastructure_gate_closure(reason)


@pytest.mark.parametrize('reason', ['required_diagnostics_failed', 'required_backup_failed',
    'accounting_unavailable_60s', 'observer_heartbeat_expired', 'deadline_unavailable',
    'watchdog_or_supervisor_closed', 'unrecognized_reason', None])
def test_diagnostic_or_unrecognized_failure_still_stops_workers(reason):
    assert d.infrastructure_gate_closure(reason)


@pytest.mark.parametrize('reason,should_kill', [
    ('api_period_admission_closed', False), ('shared_300_admission_threshold', False),
    ('response_billing_unknown', False), ('required_diagnostics_failed', True),
    ('unrecognized_reason', True)])
def test_watchdog_distinguishes_api_stop_from_diagnostic_loss(tmp_path, monkeypatch, reason, should_kill):
    from types import SimpleNamespace

    clock = [0.]
    calls = []
    d.write(tmp_path / 'CONFIG.json', {'jobs': {'regression': {'kind': 'regression',
        'started': True, 'deadline_mono': 1000., 'container': 'not-a-real-container'}}})
    for owner in ('observer', 'gateway'):
        d.write(tmp_path / f'{owner}_heartbeat.json', {'mono': 0.})
    d.write(tmp_path / 'gateway_ready.json', {'port': 1})
    d.write(tmp_path / 'CLOSED.json', {'reason': reason})
    (tmp_path / 'CLOSE').touch()
    monkeypatch.setattr(d.time, 'monotonic', lambda: clock[0])
    def advance(_):
        clock[0] += 1.
        if clock[0] >= 4.:
            (tmp_path / 'WATCHDOG_DONE').touch()
    monkeypatch.setattr(d.time, 'sleep', advance)
    monkeypatch.setattr(d.httpx, 'get', lambda *a, **kw: SimpleNamespace(
        status_code=200, json=lambda: {'closed': True, 'reason': reason}))
    monkeypatch.setattr(d, 'emergency_reason', lambda _: None)
    def docker(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(d, 'docker', docker)
    d.watchdog_main(tmp_path, grace=2., free_disk=lambda: 10 * d.GIB)
    assert bool(calls) is should_kill
    if should_kill:
        assert calls == [('kill', 'not-a-real-container')]
    assert (tmp_path / 'regression_watchdog_killed.json').exists() is should_kill


def test_observed_response_latency_rejects_unsent_request_without_closing_gate(tmp_path):
    def provider(*args):
        time.sleep(.2)
        return 200, {'usage': {'total_tokens': 1}}

    with gateway(tmp_path / 'admission', provider) as (server, guard, calls):
        assert post(server).status_code == 200
        assert server.upstream_response_reserve >= .25
        response = post(server, deadline=time.monotonic() + .08)
        assert response.status_code == 402
        assert response.json()['error']['code'] == 'insufficient_response_budget'
        assert len(calls) == 1 and not guard.closed and not server.unknown
        assert post(server, deadline=time.monotonic() + 2).status_code == 200
        assert len(calls) == 2  # Other tasks/requests with enough time remain admissible.
