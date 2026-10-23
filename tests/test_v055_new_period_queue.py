from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / 'outputs/quality_goal_20261023/v055_new_period_full_launcher.py'
spec = importlib.util.spec_from_file_location('v055_new_period_queue', PATH)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def snapshot():
    return {'key_fingerprint': 'fake-test-key', 'remaining_budget': 600.,
            'budget_exhausted': False, 'limits': {'budget_reset_at': queue.NEXT_RESET.isoformat(), 'blocked': False, 'spend': 0.}}


@pytest.mark.parametrize('value', [None, True, -1., 0., 5., float('nan'), float('inf')])
def test_unavailable_budget_does_not_start(value):
    state = snapshot()
    state['remaining_budget'] = value
    assert queue.readiness(queue.START, state, state) == 'account_unavailable'


@pytest.mark.parametrize('value', [None, True, -1., 300., 593., float('nan'), float('inf')])
def test_counter_must_allow_shared_ledger_admission_before_creation(value):
    state = snapshot()
    state['limits']['spend'] = value
    assert queue.readiness(queue.START, state, state) == 'account_unavailable'


def test_exact_authorized_period_and_window():
    state = snapshot()
    assert queue.readiness(queue.START - timedelta(seconds=1), state, state) == 'wait_reset'
    assert queue.readiness(queue.START, state, state) == 'ready'
    assert queue.readiness(queue.END, state, state) == 'start_window_closed'
    for reset in ('2026-10-23T00:00:00+00:00', '2026-10-25T00:00:00+00:00'):
        old = snapshot()
        old['limits']['budget_reset_at'] = reset
        assert queue.readiness(queue.START, old, state) == 'wait_new_period_confirmation'


@pytest.mark.parametrize('seconds', [1, 30, 600, 1799])
def test_delayed_background_reset_stays_within_authorized_period(seconds):
    state = snapshot()
    state['limits']['budget_reset_at'] = (queue.NEXT_RESET + timedelta(seconds=seconds)).isoformat()
    assert queue.readiness(queue.START, state, state) == 'ready'


@pytest.mark.parametrize('seconds', [-1, 1800, 86400])
def test_reset_outside_bounded_next_period_is_not_authorized(seconds):
    state = snapshot()
    state['limits']['budget_reset_at'] = (queue.NEXT_RESET + timedelta(seconds=seconds)).isoformat()
    assert queue.readiness(queue.START, state, state) == 'wait_new_period_confirmation'


def test_changed_key_and_naive_reset_are_rejected():
    state = snapshot()
    wrong = snapshot()
    wrong['key_fingerprint'] = 'another-fake-key'
    with pytest.raises(ValueError, match='key changed'):
        queue.readiness(queue.START, wrong, state)
    wrong = snapshot()
    wrong['limits']['budget_reset_at'] = '2026-10-24T00:00:00'
    with pytest.raises(ValueError, match='timezone'):
        queue.readiness(queue.START, wrong, state)


@pytest.mark.parametrize('field', ['blocked', 'budget_exhausted'])
def test_blocked_account_does_not_start(field):
    state = snapshot()
    (state['limits'] if field == 'blocked' else state)[field] = True
    assert queue.readiness(queue.START, state, state) == 'account_unavailable'


@pytest.fixture
def mocked_main(tmp_path, monkeypatch):
    clock = [queue.START - timedelta(seconds=1)]
    class Clock(datetime):
        @staticmethod
        def now(_):
            return clock[0]
    monkeypatch.setattr(queue, 'datetime', Clock)
    monkeypatch.setattr(queue.time, 'sleep', lambda _: clock.__setitem__(0, clock[0] + timedelta(seconds=15)))
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    monkeypatch.setattr(queue, 'RESULTS', tmp_path)
    monkeypatch.setattr(queue, 'OUTPUT', tmp_path / 'future_full_pair')
    archive = tmp_path / 'fake.zip'
    archive.write_bytes(b'test-only-archive')
    monkeypatch.setattr(queue, 'ARCHIVE', archive)
    monkeypatch.setattr(queue, 'SHA', hashlib.sha256(archive.read_bytes()).hexdigest())
    (tmp_path / 'api_limits_initial.json').write_text(json.dumps(snapshot()))
    monkeypatch.setattr(queue, 'verify_previous', lambda: None)
    monkeypatch.setattr(queue.d.limits, 'load_key', lambda _: 'fake-test-key')
    observed = []
    def fetch(_):
        observed.append(clock[0])
        return snapshot()
    monkeypatch.setattr(queue.d.limits, 'fetch_info', fetch)
    monkeypatch.setattr(queue.d.limits, 'snapshot', lambda value, *args: value)
    monkeypatch.setattr(queue.d, 'docker', lambda *a: SimpleNamespace(stdout=''))
    monkeypatch.setattr(queue.shutil, 'disk_usage', lambda _: SimpleNamespace(free=8 * 1024**3))
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(queue.subprocess, 'run', run)
    return clock, observed, commands


def test_main_waits_then_runs_one_stock_shared_budget_pair(mocked_main):
    _, observed, commands = mocked_main
    assert queue.main() == 0
    assert observed and min(observed) >= queue.START
    assert len(commands) == 1
    command = commands[0]
    for flag, value in (('--seconds', '10800'), ('--api-cap', '400'), ('--cpu', '5'), ('--memory-gib', '4.5')):
        assert command[command.index(flag) + 1] == value
    assert command[command.index('--period-anchor') + 1].endswith('api_limits_initial.json')
    with pytest.raises(FileExistsError):
        queue.main()  # Even if a failed driver created no output, its x-mode log prevents a repeat.
    assert len(commands) == 1


def test_old_period_observation_waits_before_single_launch(mocked_main, monkeypatch):
    _, observed, commands = mocked_main
    sequence = [snapshot(), snapshot()]
    sequence[0]['limits']['budget_reset_at'] = '2026-10-23T00:00:00+00:00'
    monkeypatch.setattr(queue.d.limits, 'fetch_info', lambda _: sequence.pop(0))
    assert queue.main() == 0
    assert len(commands) == 1 and not sequence


@pytest.mark.parametrize('failure', ['previous', 'running', 'disk', 'archive', 'already_started'])
def test_failed_precondition_never_invokes_driver(mocked_main, monkeypatch, failure):
    _, _, commands = mocked_main
    if failure == 'previous':
        monkeypatch.setattr(queue, 'verify_previous', lambda: (_ for _ in ()).throw(AssertionError('previous')))
    elif failure == 'running':
        monkeypatch.setattr(queue.d, 'docker', lambda *a: SimpleNamespace(stdout='other-container'))
    elif failure == 'disk':
        monkeypatch.setattr(queue.shutil, 'disk_usage', lambda _: SimpleNamespace(free=6 * 1024**3))
    elif failure == 'archive':
        queue.ARCHIVE.write_bytes(b'changed')
    else:
        queue.OUTPUT.mkdir()
    with pytest.raises(AssertionError):
        queue.main()
    assert not commands
