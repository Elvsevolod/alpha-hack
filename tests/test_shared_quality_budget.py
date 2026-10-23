import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools.shared_quality_budget import SharedQualityBudget
from tools.test_diagnostic_3h import gateway, post, snap


BODY = {'max_tokens': 32768}


def test_exclusive_process_lock_and_restart_preserve_all_prior_costs(tmp_path):
    path = tmp_path / 'period.json'
    first = SharedQualityBudget(path, snap())
    try:
        with pytest.raises(BlockingIOError): SharedQualityBudget(path, snap())
        assert first.begin('one', BODY, b'{}')
        assert first.complete('one', 200, {'cost': 9.})
    finally: first.release()
    restarted = SharedQualityBudget(path, snap())
    try:
        assert restarted.spent == 9 and restarted.state['requests']['one']['cost'] == 9
        assert restarted.begin('two', BODY, b'{}')
        assert restarted.complete('two', 200, {'cost': 8.})
        assert restarted.spent == 17
    finally: restarted.release()


def test_crash_pending_cost_cannot_be_erased_by_restart(tmp_path):
    path = tmp_path / 'period.json'
    first = SharedQualityBudget(path, snap())
    assert first.begin('pending', BODY, b'{}')
    assert json.loads(path.read_text())['inflight'] == 'pending'
    first.release()
    second = SharedQualityBudget(path, snap())
    try:
        assert second.state['closed'] == 'request_still_inflight_at_release'
        assert not second.begin('new', BODY, b'{}')
        assert second.state['inflight'] == 'pending'
    finally: second.release()


def test_late_response_cannot_overwrite_a_restarted_process_ledger(tmp_path):
    path = tmp_path / 'period.json'
    old = SharedQualityBudget(path, snap())
    assert old.begin('pending', BODY, b'{}')
    old.release()
    restarted = SharedQualityBudget(path, snap())
    try:
        saved = path.read_bytes()
        assert not old.complete('pending', 200, {'cost': 9.})
        assert path.read_bytes() == saved
        assert restarted.state['closed'] and not restarted.begin('next', BODY, b'{}')
    finally: restarted.release()


def test_abrupt_process_exit_leaves_pending_billing_latched(tmp_path):
    path = tmp_path / 'period.json'
    code = """import os,sys,json
from tools.shared_quality_budget import SharedQualityBudget
v=SharedQualityBudget(sys.argv[1],json.loads(sys.argv[2]))
assert v.begin('interrupted',{'max_tokens':32768},b'{}')
os._exit(23)
"""
    result = subprocess.run([sys.executable, '-c', code, str(path), json.dumps(snap())],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=10)
    assert result.returncode == 23, result.stderr
    restarted = SharedQualityBudget(path, snap())
    try:
        assert restarted.state['closed'] == 'previous_request_billing_unknown'
        assert restarted.state['inflight'] == 'interrupted'
        assert not restarted.begin('next', BODY, b'{}')
    finally: restarted.release()


def test_counter_and_usage_conservatively_sum_without_releasing_any_spend(tmp_path):
    value = SharedQualityBudget(tmp_path / 'period.json', snap(spend=20, remaining=580))
    try:
        assert value.begin('one', BODY, b'{}') and value.complete('one', 200, {'cost': 9.})
        assert value.spent == 29
        assert value.observe(snap(spend=29, remaining=571)) and value.spent == 38
        assert value.observe(snap(spend=39, remaining=561)) and value.spent == 48
        assert value.observe(snap(spend=30, remaining=570)) and value.spent == 48
        assert value.observe(snap(spend=290, remaining=310)) and value.spent == 299
        assert not value.observe(snap(spend=291, remaining=309))
        assert not value.observe(snap(spend=20, remaining=580))
    finally: value.release()


def test_disjoint_counter_spend_cannot_hide_already_reported_own_charges(tmp_path):
    value = SharedQualityBudget(tmp_path / 'period.json', snap())
    try:
        assert value.observe(snap(spend=150, remaining=450))
        for index in range(17):
            assert value.begin(str(index), BODY, b'{}')
            admitted = value.complete(str(index), 200, {'cost': 9.})
            assert admitted is (index < 16)
        assert value.spent == 303 and not value.begin('eighteen', BODY, b'{}')
    finally: value.release()


@pytest.mark.parametrize('usage', [None, {}, {'cost': None}, {'cost': True}, {'cost': float('nan')}, {'cost': -1}, {'cost': float('inf')}])
def test_unknown_or_invalid_billing_stops_all_future_admission(tmp_path, usage):
    value = SharedQualityBudget(tmp_path / 'period.json', snap())
    try:
        assert value.begin('one', BODY, b'{}')
        assert not value.complete('one', 502, usage)
        assert value.state['closed'] == 'response_billing_unknown'
        assert value.state['inflight'] == 'one'
        assert not value.begin('two', BODY, b'{}')
    finally: value.release()


@pytest.mark.parametrize('changed', [snap(key='other'), snap(period='other'), snap(spend=-1), {**snap(), 'budget_exhausted': True}])
def test_account_identity_period_or_availability_change_closes(tmp_path, changed):
    value = SharedQualityBudget(tmp_path / 'period.json', snap())
    try: assert not value.observe(changed) and value.state['closed']
    finally: value.release()


def test_replayed_completion_does_not_double_charge(tmp_path):
    value = SharedQualityBudget(tmp_path / 'period.json', snap())
    try:
        assert value.begin('one', BODY, b'{}') and value.complete('one', 200, {'cost': 9.})
        assert not value.complete('one', 200, {'cost': 9.})
        assert value.spent == 9 and value.state['closed'] == 'billing_request_identity_mismatch'
    finally: value.release()


@pytest.mark.parametrize('body,raw', [({}, b'{}'), ({'max_tokens': True}, b'{}'), ({'max_tokens': 65537}, b'{}'), (BODY, b' ' * 262145)])
def test_external_request_bounds_reject_before_paid_send(tmp_path, body, raw):
    value = SharedQualityBudget(tmp_path / 'period.json', snap())
    try:
        assert not value.begin('one', body, raw)
        assert value.state['inflight'] is None and value.spent == 0
    finally: value.release()


def test_both_gateway_tasks_share_cost_even_while_account_counter_lags(tmp_path):
    entered, release = threading.Event(), threading.Event()
    ledger = SharedQualityBudget(tmp_path / 'period.json', snap(spend=285, remaining=315))
    def upstream(handler, body, count):
        if count == 1:
            entered.set(); assert release.wait(5)
        return 200, {'choices': [], 'usage': {'cost': 9 if count == 1 else 8}}
    try:
        with gateway(tmp_path / 'gateway', upstream) as (server, guard, calls):
            server.accounting = ledger
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(post, server, task='regression', **BODY)
                assert entered.wait(5)
                second = pool.submit(post, server, task='classification', **BODY)
                release.set()
                assert first.result(timeout=10).status_code == second.result(timeout=10).status_code == 200
            assert ledger.spent == 302 and len(calls) == 2
            assert guard.reason == 'shared_300_admission_threshold'
            assert post(server, **BODY).status_code == 402 and len(calls) == 2
            assert len(ledger.state['requests']) == 2 and ledger.state['inflight'] is None
    finally:
        release.set(); ledger.release()


def test_unpriced_http_failure_is_forwarded_once_then_both_tasks_stop(tmp_path):
    ledger = SharedQualityBudget(tmp_path / 'period.json', snap())
    try:
        with gateway(tmp_path / 'gateway', lambda *args: (429, {'error': 'offline rate limit'})) as (server, guard, calls):
            server.accounting = ledger
            assert post(server, **BODY).status_code == 429
            assert guard.reason == 'response_billing_unknown'
            assert post(server, task='classification', **BODY).status_code == 402
            assert len(calls) == 1 and ledger.state['inflight'] is not None
    finally: ledger.release()


def test_final_priced_sse_is_preserved_before_the_next_request_is_blocked(tmp_path):
    from tools.goal_quality_run import decode_response
    raw = ('data: ' + json.dumps({'choices': [{'index': 0, 'delta': {'content': '{}'}, 'finish_reason': 'stop'}]})
        + '\n\ndata: ' + json.dumps({'choices': [], 'usage': {'cost': 9., 'prompt_tokens': 4, 'completion_tokens': 3}})
        + '\n\ndata: [DONE]\n\n').encode()
    def upstream(handler, *_):
        handler.send_response(200)
        handler.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        handler.send_header('Content-Length', str(len(raw)))
        handler.end_headers(); handler.wfile.write(raw)
    ledger = SharedQualityBudget(tmp_path / 'period.json', snap(spend=293, remaining=307))
    try:
        with gateway(tmp_path / 'gateway', upstream) as (server, guard, calls):
            server.accounting, server.response_decoder = ledger, decode_response
            response = post(server, stream=True, **BODY)
            assert response.status_code == 200 and response.content == raw
            assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'
            assert ledger.spent == 302 and guard.closed
            assert post(server, task='classification', **BODY).status_code == 402
            assert len(calls) == 1 and not server.unknown
    finally: ledger.release()


def test_durable_begin_crossing_deadline_never_sends_upstream(tmp_path):
    ledger = SharedQualityBudget(tmp_path / 'period.json', snap())
    began, cancelled = threading.Event(), threading.Event()
    try:
        with gateway(tmp_path / 'gateway', lambda *args: (200, {'usage': {'cost': 9}})) as (server, guard, calls):
            server.accounting = ledger
            deadline = [None]
            def read_deadline(*_):
                deadline[0] = time.monotonic() + 2
                return deadline[0], 0, 'offline_test'
            server.deadline_reader = read_deadline
            original_begin, original_cancel = ledger.begin, ledger.cancel_before_send
            def slow_begin(*args):
                result = original_begin(*args)
                began.set()
                time.sleep(max(0, deadline[0] - time.monotonic()) + .02)
                return result
            def cancel(*args):
                result = original_cancel(*args); cancelled.set(); return result
            ledger.begin, ledger.cancel_before_send = slow_begin, cancel
            assert post(server, **BODY).status_code == 402
            assert began.is_set() and cancelled.wait(3)
            assert not calls and ledger.state['inflight'] is None and ledger.spent == 0
            assert next(iter(ledger.state['requests'].values()))['not_sent']
    finally: ledger.release()
