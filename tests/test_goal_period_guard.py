import pytest

from tools import diagnostic_3h as d
from tools.goal_quality_run import PeriodGuard, SharedPeriodGuard
from tools.test_diagnostic_3h import snap


def guard(tmp_path, reset='2026-10-23T00:00:00+00:00'):
    from datetime import datetime
    before=snap(period=reset)
    wall=datetime.fromisoformat('2026-10-22T23:45:00+00:00').timestamp()
    now=[100.]
    value=PeriodGuard(before,lambda:before,tmp_path,clock=lambda:now[0],wall_time=lambda:wall,
        disk=lambda:10*d.GIB)
    d.write(tmp_path/'observer_heartbeat.json',{'mono':now[0]})
    return value,now


def test_period_close_does_not_wait_for_cached_api_refresh(tmp_path):
    value,now=guard(tmp_path)
    assert value.period_end==400 and value.allowed()
    now[0]=399.99
    d.write(tmp_path/'observer_heartbeat.json',{'mono':now[0]})
    assert value.allowed()
    now[0]=400
    assert not value.allowed() and value.reason=='api_period_admission_closed'
    now[0]=200
    assert not value.allowed()


def test_response_deadline_cannot_cross_period_even_if_agent_has_hours(tmp_path,monkeypatch):
    value,_=guard(tmp_path)
    monkeypatch.setattr(d,'response_deadline',lambda body,job:(10000.,300.,'stock'))
    assert value.response_deadline({}, {})==(400.,300.,'stock+api_period')
    monkeypatch.setattr(d,'response_deadline',lambda body,job:(200.,300.,'stock'))
    assert value.response_deadline({}, {})==(200.,300.,'stock+api_period')


def test_slow_limit_refresh_cannot_admit_after_boundary(tmp_path):
    value,now=guard(tmp_path)
    now[0]=399
    d.write(tmp_path/'observer_heartbeat.json',{'mono':now[0]})
    def delayed_fetch():
        now[0]=401
        return value.before
    value.fetch=delayed_fetch
    assert not value.allowed() and value.reason=='api_period_admission_closed'


def test_slow_shared_ledger_fsync_cannot_admit_after_boundary(tmp_path):
    from datetime import datetime
    before=snap(period='2026-10-23T00:00:00+00:00')
    wall=datetime.fromisoformat('2026-10-22T23:45:00+00:00').timestamp()
    now=[100.]
    class Accounting:
        state={'closed':None}
        def observe(self, _):
            now[0]=401.
            return True
    value=SharedPeriodGuard(before,lambda:before,tmp_path,clock=lambda:now[0],wall_time=lambda:wall,
        disk=lambda:10*d.GIB,accounting=Accounting())
    now[0]=399.
    d.write(tmp_path/'observer_heartbeat.json',{'mono':now[0]})
    assert not value.allowed() and value.reason=='api_period_admission_closed'


@pytest.mark.parametrize('reset',['2026-10-22T23:54:00+00:00','2026-10-22T23:50:00+00:00',
    '2026-10-22T23:00:00+00:00','2026-10-23T00:00:00','bad'])
def test_unsafe_or_unknown_period_refuses_start(tmp_path,reset):
    with pytest.raises(ValueError):guard(tmp_path,reset)
