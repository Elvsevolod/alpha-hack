import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from tools.diagnostic_store import Store
from tools import diagnostic_3h as d
from automl_agent import diagnostics


def test_durable_idempotent_event_and_backup_intent(tmp_path):
    store=Store(tmp_path);body={'event_id':'one','kind':'experiment_record','payload':{'result':{'status':'completed'}}}
    a=store.receive('regression',body);b=Store(tmp_path).receive('regression',body)
    assert a['event_status']=='committed' and a['backup_status']=='queued' and b['duplicate']
    assert a['backup_job_id']==b['backup_job_id'] and len(store.jobs())==1
    assert len(list(store.records()))==1
    with pytest.raises(ValueError,match='identity_conflict'):store.receive('regression',{**body,'payload':{'result':{'status':'error'}}})
    assert len(list(store.records()))==1


def test_event_and_job_rollback_then_replay(tmp_path,monkeypatch):
    store=Store(tmp_path);original=store._enqueue
    def fail(*a,**k):raise OSError('crash after append')
    monkeypatch.setattr(store,'_enqueue',fail)
    body={'event_id':'one','kind':'experiment_record','payload':{'result':{'status':'completed'}}}
    with pytest.raises(OSError):store.receive('regression',body)
    assert not store.jobs() and not list(store.records())
    monkeypatch.setattr(store,'_enqueue',original)
    assert store.receive('regression',body)['event_status']=='committed'
    assert store.verify()['records']==1


def test_read_committed_prefix_while_writer_has_uncommitted_tail(tmp_path):
    store=Store(tmp_path);store.append('ok');segment=store.index()['segments'][0]
    with (store.root/segment['path']).open('ab') as out:out.write(b'partial')
    assert [r['kind'] for r in store.records()]==['ok']
    with pytest.raises(ValueError,match='tail'):store.verify()
    store.append('next')
    assert store.verify()['records']==2


def test_queue_bound_and_overdue_close_gate(tmp_path):
    store=Store(tmp_path);store.enqueue('regression','test',maximum=1,deadline_seconds=-1)
    with pytest.raises(ValueError,match='queue_full'):store.enqueue('regression','more',maximum=1)
    from tools.test_diagnostic_3h import snap,beat
    beat(tmp_path);guard=d.Guard(snap(),lambda:snap(),tmp_path)
    assert not guard.allowed() and guard.reason=='required_backup_failed_or_overdue'
    assert guard.closed


def test_independent_latch_survives_audit_failure(tmp_path,monkeypatch):
    root=tmp_path/'signals'/'regression';root.mkdir(parents=True)
    (root/'HEALTHY').touch();(root/'ARMED').touch()
    monkeypatch.setenv('AGENT_DIAGNOSTICS_FAILURE_DIR',str(root))
    import automl_agent.util as util
    monkeypatch.setattr(util,'atomic_json',lambda *a:(_ for _ in ()).throw(OSError('audit disk error')))
    diagnostics.fail('receipt_failed')
    assert not (root/'HEALTHY').exists() and d.emergency_reason(tmp_path)=='agent_diagnostics_failed:regression'
    from tools.test_diagnostic_3h import snap,beat
    beat(tmp_path);guard=d.Guard(snap(),lambda:snap(),tmp_path)
    monkeypatch.setattr(d,'event',lambda *a,**k:(_ for _ in ()).throw(OSError('host audit disk error')))
    assert not guard.allowed() and guard.closed and (tmp_path/'CLOSE').exists()


def test_lost_receipt_retries_exact_id(tmp_path,monkeypatch):
    store=Store(tmp_path);calls=[]
    class Response:
        def __init__(self,receipt):self.receipt=receipt
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self):return json.dumps(self.receipt).encode()
    def urlopen(request,**kwargs):
        body=json.loads(request.data);calls.append(body);receipt=store.receive('regression',body)
        if len(calls)==1:raise TimeoutError('receipt lost after durable write')
        return Response(receipt)
    monkeypatch.setenv('AGENT_DIAGNOSTICS_URL','http://offline/diagnostics')
    monkeypatch.setattr(diagnostics.urllib.request,'urlopen',urlopen)
    receipt=diagnostics.emit('test',value=1)
    assert receipt['duplicate'] and calls[0]==calls[1] and len(list(store.records()))==1


def test_backup_15_seconds_does_not_delay_event_receipt(tmp_path,monkeypatch):
    # Real HTTP and actual 15s copy delay; no accelerated clock.
    from tools.test_diagnostic_3h import gateway
    started=threading.Event()
    def slow(*a,**k):started.set();time.sleep(15);return [{'sha256':'verified_fixture'}]
    monkeypatch.setattr(d,'backup_sqlite',slow)
    with gateway(tmp_path/'gateway',lambda *a:(200,{})) as (server,guard,calls):
        body={'event_id':'slow','kind':'experiment_record','payload':{'result':{'status':'completed'}}}
        begin=time.monotonic()
        response=d.httpx.post(f'http://127.0.0.1:{server.server_port}/regression/diagnostics',json=body,timeout=10)
        elapsed=time.monotonic()-begin
        receipt=response.json()
        assert response.status_code==200 and elapsed<3 and receipt['backup_status']=='queued'
        assert started.wait(2)
        assert server.store.jobs()[0]['status']=='running'
        assert not calls and not guard.closed
        server.backups['regression'].thread.join(timeout=.01)
        until=time.monotonic()+18
        while server.store.jobs()[0]['status']!='completed':
            assert time.monotonic()<until;time.sleep(.1)
        assert server.store.jobs()[0]['ended']-server.store.jobs()[0]['started']>=15


def test_missing_required_snapshot_latches_before_cleanup(tmp_path,monkeypatch):
    signal=tmp_path/'signals'/'regression';signal.mkdir(parents=True)
    (signal/'ARMED').touch();(signal/'HEALTHY').touch()
    monkeypatch.setenv('AGENT_DIAGNOSTICS_URL','http://unused')
    monkeypatch.setenv('AGENT_DIAGNOSTICS_FAILURE_DIR',str(signal))
    with pytest.raises(diagnostics.DiagnosticUnavailable):diagnostics.artifact(tmp_path/'missing.json')
    assert not (signal/'HEALTHY').exists()
    first=json.loads((signal/'FAILURE.json').read_text())
    diagnostics.fail('later_failure')
    assert json.loads((signal/'FAILURE.json').read_text())==first


def test_diagnostic_missing_request_link_preserves_explicit_reason(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tools import diagnostic_3h
    records=[{'source':'gateway','kind':'request_link','task_id':'regression',
        'payload':{'request':'gateway1','call_id':'call1','request_id':None,
        'request_id_missing_reason':'no_accepted_command'}}]
    monkeypatch.setattr(diagnostic_3h,'Store',lambda directory:SimpleNamespace(root=directory,
        records=lambda:iter(records),verify=lambda:{'passed':True}))
    result=diagnostic_3h.diagnostic_report(tmp_path)
    assert result['links_complete']
    assert result['missing_links']==[{'gateway_id':'gateway1','reason':'no_accepted_command'}]
