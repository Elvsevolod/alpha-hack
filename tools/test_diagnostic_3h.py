"""Failure-injection acceptance of the external supervisor; no organizer requests."""
import argparse
from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnostic_3h as d
from tools.verify_api_v2_flat import fixture


def snap(spend=0, remaining=600, key='test-key-fingerprint', period='period-1'):
    return {'phase':'before', 'key_fingerprint':key, 'remaining_budget':remaining, 'budget_exhausted':False,
        'limits':{'spend':spend,'total_spend':spend,'max_budget':600,'blocked':False,'budget_reset_at':period}}


def beat(directory):
    d.write(directory/'observer_heartbeat.json', {'mono':time.monotonic()})


@contextmanager
def gateway(directory, upstream_fn, *, exact=False, jobs=None):
    directory.mkdir(parents=True)
    beat(directory)
    calls = []
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            raw=self.rfile.read(int(self.headers['Content-Length']))
            body=json.loads(raw)
            calls.append({'body':body,'mono':time.monotonic()})
            response=upstream_fn(self,body,len(calls))
            if response is None: return
            status, value=response
            content=value if isinstance(value,bytes) else json.dumps(value).encode()
            try:
                self.send_response(status);self.send_header('x-request-id','offline-provider-request');self.send_header('Retry-After','0');self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
            except OSError: pass
    upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
    threading.Thread(target=upstream.serve_forever,daemon=True).start()
    guard=d.Guard(snap(),lambda:snap(),directory)
    jobs=jobs or {kind:{'kind':kind,'deadline_mono':time.monotonic()+10800,'directory':str(directory)} for kind in ('regression','classification')}
    reader=d.response_deadline if exact else lambda b,j:(b.get('deadline',time.monotonic()+10),0,'test')
    server=d.Gateway(('0.0.0.0',0),guard,jobs,f'http://127.0.0.1:{upstream.server_port}','offline-secret-never-log',deadline_reader=reader)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    done=threading.Event()
    def pulse():
        while not done.wait(.2): beat(directory)
    threading.Thread(target=pulse,daemon=True).start()
    try: yield server,guard,calls
    finally:
        done.set();server.shutdown();server.server_close();upstream.shutdown();upstream.server_close()


def post(server,task='regression',**body):
    return d.httpx.post(f'http://127.0.0.1:{server.server_port}/{task}/v1/chat/completions',json=body,timeout=20)


def raw_client(server,task='regression',**body):
    raw=json.dumps(body).encode()
    sock=socket.create_connection(('127.0.0.1',server.server_port))
    sock.sendall(f'POST /{task}/v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n'.encode()+raw)
    return sock


def budget_tests(root):
    clock=[100.]
    path=root/'budget';path.mkdir()
    current=[snap()]
    guard=d.Guard(snap(),lambda:current[0],path/uuid_short(),clock=lambda:clock[0])
    assert guard.budget==180 and guard.threshold==162
    current[0]=snap(161);assert guard.refresh(True)
    current[0]=snap(162);assert not guard.refresh(True) and guard.closed
    current[0]=snap(0);assert not guard.refresh(True)
    small=d.Guard(snap(0,400),lambda:snap(),path/uuid_short(),clock=lambda:clock[0])
    assert small.budget==120 and small.threshold==108
    capped_current=[snap()]
    capped=d.Guard(snap(),lambda:capped_current[0],path/uuid_short(),clock=lambda:clock[0],budget_cap=100)
    assert capped.budget==100 and capped.threshold==90
    capped_current[0]=snap(89);assert capped.refresh(True)
    capped_current[0]=snap(90);assert not capped.refresh(True) and capped.closed
    current[0]=snap(0)
    delayed=d.Guard(snap(),lambda:current[0],path/uuid_short(),clock=lambda:clock[0])
    assert delayed.refresh(True)
    current[0]=snap(170);assert not delayed.refresh(True)
    for label, new in [('key_change',snap(key='other')),('period_reset',snap(period='period-2')),('counter_reset',snap(-1))]:
        current[0]=new
        guard=d.Guard(snap(),lambda:current[0],path/uuid_short(),clock=lambda:clock[0]);assert not guard.refresh(True) and guard.closed,label
    def fail():raise RuntimeError('offline')
    guard=d.Guard(snap(),fail,path/uuid_short(),clock=lambda:clock[0])
    assert not guard.refresh(True) and not guard.closed
    clock[0]+=59;assert not guard.refresh(True) and not guard.closed
    clock[0]+=1;assert not guard.refresh(True) and guard.closed
    heartbeat=d.Guard(snap(),lambda:snap(),path/uuid_short())
    beat(heartbeat.directory);assert heartbeat.allowed()
    d.write(heartbeat.directory/'observer_heartbeat.json',{'mono':time.monotonic()-16});assert not heartbeat.allowed() and heartbeat.closed
    beat(path)
    low=d.Guard(snap(),lambda:snap(),path/uuid_short(),disk=lambda:4*d.GIB);beat(low.directory);assert not low.allowed() and low.closed
    return {'budget':180,'threshold':162,'smaller_budget':120,'smaller_threshold':108,
        'scenarios':['joint_threshold','frozen_budget','fresh_remaining','delayed_counters','key_change','period_reset','counter_reset','accounting_loss_60s','heartbeat_expiry','disk_5gib']}


def queue_tests(root):
    release=threading.Event();started=threading.Event()
    def fn(handler,body,n):
        if n==1:started.set();release.wait(10)
        return 200,{'usage':{'prompt_tokens':1,'completion_tokens':2},'choices':[]}
    with gateway(root/'queue',fn) as (server,guard,calls):
        first=raw_client(server);assert started.wait(3)
        abandoned=raw_client(server,'classification');time.sleep(.15);abandoned.close()
        expired=raw_client(server,'classification',deadline=time.monotonic()+.15)
        time.sleep(.4);assert b'402' in expired.recv(4096);expired.close()
        assert len(calls)==1
        second=raw_client(server,'classification');time.sleep(.2);assert len(calls)==1
        release.set();assert b'200' in first.recv(4096);first.close();assert b'200' in second.recv(4096);second.close()
        assert len(calls)==2
        guard.close('test_budget');assert post(server).status_code==402 and len(calls)==2
        server.client_key='local-test-secret'
        assert post(server).status_code==403 and len(calls)==2
    return {'upstream_sends':2,'scenarios':['fifo_two_tasks','queued_disconnect','expired_before_send','serialization','permanent_402_no_upstream']}


def interrupted_csv(root,archive,semantic=False):
    task=root/'interrupted_csv';task.mkdir()
    with zipfile.ZipFile(archive) as package:package.extractall(task)
    fixture(task)
    seen=threading.Event();release=threading.Event()
    def hold(handler,body,n):
        if semantic:
            from tools.mock_task_resolution import fixture_resolution
            reply=fixture_resolution(body)
            if reply is not None:return 200,reply
        seen.set();release.wait(30)
        return 402,{'error':{'code':'synthetic_permanent_402'}}
    job={'kind':'regression','directory':str(task),'deadline_mono':time.monotonic()+10800}
    name='alpha-diag-interrupt-'+uuid_short()
    job['container']=name
    with gateway(root/'interrupted_gate',hold,exact=True,jobs={'regression':job}) as (server,guard,calls):
        try:
            d.docker('run','-d','--name',name,'--cpus','5','--memory','5g','--memory-swap','5g',
                '-v',str(task)+':/task','-w','/task','-e','OPENAI_API_KEY=offline-fixture',
                '-e',f'OPENAI_BASE_URL=http://host.docker.internal:{server.server_port}/regression/v1',
                '-e',f'AGENT_DIAGNOSTICS_URL=http://host.docker.internal:{server.server_port}/regression/diagnostics',d.IMAGE,'python','solution.py','task.md','--run-id','diagnostic')
            assert seen.wait(45),'request_not_received'
            publication=d.metadata(task,name)['publication']
            old=d.digest(task/'submission.csv')
            assert old==publication['csv_sha256'] and publication['scope']!='full_train_refit'
            d.docker('kill',name)
            assert d.digest(task/'submission.csv')==old
            state=json.loads(d.docker('inspect','--format','{{json .State}}',name).stdout)
            assert not state['Running']
            release.set();time.sleep(.2)
            assert len(calls)==(2 if semantic else 1)
            logs=d.docker('logs',name,check=False);(task/'container.log').write_text(logs.stdout+logs.stderr)
            return {'preserved_sha256':old,'scope_before_interruption':publication['scope'],'incomplete_run':True,'deadline_source':'wire_core.policies_resources'}
        finally:
            release.set();d.docker('rm','-f',name,check=False)
            with zipfile.ZipFile(archive) as package:
                for member in package.namelist():(task/member).unlink(missing_ok=True)
            shutil.rmtree(task/'.agent_runtime',ignore_errors=True)


def inflight_tests(root):
    ready=threading.Event();release=threading.Event()
    def late(handler,body,n):
        ready.set();release.wait(5)
        return 200,{'usage':{'total_tokens':3}}
    with gateway(root/'late',late) as (server,guard,calls):
        client=raw_client(server);assert ready.wait(3);client.close()
        queued=raw_client(server,'classification');time.sleep(.3);assert len(calls)==1
        release.set();assert b'200' in queued.recv(4096);queued.close();time.sleep(.2)
        events=d.events(guard.directory)
        assert any(x['kind']=='request_response' and x['late'] for x in events)
        assert len(calls)==2
    def broken(handler,body,n):
        handler.connection.shutdown(socket.SHUT_RDWR);handler.connection.close()
        return None
    with gateway(root/'unknown',broken) as (server,guard,calls):
        assert post(server).status_code==402
        assert post(server,'classification').status_code==402
        assert len(calls)==1 and server.unknown and guard.closed
    def hang(handler,body,n):time.sleep(1);return 200,{}
    with gateway(root/'deadline',hang) as (server,guard,calls):
        assert post(server,deadline=time.monotonic()+.3).status_code==402
        assert post(server,'classification').status_code==402
        assert len(calls)==1 and server.unknown
    return {'scenarios':['inflight_disconnect_holds_slot','late_reply_accounted','unknown_no_retries_no_new_send','wall_response_deadline']}


def reconciliation_tests(root):
    path=root/'reconciliation';path.mkdir();d.event(path,'request_unknown',task='regression')
    values=iter([snap(1),snap(2),snap(2),snap(2)])
    result=d.reconcile(path,'offline',snap(),interval=.01,duration=1,fetch=lambda:next(values))
    assert result['interval_increase']==2 and result['additional_since_first_after']==1
    assert result['stable_successful_samples']==3 and result['unknown_upstream_outcome']
    assert not result['accounting_complete'] and result['spend_status']=='preliminary'
    fail=root/'reconciliation_failed';fail.mkdir();d.event(fail,'test')
    def error():raise RuntimeError('offline')
    result=d.reconcile(fail,'offline',snap(),interval=.01,duration=.02,fetch=error)
    assert not result['remaining_confirmed'] and result['remaining'] is None and result['interval_increase'] is None
    return {'scenarios':['delayed_charge','first_after_delta','unknown_outcome_preliminary','last_check_error_unknown_remaining']}


def dependencies_and_resources(root):
    code='''import importlib.metadata as m,json,re,shutil,psutil,sys
rows=[(re.split(r"[=\\[]",s)[0],s.split("==")[1].strip()) for s in open("/requirements.txt") if "==" in s]
result=[{"package":p,"expected":v,"installed":m.version(p)} for p,v in rows]
print(json.dumps({"versions":result,"passed":all(x["installed"].split("+")[0]==x["expected"] for x in result),"python":sys.version,"disk":[shutil.disk_usage(p).free for p in ("/","/tmp")],"available_memory":psutil.virtual_memory().available}))'''
    result=json.loads(d.docker('run','--rm','--network','none','-v',str(d.ROOT/'data_for_participants/requirements.txt')+':/requirements.txt:ro',d.IMAGE,'python','-c',code).stdout)
    assert result['passed'] and result['python'].startswith('3.12')
    info=json.loads(d.docker('info','--format','{{json .}}').stdout)
    assert info['NCPU']>=10 and info['MemTotal']>=11*d.GIB
    assert min(result['disk']+[shutil.disk_usage(root).free,shutil.disk_usage('/tmp').free])>=15*d.GIB
    result.update(linux_cpus=info['NCPU'],linux_memory=info['MemTotal'],reserve=info['MemTotal']-10*d.GIB)
    return result


def exact_402(root,archive,semantic=False,one_kind=None):
    if semantic and one_kind is None:
        results={}
        for kind in ('regression','classification'):
            path=root/('semantic_402_'+kind);path.mkdir()
            results[kind]=exact_402(path,archive,semantic=True,one_kind=kind)
        return {'semantic_contract':True,'independent_recovery_per_task':results}
    jobs={};names=[]
    for kind in ((one_kind,) if one_kind else ('regression','classification')):
        task=root/'exact_402'/kind;task.mkdir(parents=True)
        with zipfile.ZipFile(archive) as package:package.extractall(task)
        fixture(task)
        # Both task kinds are covered by the live run; these small fixtures exercise
        # exact archive recovery without spending three hours on synthetic data.
        jobs[kind]={'kind':kind,'directory':str(task),'deadline_mono':time.monotonic()+10800}
    def provider(handler,body,n):
        from tools.mock_task_resolution import fixture_resolution
        reply=fixture_resolution(body)
        if reply is not None:return 200,reply
        guard.close('synthetic_permanent_402')
        return 402,{'error':{'code':'synthetic_permanent_402'}}
    with gateway(root/'exact_gate',provider,exact=True,jobs=jobs) as (server,guard,calls):
        if not semantic:guard.close('synthetic_permanent_402')
        try:
            for kind,job in jobs.items():
                name='alpha-diag-preflight-'+kind+'-'+uuid_short();names.append(name)
                job['container']=name
                d.docker('run','-d','--name',name,'--cpus','5','--memory','5g','--memory-swap','5g',
                    '-v',job['directory']+':/task','-w','/task','-e','OPENAI_API_KEY=offline-fixture',
                    '-e',f'OPENAI_BASE_URL=http://host.docker.internal:{server.server_port}/{kind}/v1',
                    '-e',f'AGENT_DIAGNOSTICS_URL=http://host.docker.internal:{server.server_port}/{kind}/diagnostics',d.IMAGE,'python','solution.py','task.md','--run-id','diagnostic')
            effective=[]
            until=time.monotonic()+180
            while True:
                running=[json.loads(d.docker('inspect','--format','{{json .State}}',n).stdout)['Running'] for n in names]
                if not any(running):break
                assert time.monotonic()<until,'exact402_timeout'
                time.sleep(1)
            for (kind,job),name in zip(jobs.items(),names):
                state=json.loads(d.docker('inspect','--format','{{json .State}}',name).stdout)
                logs=d.docker('logs',name,check=False);(Path(job['directory'])/'container.log').write_text(logs.stdout+logs.stderr)
                report=d.read(Path(job['directory'])/'.agent_work/diagnostic/run_report.json')
                assert state['ExitCode']==0 and report['state']==report['control_status']==report['refit_status']=='completed',report
                assert report['publication']['scope']=='full_train_refit'
                assert report['effective_cpu']==5 and abs(report['effective_memory_mb']-4352)<1
                # Fixture uses y; inspect directly rather than the public target checker.
                import csv,math
                with (Path(job['directory'])/'submission.csv').open() as stream:rows=list(csv.DictReader(stream))
                assert [r['id'] for r in rows]==[str(i) for i in range(100,112)] and all(math.isfinite(float(r['y'])) for r in rows)
                effective.append({'kind':kind,'cpu':report['effective_cpu'],'memory_mb':report['effective_memory_mb'],'report':report})
            assert len(calls)==(2 if semantic else 0)
            assert server.guard.closed
        finally:
            for name in names:d.docker('rm','-f',name,check=False)
            # Remove only reproducible code extracted by this preflight. Keep all
            # original journals, requests, model results, CSVs and process logs.
            with zipfile.ZipFile(archive) as package:
                members=package.namelist()
            for job in jobs.values():
                task=Path(job['directory'])
                shutil.rmtree(task/'.agent_runtime',ignore_errors=True)
                for member in members:(task/member).unlink(missing_ok=True)
    return {'organizer_requests':0,'synthetic_upstream_requests':len(calls),'concurrent_containers':len(jobs),'results':effective}


def uuid_short():
    import uuid
    return uuid.uuid4().hex[:8]


def watchdog_tests(root):
    path=root/'watchdog';path.mkdir()
    name='alpha-diag-watchdog-'+uuid_short()
    d.docker('run','-d','--name',name,'--network','none','--cpus','1','--memory','128m',d.IMAGE,
        'python','-c','import subprocess,time;subprocess.Popen(["python","-c","while True: pass"]);time.sleep(300)')
    job={'kind':'test','container':name,'started':True,'finished':False,'deadline_mono':time.monotonic()+200}
    d.write(path/'CONFIG.json',{'jobs':{'test':job}})
    # The producer is distinct from the watchdog and from this test driver.
    observer=subprocess.Popen([sys.executable,'-c',
        'import sys,time;from pathlib import Path;from tools.diagnostic_3h import write;p=Path(sys.argv[1]);\nwhile True:\n write(p/"observer_heartbeat.json",{"mono":time.monotonic()});write(p/"gateway_heartbeat.json",{"mono":time.monotonic()});time.sleep(.2)',str(path)],cwd=d.ROOT,start_new_session=True)
    watcher=subprocess.Popen([sys.executable,str(d.ROOT/'tools/diagnostic_3h.py'),'watchdog','--output',str(path)],cwd=d.ROOT,start_new_session=True)
    try:
        time.sleep(1);observer.kill();observer.wait()
        start=time.monotonic()
        while not (path/'test_watchdog_killed.json').exists():
            assert watcher.poll() is None,'watchdog_died_with_observer'
            assert time.monotonic()-start<80,'watchdog_grace_missed'
            time.sleep(.3)
        elapsed=time.monotonic()-start
        killed=d.read(path/'test_watchdog_killed.json')
        events=d.events(path)
        failed=next(e for e in events if e['kind']=='watchdog_infrastructure_failure')
        assert killed['reason']=='infrastructure_grace' and killed['mono']-failed['mono']<=60
        state=json.loads(d.docker('inspect','--format','{{json .State}}',name).stdout)
        assert not state['Running']
        assert not d.docker('top',name,check=False).stdout.strip().startswith('UID')
    finally:
        (path/'WATCHDOG_DONE').touch();watcher.wait(timeout=5)
        if observer.poll() is None:observer.kill();observer.wait()
        d.docker('rm','-f',name,check=False)
    # Short fault injection uses the same production loop, including suspended
    # observer, missing gateway, queued network client and active child training.
    outcomes=[]
    for mode in ('hung_observer','failed_gateway','hard_deadline','disk_2gib'):
        target=root/mode;target.mkdir()
        name='alpha-diag-fault-'+uuid_short()
        d.docker('run','-d','--name',name,'--network','none','--memory','128m',d.IMAGE,'python','-c',
            'import subprocess,time;subprocess.Popen(["python","-c","while True: pass"]);time.sleep(60)')
        job={'kind':'test','container':name,'started':True,'finished':False,'deadline_mono':time.monotonic()+(1 if mode=='hard_deadline' else 30)}
        d.write(target/'CONFIG.json',{'jobs':{'test':job}})
        d.write(target/'observer_heartbeat.json',{'mono':time.monotonic()+100})
        d.write(target/'gateway_heartbeat.json',{'mono':time.monotonic()+100})
        producer=None
        if mode in ('hung_observer','failed_gateway'):
            owner='observer' if mode=='hung_observer' else 'gateway'
            code='import time,sys;from pathlib import Path;from tools.diagnostic_3h import write;p=Path(sys.argv[1]);\nwhile True:\n write(p/sys.argv[2],{"mono":time.monotonic()});time.sleep(.05)'
            producer=subprocess.Popen([sys.executable,'-c',code,str(target),owner+'_heartbeat.json'],cwd=d.ROOT,start_new_session=True)
            time.sleep(.3)
            if mode=='hung_observer':os.kill(producer.pid,signal.SIGSTOP)
            else:producer.kill();producer.wait()
        code='from tools.diagnostic_3h import watchdog_main;from pathlib import Path;import sys;watchdog_main(Path(sys.argv[1]),poll=.05,stale=.2,grace=.5'+(',free_disk=lambda:1' if mode=='disk_2gib' else '')+')'
        watcher=subprocess.Popen([sys.executable,'-c',code,str(target)],cwd=d.ROOT,start_new_session=True)
        try:
            end=time.monotonic()+5
            while not (target/'test_watchdog_killed.json').exists():
                assert time.monotonic()<end;time.sleep(.05)
            assert not json.loads(d.docker('inspect','--format','{{json .State}}',name).stdout)['Running']
            outcomes.append({'mode':mode,'result':d.read(target/'test_watchdog_killed.json')})
        finally:
            (target/'WATCHDOG_DONE').touch();watcher.wait(timeout=5);d.docker('rm','-f',name,check=False)
            if producer is not None and producer.poll() is None:producer.kill();producer.wait()
    return {'observer_force_killed':True,'watchdog_survived':True,'actual_60_second_grace_elapsed':elapsed,'faults':outcomes}


def main(output,archive,semantic=False):
    output.mkdir(parents=True,exist_ok=False)
    report={'archive_sha256':d.digest(archive),'organizer_requests':0,'cases':[],'started':d.utc()}
    tests=[('archive',lambda:d.verify_archive(archive)['sha256']),('budget',lambda:budget_tests(output)),
        ('queue',lambda:queue_tests(output)),('inflight',lambda:inflight_tests(output)),
        ('reconciliation',lambda:reconciliation_tests(output)),('dependencies_resources',lambda:dependencies_and_resources(output)),
        ('exact_402_recovery',lambda:exact_402(output,archive,semantic=semantic)),('interrupted_csv',lambda:interrupted_csv(output,archive,semantic=semantic)),
        ('independent_watchdog',lambda:watchdog_tests(output))]
    for name,fn in tests:
        started=time.monotonic()
        try:
            evidence=fn();row={'name':name,'passed':True,'seconds':time.monotonic()-started,'evidence':evidence}
        except Exception as exc:
            row={'name':name,'passed':False,'seconds':time.monotonic()-started,'error':str(exc),'traceback':traceback.format_exc()}
        report['cases'].append(row);d.write(output/'REPORT.json',report)
        print(json.dumps(row,ensure_ascii=False),flush=True)
        if not row['passed']:break
    report['passed']=len(report['cases'])==len(tests) and all(row['passed'] for row in report['cases'])
    report['finished']=d.utc();report['supervisor_sha256']=d.digest(d.ROOT/'tools/diagnostic_3h.py');report['supervisor_sources']=d.supervisor_sources()
    d.write(output/'REPORT.json',report)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--semantic-contract',action='store_true')
    args=p.parse_args();raise SystemExit(main(args.output.resolve(),args.archive.resolve(),semantic=args.semantic_contract))
