"""Exact-archive diagnostic logging acceptance, using only synthetic HTTP."""
import argparse
import base64
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import diagnostic_3h as d
from tools.diagnostic_store import Store,scan_secrets,register_secret
from tools.test_diagnostic_3h import gateway,fixture,post,uuid_short
from tools.mock_task_resolution import fixture_resolution


def store_tests(root):
    directory=root/'storage';directory.mkdir()
    secret='sk-syntheticOnlySecret0123456789'
    register_secret(secret)
    store=Store(directory,segment_max=4096)
    cpu=time.process_time();started=time.monotonic()
    for i in range(40):store.append('test',task='test',counter=i,secret=secret)
    artifact=store.put(('HTTP failure '+secret+'\nTraceback\n'+secret).encode(),name='error.txt')
    assert store.verify()['passed'] and len(store.index()['segments'])>1
    assert scan_secrets(directory)['passed']
    overhead={'write_wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu,'bytes':store.index()['bytes']}
    last=store.index()['segments'][-1]
    with (store.root/last['path']).open('ab') as stream:stream.write(b'incomplete-tail')
    try:store.verify();raise AssertionError('tail_not_detected')
    except ValueError as error:assert 'tail' in str(error)
    # The last confirmed byte range remains independently readable.
    packed=(store.root/last['path']).read_bytes()[:last['bytes']]
    assert gzip.decompress(packed)
    failed=root/'storage_limit';failed.mkdir()
    bounded=Store(failed,total_max=100)
    try:bounded.put(os.urandom(200),name='overflow');raise AssertionError('missing_limit')
    except ValueError:pass
    assert (failed/'CLOSE').exists() and (failed/'DIAGNOSTIC_LIMIT.json').exists()
    # Kill a real writer while it appends. The index is atomic; any extra tail
    # is detectable and the preceding committed segment remains verifiable.
    crash=root/'crash_writer';crash.mkdir()
    code='from tools.diagnostic_store import Store;from pathlib import Path;import sys;s=Store(Path(sys.argv[1]));\nwhile True:s.append("writing",value="x"*50000)'
    writer=subprocess.Popen([sys.executable,'-c',code,str(crash)],cwd=d.ROOT,start_new_session=True)
    end=time.monotonic()+5
    while not (crash/'history/INDEX.json').exists():assert time.monotonic()<end;time.sleep(.01)
    time.sleep(.05);writer.kill();writer.wait()
    index=Store(crash).index()
    for segment in index['segments']:
        content=(crash/'history'/segment['path']).read_bytes()[:segment['bytes']]
        assert d.hashlib.sha256(content).hexdigest()==segment['sha256']
        assert gzip.decompress(content)
    return {'rotation':True,'secret_cleaning':True,'corrupt_tail_detected':True,'real_writer_killed':True,'committed_index_readable':True,'storage_limit_explicit':True,'overhead':overhead}


def sqlite_crash(root):
    path=root/'sqlite_crash';path.mkdir()
    source=sqlite3.connect(path/'source.sqlite');source.execute('PRAGMA journal_mode=WAL');source.execute('CREATE TABLE data(x)')
    source.executemany('INSERT INTO data VALUES (?)',[(i,) for i in range(10000)]);source.commit()
    with sqlite3.connect(path/'confirmed.sqlite') as confirmed:source.backup(confirmed)
    code='import sqlite3,sys,time;from pathlib import Path;p=Path(sys.argv[1]);a=sqlite3.connect(p/"source.sqlite");b=sqlite3.connect(p/"incomplete.pending");a.backup(b,pages=1,progress=lambda *x:time.sleep(.05))'
    copier=subprocess.Popen([sys.executable,'-c',code,str(path)],start_new_session=True)
    time.sleep(.1);copier.kill();copier.wait()
    with sqlite3.connect(path/'confirmed.sqlite') as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok' and db.execute('SELECT COUNT(*) FROM data').fetchone()[0]==10000
    source.close()
    return {'confirmed_backup_readable':True,'partial_file_detected':(path/'incomplete.pending').exists()}


def output_rotation(root):
    from automl_agent.diagnostics import RotatingOutput
    path=root/'output_rotation';path.mkdir()
    secret='sk-artificialWorkerSecret0123456789';register_secret(secret)
    body=(b'line '+b'x'*1000+b'\n')*12000+secret.encode()+b'\nEND\n'
    output=RotatingOutput(io.BytesIO(body),path);assert output.finish() is None
    combined=b''.join(gzip.decompress(p.read_bytes()) for p in output.segments)
    assert len(combined)>10*1024**2 and combined.endswith(b'\nEND\n') and secret.encode() not in combined
    assert len(output.segments)>10 and scan_secrets(path)['passed']
    import automl_agent.diagnostics as diagnostic
    previous=diagnostic.LOG_TOTAL;diagnostic.LOG_TOTAL=100
    try:
        limit=root/'output_limit';limit.mkdir();limited=RotatingOutput(io.BytesIO(b'x'*10000),limit);assert limited.finish()=='log_limit'
    finally:diagnostic.LOG_TOTAL=previous
    return {'input_bytes':len(body),'saved_clean_bytes':len(combined),'segments':len(output.segments),'continued_over_10mib':True,'limit_reason':'log_limit'}


def provider_case(body,n):
    # All paths go through the exact archive's parser and schema validation.
    command={'tool':'get_dataset_report','operation_id':f'observe_{n}','arguments':{'report':'missing'}}
    if body.get('tools'):
        message={'tool_calls':[{'id':f'call_{n}','type':'function','function':{'name':command['tool'],'arguments':json.dumps({'operation_id':command['operation_id'],**command['arguments']})}}]}
    else:message={'content':json.dumps(command)}
    response={'id':f'offline_response_{n}','choices':[{'finish_reason':'stop','message':message}],
              'usage':{'prompt_tokens':10,'completion_tokens':20,'completion_tokens_details':{'reasoning_tokens':5}}}
    if n==2:response['choices'][0]['message']['tool_calls']*=2
    if n==3:return 200,b'{invalid JSON sk-providerSecretSynthetic123456}'
    if n==4:response['choices'][0]['message']={'content':'{"tool":"does_not_exist","operation_id":"bad","arguments":{}}'}
    if n==5:response['choices'][0]['finish_reason']='length'
    if n==6:return 429,{'error':'synthetic rate limit'}
    if n==7:return 503,{'error':'synthetic server failure'}
    if n==8:response.pop('usage')
    if n>=9:return 402,{'error':'synthetic permanent refusal'}
    return 200,response


def exact_history(root,archive):
    path=root/'exact_history';path.mkdir()
    task=path/'task';task.mkdir()
    with zipfile.ZipFile(archive) as package:package.extractall(task)
    fixture(task)
    job={'kind':'regression','directory':str(task),'deadline_mono':time.monotonic()+10800}
    name='alpha-diag-history-'+uuid_short()
    job['container']=name
    with gateway(root/'history_gate',lambda h,b,n:(200,fixture_resolution(b)) if fixture_resolution(b) is not None else provider_case(b,n-1),exact=True,jobs={'regression':job}) as (server,guard,calls):
        try:
            d.docker('run','-d','--name',name,'--cpus','5','--memory','5g','--memory-swap','5g',
                '-v',str(task)+':/task','-w','/task','-e','OPENAI_API_KEY=offline-fixture',
                '-e',f'OPENAI_BASE_URL=http://host.docker.internal:{server.server_port}/regression/v1',
                '-e',f'AGENT_DIAGNOSTICS_URL=http://host.docker.internal:{server.server_port}/regression/diagnostics',
                d.IMAGE,'python','solution.py','task.md','--run-id','diagnostic')
            end=time.monotonic()+180
            while json.loads(d.docker('inspect','--format','{{json .State}}',name).stdout)['Running']:
                assert time.monotonic()<end,'exact_history_timeout';time.sleep(.5)
            state=json.loads(d.docker('inspect','--format','{{json .State}}',name).stdout)
            logs=d.docker('logs',name,check=False);(path/'container.log').write_text(logs.stdout+logs.stderr)
            report=d.read(task/'.agent_work/diagnostic/run_report.json')
            assert state['ExitCode']==0 and report['state']==report['control_status']==report['refit_status']=='completed',report
            records=list(server.store.records());kinds={r['kind'] for r in records}
            assert {'qwen_exchange_start','qwen_parse_outcome','operation_started','operation_record','experiment_record','worker_started','worker_finished','phase_transition','snapshot_verified','response_saved','request_link'}<=kinds
            # The malformed third research response has an unknown generation
            # outcome. A correct agent preserves it and finalizes without retry.
            saved=[r for r in records if r['kind']=='response_saved'];assert len(saved)==len(calls)==4
            assert report['qwen_unknown_outcome']
            assert any(r['payload']['usage'] is None for r in saved)
            links=[r['payload'] for r in records if r['kind']=='request_link']
            assert all(r['call_id'] is not None for r in links),links
            assert all(r['request_id'] is not None for r in links[1:]),links
            # Reparse only saved bodies; do not call the model or train again.
            from context_compressor.transport import Transport
            from automl_agent.contracts import TOOL_SCHEMAS
            transport=Transport(system='offline replay',tool_schemas={k:v.model_json_schema() for k,v in TOOL_SCHEMAS.items()})
            requests={r['payload']['request']:r['payload'] for r in records if r['kind']=='request_link'}
            replay=[]
            for r in saved:
                payload=r['payload'];item=requests[payload['request']]
                raw=gzip.decompress((server.store.root/item['request_artifact']).read_bytes());wire=json.loads(raw)
                mode='native' if wire.get('tools') else 'json'
                try:
                    response=json.loads(gzip.decompress((server.store.root/r['artifact']).read_bytes()))
                    if payload['status']!=200:raise ValueError('http_error')
                    command=transport.parse_response(response,mode)
                    TOOL_SCHEMAS[command.tool].model_validate(command.as_dict()['arguments'])
                    accepted=True
                except Exception:accepted=False
                replay.append({'call_id':item['call_id'],'accepted':accepted,'status':payload['status']})
            assert not replay[0]['accepted'] and replay[1]['accepted'] and not replay[3]['accepted']
            verified=server.store.verify();assert verified['passed'] and scan_secrets(guard.directory)['passed']
            d.write(path/'REPLAY.json',replay)
            return {'calls':len(calls),'archive_sha256':d.digest(archive),'state':report['state'],'control':report['control_status'],
                'refit':report['refit_status'],'csv_scope':report['publication']['scope'],'replay':replay,'history':verified,
                'available_bodies_and_headers_saved':True,'missing_usage_unknown':True,
                'unknown_generation_stopped_without_retry':True}
        finally:
            d.docker('rm','-f',name,check=False)
            with zipfile.ZipFile(archive) as package:
                for member in package.namelist():(task/member).unlink(missing_ok=True)
            shutil.rmtree(task/'.agent_runtime',ignore_errors=True)


def exact_worker_probes(root,archive):
    task=root/'worker_probes';task.mkdir()
    with zipfile.ZipFile(archive) as package:package.extractall(task)
    fixture(task)
    job={'kind':'regression','directory':str(task),'deadline_mono':time.monotonic()+10800}
    code='''from pathlib import Path
import sys,json,shutil
from agent_bundle import materialize
sys.path.insert(0,str(materialize(Path.cwd())))
from automl_agent.controller import Controller
from automl_agent.budget import Limits
from automl_agent.contracts import PipelineSpec,ToolCommand,Hypothesis
from automl_agent.tool_results import payload
from automl_agent.cleanup import reclaim
c=Controller(Path('task.md'),Limits(seconds=180,cpu=5,memory_mb=5120),run_id='diagnostic')
c.prepare()
c.qwen.critique=lambda request,**kw:dict(review_id=request['review_id'],verdict='approve',evaluation_scope='research_validation',
    comparator_id=c.journal.get('baseline_id'),evidence_assessment='Fixed offline fixture',comparison_assessment='Use the saved baseline',
    leakage_assessment='Fit rows only',cost_assessment='Bounded technical check',falsification_assessment='Require valid diagnostics',revision_requests=[])
prior=c.journal.get('publication')
def execute(operation,source):
    return payload(c.dispatch(ToolCommand(operation_id=operation,tool='run_custom_experiment',arguments={
        'pipeline':PipelineSpec(engine='custom',model='ridge',custom_component='transformer').model_dump(mode='json'),
        'hypothesis':Hypothesis(observation='Synthetic diagnostics test',expected_effect='Preserve inputs',changed_component='stdout',acceptance='Diagnostic preservation').model_dump(),
        'code':source,'budget_seconds':30})))
source="class Features:\\n def fit(self,X,y=None):\\n  print('BEGIN'+ 'x'*12000000 + 'END')\\n  return self\\n def transform(self,X):return X\\ndef build_transformer(context):return Features()\\n".replace('\\\\n','\\n')
good=execute('noisy',source)
assert good['result']['status']=='completed',good
identifier=good['experiment']['experiment_id']
c.journal.put('leader_id',c.journal.get('baseline_id'));c.journal.put('publication',prior)
c.journal.put('finalist_pool_ids',[c.journal.get('baseline_id')])
usage=shutil.disk_usage
measured=usage(c.workdir)
shutil.disk_usage=lambda *args:measured._replace(free=0)
try:reclaim(c,1)
finally:shutil.disk_usage=usage
directory=c.workdir/'attempts'/identifier
assert not (directory/'validation_predictions.json').exists()
bad=execute('bad',"class Features:\\n def fit(self,X,y=None):raise RuntimeError('sk-fakeTracebackSecret0123456789')\\n def transform(self,X):return X\\ndef build_transformer(context):return Features()\\n".replace('\\\\n','\\n'))
assert bad['result']['status']!='completed'
import automl_agent.diagnostics as diagnostic
diagnostic.LOG_TOTAL=1024
limited=execute('limited',source.replace('12000000','12000001'))
assert limited['result']['status']!='completed'
assert c.lookup(limited['experiment']['experiment_id'])[1].training_summary['diagnostic']['reason']=='log_limit',limited
diagnostic.LOG_TOTAL=256*1024**2
def tune(operation,model,space,seconds):
    return payload(c.dispatch(ToolCommand(operation_id=operation,tool='tune_model',arguments={
        'pipeline':PipelineSpec(engine='optuna',model=model,search_owner='optuna',n_trials=2,search_space=space).model_dump(mode='json'),
        'hypothesis':Hypothesis(observation='Fault injection',expected_effect='Record attempts',changed_component='trial',acceptance='History retained').model_dump(),
        'budget_seconds':seconds})))
failed_trial=tune('failed_trial','ridge',{'alpha':{'type':'categorical','choices':[-1.0]}},8)
interrupted_trial=tune('interrupted_trial','extra_trees',{'n_estimators':{'type':'categorical','choices':[100000]}},3)
c.freeze('Completed offline diagnostic fault injection');c.finalize();c.close()
other=Controller(Path('task.md'),Limits(seconds=30,cpu=5,memory_mb=5120),run_id='controller_fault')
def explode():raise RuntimeError('sk-fakeControllerSecret0123456789')
other.prepare=explode
fault=other.run(no_llm=True);assert fault['state']=='failed';other.close()
print(json.dumps({'good':good,'bad':bad,'limited':limited}))
'''
    with gateway(root/'worker_gate',lambda *x:(402,{}),exact=True,jobs={'regression':job}) as (server,guard,calls):
        name='alpha-diag-workers-'+uuid_short()
        job['container']=name
        try:
            d.docker('run','-d','--name',name,'--cpus','5','--memory','5g','-v',str(task)+':/task','-w','/task',
                '-e','OPENAI_API_KEY=offline-fixture','-e',f'AGENT_DIAGNOSTICS_URL=http://host.docker.internal:{server.server_port}/regression/diagnostics',
                d.IMAGE,'python','-c',code)
            end=time.monotonic()+160
            while json.loads(d.docker('inspect','--format','{{json .State}}',name).stdout)['Running']:
                assert time.monotonic()<end;time.sleep(.5)
            state=json.loads(d.docker('inspect','--format','{{json .State}}',name).stdout)
            logs=d.docker('logs',name,check=False);(task/'probe.log').write_bytes(d.scrub((logs.stdout+logs.stderr).encode()))
            assert state['ExitCode']==0,(logs.stdout+logs.stderr)[-2500:]
            artifacts=server.store.index()['artifacts'].values()
            assert any(a['name']=='validation_predictions.json' for a in artifacts)
            assert any(a['name'].startswith('worker.log.') for a in artifacts)
            trials=[a for a in artifacts if a['name']=='diagnostic_trials.jsonl']
            assert trials,'missing_attempt_history'
            trial_text='\n'.join(gzip.decompress((server.store.root/a['path']).read_bytes()).decode() for a in trials)
            assert 'trial_started' in trial_text and 'trial_finished' in trial_text and 'failed' in trial_text
            controller_errors=[json.loads(gzip.decompress((server.store.root/r['artifact']).read_bytes()))['payload']
                for r in server.store.records() if r['kind']=='controller_error']
            assert any(e['category']=='controller' and e['error_code'].startswith('controller:') and 'Traceback' in e['traceback'] for e in controller_errors)
            assert server.store.verify()['passed'] and scan_secrets(guard.directory)['passed']
            return {'exact_archive_worker_over_10mib':True,'loser_snapshot_before_cleanup':True,
                'worker_traceback':True,'controller_traceback':True,'failed_and_budget_limited_trials_retained':True,
                'log_limit_explicit':True,'history':server.store.verify()}
        finally:
            d.docker('rm','-f',name,check=False)
            with zipfile.ZipFile(archive) as package:
                for member in package.namelist():(task/member).unlink(missing_ok=True)
            shutil.rmtree(task/'.agent_runtime',ignore_errors=True)


def main(output,archive):
    output.mkdir(parents=True,exist_ok=False)
    report={'archive_sha256':d.digest(archive),'organizer_requests':0,'cases':[]}
    for name,fn in [('store',lambda:store_tests(output)),('sqlite_crash',lambda:sqlite_crash(output)),
        ('output_rotation',lambda:output_rotation(output)),('exact_saved_history',lambda:exact_history(output,archive)),
        ('exact_worker_probes',lambda:exact_worker_probes(output,archive))]:
        started=time.monotonic()
        try:row={'name':name,'passed':True,'evidence':fn(),'seconds':time.monotonic()-started}
        except Exception as exc:row={'name':name,'passed':False,'error':str(exc),'traceback':traceback.format_exc()}
        report['cases'].append(row);d.write(output/'REPORT.json',report);print(json.dumps(row,ensure_ascii=False),flush=True)
        if not row['passed']:break
    report['passed']=len(report['cases'])==5 and all(r['passed'] for r in report['cases'])
    d.write(output/'REPORT.json',report)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    args=p.parse_args();raise SystemExit(main(args.output.resolve(),args.archive.resolve()))
