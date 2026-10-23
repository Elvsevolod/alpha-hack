"""Real-clock two-task stress rehearsal; all Qwen/accounting traffic is local.

Synthetic diagnostic_load records exercise run02 volume independently from ML
results. They never affect training, metrics, selection, or the agent journal.
"""
from __future__ import annotations
import argparse
import csv
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import diagnostic_3h as d
from tools.diagnostic_store import Store
from tools.test_diagnostic_3h import fixture

MIN_EVENTS=20910
MIN_ARTIFACTS=16138
MIN_SQLITE=17690624
OFFLINE_INFO={'spend':0,'total_spend':0,'max_budget':600,'blocked':False,'budget_reset_at':'offline-period'}


def source_tasks(root):
    for folder,target,kind in [('Регрессия','site_eui','regression'),('Многоклассовая классификация','attack_cat','classification')]:
        task=root/folder;task.mkdir(parents=True);fixture(task)
        for name in ('train.csv','sample_submission.csv'):
            with (task/name).open() as stream:rows=list(csv.reader(stream))
            rows[0][-1]=target
            if kind=='classification':
                for row in rows[1:]:row[-1]='A' if float(row[-1])>=0 else 'B'
            with (task/name).open('w') as stream:csv.writer(stream).writerows(rows)
        (task/'task_spec.json').unlink()
        (task/'task.md').write_text(f'Synthetic {kind}. Target: `{target}`. Primary metric: '+('RMSE' if kind=='regression' else 'Macro F1'))
        (task/'data_dictionary.csv').write_text('feature,description\nx,Synthetic numeric feature\n')


def load_history(directory,stop,seconds,errors):
    try:
        store=None
        while not stop.wait(.1):
            if (directory/'CONFIG.json').exists() and (directory/'history/ledger.sqlite').exists():
                store=Store(directory);break
        if store is None:return
        # Exercise a large, consistent SQLite source without changing execution truth.
        jobs={};begin=time.monotonic()
        while len(jobs)<2 and not stop.is_set():
            jobs=d.read(directory/'CONFIG.json',{}).get('jobs',{})
            if not all((Path(j['directory'])/'.agent_work/diagnostic/journal.sqlite').exists() for j in jobs.values()) or len(jobs)<2:
                stop.wait(.1);jobs={};continue
        if not jobs:return
        sizes={}
        for kind,job in jobs.items():
            path=Path(job['directory'])/'.agent_work/diagnostic/diagnostic_load.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE load_records(id INTEGER PRIMARY KEY, record TEXT)')
                for i in range(2000):
                    # Representative long JSON result records, explicitly synthetic.
                    row={'kind':'diagnostic_load','index':i,'task':kind,'trials':[
                        {'trial_id':f'{i}_{n}','status':'completed','parameters':{'alpha':1+n/10},
                         'hypothesis':'Synthetic bounded audit throughput fixture','score':n/1000}
                        for n in range(75)]}
                    db.execute('INSERT INTO load_records VALUES (?,?)',(i,json.dumps(row)))
            sizes[kind]=path.stat().st_size
            assert sizes[kind]>=MIN_SQLITE,sizes
        d.write(directory/'LOAD_SQLITE.json',{'synthetic':True,'bytes':sizes})
        # Complete the threshold workload during the first 90% of the wall run.
        window=max(1.,seconds*.90)
        for i in range(MIN_EVENTS):
            if stop.is_set():break
            task=('regression','classification')[i%2]
            if i<MIN_ARTIFACTS:
                store.receive(task,{'kind':'diagnostic_load','event_id':f'synthetic_load_{i}',
                    'payload':{'synthetic':True,'index':i,'trace':{'status':'observed','measurement':i/7,
                        'explanation':'Synthetic throughput record; excluded from ML quality and experiment counts'}}})
            else:store.append('diagnostic_load_interval',task=task,synthetic=True,index=i)
            if i%100==0:
                d.write(directory/'LOAD_PROGRESS.json',{'synthetic':True,'events_generated':i+1,
                    'artifacts_generated':min(i+1,MIN_ARTIFACTS),'elapsed_seconds':time.monotonic()-begin})
            target=begin+window*(i+1)/MIN_EVENTS
            stop.wait(max(0.,target-time.monotonic()))
        d.write(directory/'LOAD_PROGRESS.json',{'synthetic':True,'events_generated':i+1,
            'artifacts_generated':min(i+1,MIN_ARTIFACTS),'elapsed_seconds':time.monotonic()-begin})
    except Exception as error:
        errors.append(type(error).__name__+':'+str(error))
        (directory/'CLOSE').touch()


def run(output,archive,checks,seconds=10800,pace=20,port=18845):
    output.mkdir(parents=True,exist_ok=False)
    source=output/'source';source_tasks(source)
    calls=[];started=time.monotonic();halt=threading.Event();errors=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            number=len(calls)+1;calls.append({'number':number,'mono':time.monotonic()})
            # A real response delay exercises serialized queueing for both tasks.
            delay=min(pace,max(0.,seconds-340-(time.monotonic()-started)))
            for message in body.get('messages',[]):
                if message.get('role')!='user':continue
                try:resources=json.loads(message.get('content','')).get('core',{}).get('policies_resources',{})
                except (ValueError,TypeError):continue
                if resources:
                    delay=min(delay,max(0.,resources['deadline_epoch']-time.time()-resources['reserve_seconds']-10))
            halt.wait(delay)
            args={'operation_id':f'load_observe_{number}','report':'missing'}
            if body.get('tools'):
                message={'tool_calls':[{'id':str(number),'type':'function','function':{
                    'name':'get_dataset_report','arguments':json.dumps(args)}}]}
            else:message={'content':json.dumps({'tool':'get_dataset_report','operation_id':args.pop('operation_id'),'arguments':args})}
            response={'id':'offline_'+str(number),'choices':[{'finish_reason':'stop','message':message}],
                'usage':{'prompt_tokens':1,'completion_tokens':2}}
            content=json.dumps(response).encode()
            try:self.send_response(200);self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
            except OSError:pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    original_popen=subprocess.Popen;original_reconcile=d.reconcile
    def spawn(arguments,*args,**kwargs):
        if isinstance(arguments,list) and len(arguments)>=3 and arguments[1]==str(Path(d.__file__).resolve()) and arguments[2]=='gateway':
            code='from pathlib import Path;from tools import diagnostic_3h as d;import sys;d.limits.fetch_info=lambda key:'+repr(OFFLINE_INFO)+';d.gateway_main(Path(sys.argv[1]),int(sys.argv[2]))'
            directory=arguments[arguments.index('--output')+1];gateway_port=arguments[arguments.index('--port')+1]
            arguments=[sys.executable,'-c',code,directory,gateway_port]
        return original_popen(arguments,*args,**kwargs)
    def reconcile(*args,**kwargs):return original_reconcile(*args,**kwargs,interval=.05,duration=.5)
    directory=output/('run_'+output.name[-12:]+'_'+uuid.uuid4().hex[:8])
    d.write(output/'RUN_LOCATION.json',{'directory':str(directory)})
    loader=threading.Thread(target=load_history,args=(directory,halt,seconds,errors),daemon=True);loader.start()
    try:
        environment={'OPENAI_API_KEY':'offline-fixture','OPENAI_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1','OPENAI_MODEL':'qwen/qwen3.8-27b'}
        with patch.dict(os.environ,environment),patch.object(d.limits,'fetch_info',lambda key:OFFLINE_INFO),patch.object(subprocess,'Popen',spawn),patch.object(d,'reconcile',reconcile):
            status=d.run(directory,archive,checks,port,source=source,seconds=seconds,cpu=5,memory_gib=5,api_budget=1)
        # Retain the observation window for the full configured wall duration,
        # including any slack after the agents' measured finalization reserve.
        if status==0:
            halt.wait(max(0.,seconds-(time.monotonic()-started)))
        report=d.read(directory/'REPORT.json',{})
        history=Store(directory).verify()
        load=d.read(directory/'LOAD_PROGRESS.json',{})
        copies=[a for a in Store(directory).index()['artifacts'].values() if str(a['identity']).startswith('sqlite_backup_')]
        largest=max([a['bytes'] for a in copies]+[0])
        elapsed=time.monotonic()-started
        passed=(status==0 and seconds==10800 and elapsed>=seconds and history['records']>=MIN_EVENTS
            and history['artifacts']>=MIN_ARTIFACTS and largest>=MIN_SQLITE and not errors
            and load.get('events_generated',0)>=MIN_EVENTS and report.get('full_run_passed'))
        result={'passed':bool(passed),'mode':'offline_real_clock_rehearsal','organizer_requests':0,
            'archive_sha256':d.digest(archive),'supervisor_sha256':d.digest(Path(d.__file__)),
            'configured_seconds':seconds,'elapsed_seconds':elapsed,'mock_calls':len(calls),'history':history,
            'synthetic_load':load,'largest_verified_sqlite_copy_bytes':largest,'errors':errors,
            'supervisor_exit':status,'two_tasks_completed':all(r['technical_passed'] for r in report.get('results',[])) and len(report.get('results',[]))==2,
            'scope':'Synthetic regression/classification fixtures and labelled diagnostic load; no public/private quality conclusion'}
        d.write(output/'REPORT.json',result)
        print(json.dumps(result),flush=True)
        return 0 if passed else 1
    finally:
        halt.set();loader.join(timeout=5);server.shutdown();server.server_close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--checks',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=10800);parser.add_argument('--pace',type=float,default=20)
    parser.add_argument('--port',type=int,default=18845);args=parser.parse_args()
    raise SystemExit(run(args.output.resolve(),args.archive.resolve(),args.checks.resolve(),args.seconds,args.pace,args.port))
