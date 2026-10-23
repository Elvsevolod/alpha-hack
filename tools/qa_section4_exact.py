"""Offline CLI checks of the exact flat ZIP, using only staged synthetic data."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(archive,output):
    output.mkdir(exist_ok=False,parents=True)
    source=output/'source';source.mkdir()
    with zipfile.ZipFile(archive) as package:package.extractall(source)
    syntax=ast.parse((source/'ml_knowledge_bundle.py').read_text())
    manifest=next(ast.literal_eval(n.value) for n in syntax.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MANIFEST' for t in n.targets))
    results=[]
    for kind,metric in [('regression','mape'),('classification','balanced_accuracy'),('clustering','silhouette'),('semantic','rmse'),('semantic_empty','rmse'),('semantic_blank','rmse')]:
        task=output/kind;task.mkdir()
        for path in source.iterdir():os.link(path,task/path.name)
        rng=np.random.default_rng(42);x=np.r_[rng.normal(-5,.1,80),rng.normal(5,.1,80)]
        train=pd.DataFrame({'id':[f'{i:05d}' for i in range(len(x))],'x':x,'z':rng.normal(size=len(x))})
        future=pd.DataFrame({'id':['00900','00700','00800'],'x':[-5.,5.,-5.1],'z':[0.,0.,.1]})
        semantic=kind.startswith('semantic');numeric=kind=='regression' or semantic
        if kind!='clustering':train['answer']=2*x if numeric else np.where(x<0,'negative','positive')
        train.to_csv(task/'observations.csv',index=False);future.to_csv(task/'future.csv',index=False)
        (task/'task.md').write_text('Explicit synthetic mechanism fixture; no external sources.')
        if kind=='clustering':
            spec={'kind':'clustering','train_file':'observations.csv','test_file':'future.csv','features':['x','z'],'id_columns':['id'],'n_clusters':2}
            (task/'clustering.json').write_text(json.dumps(spec));extra=['--clustering-contract','clustering.json']
        else:
            spec={'contract_version':2,'file_roles':{'train.csv':'observations.csv','test.csv':'future.csv'},'kind':'regression' if numeric else kind,'targets':['answer'],'id_columns':['id'],'feature_columns':['x','z'],'metric':{'name':metric},'output':{'columns':['id','answer'],'prediction_type':'values' if numeric else 'labels'}}
            if kind=='classification':spec['classes']={'answer':['negative','positive']}
            if not semantic:(task/'task_spec.json').write_text(json.dumps(spec))
            extra=[]
        command=[sys.executable,'solution.py','task.md','--no-llm','--time-limit','90','--cpu','1','--memory-mb','768','--max-candidates','0','--run-id','section4',*extra]
        server=None;semantic_calls=[];environment=dict(os.environ)
        if semantic:
            command.remove('--no-llm')
            class Handler(BaseHTTPRequestHandler):
                def log_message(self,*args):pass
                def do_POST(self):
                    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    request=json.loads(body['messages'][-1]['content'])
                    schema=request['schema']['$defs']['TaskSpec']
                    assert schema['properties']['contract_version']=={'type':'integer','const':2} and 'contract_version' in schema['required']
                    assert not (task/'task_spec.json').exists()
                    semantic_calls.append(request['request_id'])
                    confirmed={**spec,'source_hash':request['source_hash']}
                    proposal={'request_id':request['request_id'],'source_hash':request['source_hash'],'status':'proposed','task_spec':confirmed,'evidence':{field:['task','inventory'] for field in confirmed}}
                    content=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':json.dumps(proposal)}}]}).encode()
                    self.send_response(200);self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler);Thread(target=server.serve_forever,daemon=True).start()
            environment={key:value for key,value in environment.items() if not key.startswith('OPENAI_') and not key.startswith('QWEN_')}
            environment.update(OPENAI_API_KEY='offline-fixture',OPENAI_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1')
            if kind=='semantic_empty':environment['OPENAI_MODEL']=''
            if kind=='semantic_blank':environment['OPENAI_MODEL']='   '
        try:completed=subprocess.run(command,cwd=task,env=environment,capture_output=True,text=True,timeout=110)
        finally:
            if server:server.shutdown();server.server_close()
        (task/'cli.log').write_text(completed.stdout+completed.stderr)
        assert completed.returncode==0,(kind,completed.returncode,(completed.stdout+completed.stderr)[-3000:])
        csv=pd.read_csv(task/'submission.csv',dtype={'id':str});assert csv.id.tolist()==future.id.tolist() and len(csv)==3 and not csv.isna().any().any()
        report_path=(task/'outputs/clustering-section4' if kind=='clustering' else task/'.agent_work/section4')/'run_report.json'
        report=json.loads(report_path.read_text());assert report['state']=='completed';assert report['publication']['scope']=='full_train_refit'
        if kind!='clustering':assert report['control_status']==report['refit_status']=='completed'
        else:
            assert report['validation']['status']=='completed'
            process=report_path.parent/'attempt/diagnostic_process_result.json';stamp=process.stat().st_mtime_ns
            content=(task/'submission.csv').read_bytes();(task/'submission.csv').unlink()
            again=subprocess.run(command,cwd=task,capture_output=True,text=True,timeout=30)
            assert again.returncode==0 and (task/'submission.csv').read_bytes()==content and process.stat().st_mtime_ns==stamp
        if semantic:assert len(semantic_calls)==1
        results.append({'kind':kind,'metric':metric,'passed':True,'report':str(report_path.relative_to(output)),'report_sha256':sha(report_path),'csv_sha256':sha(task/'submission.csv'),'worker_recovery_without_fit':True if kind=='clustering' else None,'semantic_v2_without_task_spec':semantic and len(semantic_calls)==1})
        # All processes of this task have exited. Delete only byte-verified corpus
        # copies and flat code hardlinks reproducible from this immutable ZIP.
        for cache in task.rglob('.ml_knowledge_runtime'):
            corpus=cache/manifest['corpus'];assert {str(f.relative_to(corpus)) for f in corpus.rglob('*') if f.is_file()}==set(manifest['files'])
            for name,record in manifest['files'].items():assert sha(corpus/name)==record['sha256']
            shutil.rmtree(cache)
        for path in source.iterdir():
            staged=task/path.name;assert sha(staged)==sha(path);staged.unlink()
    result={'passed':True,'archive_sha256':sha(archive),'cases':results,'organizer_requests':0,'network':'none','scope':'Synthetic exact-ZIP delivery/control/refit and explicit target-free clustering; no live/hidden quality claim'}
    (output/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.archive.resolve(),args.output.resolve())
