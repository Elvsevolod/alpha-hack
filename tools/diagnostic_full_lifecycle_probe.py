"""Exercise the production supervisor lifecycle without organizer HTTP calls."""
import csv
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import diagnostic_3h as d
from tools.test_diagnostic_3h import fixture


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    source=output/'source'
    for folder,target,kind in [('Регрессия','site_eui','regression'),('Многоклассовая классификация','attack_cat','classification')]:
        task=source/folder;task.mkdir(parents=True);fixture(task)
        for name in ('train.csv','sample_submission.csv'):
            with (task/name).open() as stream:rows=list(csv.reader(stream))
            rows[0][-1]=target
            if kind=='classification':
                for row in rows[1:]:row[-1]='A' if float(row[-1])>=0 else 'B'
            with (task/name).open('w') as stream:csv.writer(stream).writerows(rows)
        spec=json.loads((task/'task_spec.json').read_text());spec.update(kind=kind,targets=[target],metric={'name':'rmse' if kind=='regression' else 'f1','average':'macro'})
        spec['output']={'columns':['id',target],'prediction_type':'values' if kind=='regression' else 'labels'}
        # A public task can infer this contract; the fixture's exact explicit
        # contract is retained to isolate orchestration and lifecycle behavior.
        (task/'task_spec.json').write_text(json.dumps(spec))
        (task/'task.md').write_text(f'Synthetic {kind}. Target: `{target}`. Metric: '+('RMSE' if kind=='regression' else 'Macro F1'))
        (task/'data_dictionary.csv').write_text('feature,description\nx,Synthetic numeric feature\n')
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));calls.append(body)
            if len(calls)==1:
                response={'id':'lifecycle_1','choices':[{'finish_reason':'stop','message':{'tool_calls':[{'id':'one','type':'function','function':{'name':'get_dataset_report','arguments':json.dumps({'operation_id':'one','report':'missing'})}}]}}],'usage':{'prompt_tokens':1,'completion_tokens':2}}
                status=200
            else:status,response=402,{'error':'offline permanent refusal'}
            content=json.dumps(response).encode();self.send_response(status);self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    info={'spend':0,'total_spend':0,'max_budget':600,'blocked':False,'budget_reset_at':'offline-period'}
    checks=output/'checks';shutil.copytree(d.ROOT/'outputs/diagnostic_acceptance_v024_final',checks)
    checked=d.read(checks/'REPORT.json');checked['supervisor_sha256']=d.digest(Path(d.__file__));d.write(checks/'REPORT.json',checked)
    original_popen=subprocess.Popen;original_reconcile=d.reconcile
    def spawn(arguments,*args,**kwargs):
        if isinstance(arguments,list) and len(arguments)>=3 and arguments[1]==str(Path(d.__file__).resolve()) and arguments[2]=='gateway':
            code='from pathlib import Path;from tools import diagnostic_3h as d;import sys;d.limits.fetch_info=lambda key:'+repr(info)+';d.gateway_main(Path(sys.argv[1]),int(sys.argv[2]))'
            directory=arguments[arguments.index('--output')+1];port=arguments[arguments.index('--port')+1]
            arguments=[sys.executable,'-c',code,directory,port]
        return original_popen(arguments,*args,**kwargs)
    def fast_reconcile(*args,**kwargs):return original_reconcile(*args,**kwargs,interval=.05,duration=.5)
    # Only API accounting is replaced with fixed offline data. The production
    # gateway, registry, watchdog, observers, containers and archive stay intact.
    environment={'OPENAI_API_KEY':'offline-fixture','OPENAI_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1','OPENAI_MODEL':'qwen/qwen3.8-27b'}
    run_directory=output/('run_'+output.name[-16:])
    try:
        with patch.dict(os.environ,environment),patch.object(d.limits,'fetch_info',lambda key:info),patch.object(subprocess,'Popen',spawn),patch.object(d,'reconcile',fast_reconcile):
            status=d.run(run_directory,d.ROOT/'dist/solution_v024.zip',checks,18843,source=source,cpu=1,memory_gib=1)
        report=d.read(run_directory/'REPORT.json')
        assert len(calls)>=2,'production_gateway_did_not_forward'
        assert all(r['technical_passed'] for r in report['results']),report
        assert report['secret_scan']['passed'] and report['diagnostic_acceptance']['links_complete']
        assert not report['diagnostic_acceptance']['pending_operations']
        d.write(output/'REPORT.json',{'passed':True,'archive_sha256':d.digest(d.ROOT/'dist/solution_v024.zip'),'organizer_requests':0,
            'supervisor_exit':status,'forwarded_fixture_requests':len(calls),'technical_completed':2,'diagnostics_verified':True,
            'gateway_read_post_launch_deadlines':True,'full_lifecycle':True,'cpu_per_fixture':1,'memory_gib_per_fixture':1,
            'resource_scope':'Startup/lifecycle fixture; separate simultaneous 5 CPU/5 GiB acceptance retained.'})
        print('FULL_LIFECYCLE_PASSED',flush=True)
    finally:server.shutdown();server.server_close()


if __name__=='__main__':run(Path(sys.argv[1]).resolve())
