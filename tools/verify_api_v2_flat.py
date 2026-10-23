"""Clean flat-ZIP startup checks with a local fake endpoint, never organizer API."""
import argparse
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile


def fixture(root):
    for name, start, end in [('train.csv', 0, 100), ('test.csv', 100, 112)]:
        with (root / name).open('w') as stream:
            writer = csv.writer(stream);writer.writerow(['id', 'x', 'cat'] + (['y'] if start == 0 else []))
            for i in range(start, end):
                x = (i % 29 - 14) / 7
                writer.writerow([str(i), '' if i % 11 == 0 else x, 'A' if x > 0 else 'B'] + ([2*x] if start == 0 else []))
    with (root / 'sample_submission.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['id','y']);writer.writerows((str(i),0) for i in range(100,112))
    (root / 'task.md').write_text('Synthetic offline regression fixture; minimize RMSE.\n')
    (root / 'task_spec.json').write_text(json.dumps({'kind':'regression','targets':['y'],'id_columns':['id'],
        'feature_columns':['x','cat'],'categorical_columns':['cat'],'metric':{'name':'rmse'},
        'output':{'columns':['id','y'],'prediction_type':'values'}}))


def run(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    report = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'organizer_requests': 0, 'cases': []}
    for label, model, transport in [('unset', None, 'native'), ('empty', '', 'native'), ('explicit', 'fixture/custom-model', 'json')]:
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(body)
                first = len(requests) == 1
                name = 'get_dataset_report' if first else 'finish'
                args = {'report': 'missing'} if first else {'reason':'Completed clean archive delivery test', 'imputation_skip_reason':'Bounded offline startup verification'}
                operation = 'read_data' if first else 'finish_verified'
                if body.get('tools'):
                    message = {'tool_calls':[{'id':'fixture_'+operation,'type':'function','function':{'name':name,'arguments':json.dumps({**args,'operation_id':operation})}}]}
                else:
                    message = {'content':json.dumps({'operation_id':operation,'tool':name,'arguments':args})}
                response = json.dumps({'choices':[{'finish_reason':'stop','message':message}], 'usage':{'prompt_tokens':10,'completion_tokens':10}}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(response)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix='flat-api2-') as temporary:
                root=Path(temporary)
                with zipfile.ZipFile(archive) as package:
                    assert all('/' not in name and (name.endswith('.py') or name=='requirements.txt') for name in package.namelist())
                    package.extractall(root)
                    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in package.namelist()}
                fixture(root)
                env={k:v for k,v in os.environ.items() if not k.startswith(('OPENAI_', 'AGENT_'))}
                env.update(OPENAI_API_KEY='offline_fake_key', OPENAI_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1')
                if model is not None:env['OPENAI_MODEL']=model
                proc=subprocess.run([sys.executable,'solution.py','task.md','--time-limit','120','--cpu','1','--memory-mb','2048',
                    '--max-calls','3','--run-id','flat_api2','--transport',transport],cwd=root,env=env,capture_output=True,text=True,timeout=135)
                work=root/'.agent_work/flat_api2'
                saved=json.loads((work/'run_report.json').read_text())
                expected=model or 'qwen/qwen3.8-27b'
                body=json.loads(requests[1]['messages'][1]['content']) if len(requests)>1 else {}
                delivered=any(item['operation_id']=='read_data' and '"tool_api_version":2' in item['content'] for item in body.get('observations',[]))
                row={'case':label,'transport':transport,'exit_code':proc.returncode,'model_used':[item['model'] for item in requests],
                    'calls':len(requests),'state':saved['state'],'control':saved['control_status'],'refit':saved['refit_status'],
                    'actual_result_delivered':delivered,'unchanged_flat_files':all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())}
                row['passed']=proc.returncode==0 and saved['state']=='completed' and saved['control_status']=='completed' and saved['refit_status']=='completed' and delivered and all(item['model']==expected for item in requests)
                directory=output/label;directory.mkdir()
                (directory/'REPORT.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2))
                (directory/'requests.json').write_text(json.dumps(requests,ensure_ascii=False))
                (directory/'container.log').write_text(proc.stdout+proc.stderr)
                shutil.copyfile(root/'submission.csv',directory/'submission.csv')
                report['cases'].append(row)
                print(json.dumps(row),flush=True)
        finally:
            server.shutdown();server.server_close();thread.join()
    report['all_passed']=all(row['passed'] for row in report['cases'])
    (output/'REPORT.json').write_text(json.dumps(report,indent=2))
    return report['all_passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(0 if run(args.archive,args.output) else 1)
