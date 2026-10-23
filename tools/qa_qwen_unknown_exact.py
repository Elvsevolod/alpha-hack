"""Exact-archive direct-HTTP loss test. No organizer/model endpoint is contacted."""
import argparse
import csv
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import socket
import sys
import threading
import time
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import diagnostic_3h as d
from tools.mock_task_resolution import fixture_resolution
from tools.verify_api_v2_flat import fixture


def run(output,archive):
    output=output.resolve();output.mkdir(parents=True,exist_ok=False);task=output/'task';task.mkdir()
    with zipfile.ZipFile(archive) as z:z.extractall(task)
    fixture(task);(task/'task_spec.json').unlink();requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));requests.append(body)
            reply=fixture_resolution(body)
            if reply is not None:
                content=json.dumps(reply).encode();self.send_response(200);self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
            else:
                # The generation request reached the server. Its response is lost.
                self.close_connection=True;self.connection.shutdown(socket.SHUT_RDWR);self.connection.close()
    server=ThreadingHTTPServer(('0.0.0.0',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    started=time.monotonic()
    try:
        result=d.docker('run','--rm','--cpus','1','--memory','2g','-v',str(task)+':/task','-w','/task',
            '-e','OPENAI_API_KEY=offline-fixture','-e',f'OPENAI_BASE_URL=http://host.docker.internal:{server.server_port}/v1',
            d.IMAGE,'python','solution.py','task.md','--cpu','1','--memory-mb','2048','--time-limit','120','--run-id','unknown',check=False,timeout=150)
        (output/'run.log').write_text(result.stdout+result.stderr)
        report=d.read(task/'.agent_work/unknown/run_report.json',{})
        unknown=report.get('qwen_unknown_outcome') or {}
        with (task/'submission.csv').open() as stream:reader=csv.DictReader(stream);columns=reader.fieldnames;rows=list(reader)
        passed=(result.returncode==0 and len(requests)==2 and unknown.get('call_id')==2 and unknown.get('error_type')=='RemoteProtocolError'
            and report.get('control_status')==report.get('refit_status')=='completed'
            and report.get('publication',{}).get('scope')=='full_train_refit' and len(rows)==12 and columns==['id','y'])
        verdict={'passed':passed,'archive_sha256':d.digest(archive),'organizer_requests':0,'direct_http_requests':len(requests),
            'semantic_requests':sum(fixture_resolution(body) is not None for body in requests),'sent_generation_responses_lost':1,
            'native_json_or_request_retries_after_unknown':len(requests)-2,'unknown_outcome':unknown,'run_report':report,
            'elapsed_seconds':time.monotonic()-started,'scope':'One direct HTTP response lost; local synthetic provider and fixture only'}
        d.write(output/'REPORT.json',verdict);print(json.dumps({k:verdict[k] for k in ('passed','direct_http_requests','unknown_outcome','elapsed_seconds')}))
        assert passed,report
    finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args();run(a.output,a.archive.resolve())
