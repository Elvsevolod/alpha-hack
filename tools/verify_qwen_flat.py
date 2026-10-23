"""Offline, exact-entrypoint ZIP checks with a synthetic local HTTP service."""
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.verify_api_v2_flat import fixture
from tools.mock_task_resolution import fixture_resolution


def run(archive, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'command': ['python', 'solution.py', 'task.md'], 'organizer_requests': 0, 'cases': []}
    cases = [('unset', None, False), ('empty', '', False), ('whitespace', '   ', False),
        ('json_fallback', None, True)]
    for label, model, fallback in cases:
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(body)
                # One valid observation exercises xhigh -> medium. An explicit
                # synthetic account failure then tests useful fallback + refit
                # without waiting out the real three-hour default deadline.
                semantic = fixture_resolution(body)
                if semantic is not None:
                    status, response = 200, semantic
                elif fallback and body.get('tools'):
                    status, response = 400, {'error': 'Synthetic native-tools incompatibility'}
                elif sum(not (fallback and item.get('tools')) for item in requests if fixture_resolution(item) is None) > 1:
                    status, response = 402, {'error': 'Synthetic account unavailable; end this offline fixture'}
                else:
                    args = {'report': 'missing'}
                    if body.get('tools'):
                        message = {'tool_calls': [{'id': 'fixture_read', 'type': 'function', 'function': {
                            'name': 'get_dataset_report', 'arguments': json.dumps({**args, 'operation_id': 'read_fixture'})}}]}
                    else:
                        message = {'content': json.dumps({'tool': 'get_dataset_report', 'operation_id': 'read_fixture', 'arguments': args})}
                    status, response = 200, {'choices': [{'finish_reason': 'stop', 'message': message}],
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'completion_tokens_details': {'reasoning_tokens': 10}}}
                encoded = json.dumps(response).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix='qwen-flat-') as temporary:
                root = Path(temporary)
                with zipfile.ZipFile(archive) as package:
                    assert all('/' not in name and (name.endswith('.py') or name == 'requirements.txt') for name in package.namelist())
                    package.extractall(root)
                fixture(root)
                environment = {k: v for k, v in os.environ.items() if not k.startswith(('OPENAI_', 'AGENT_', 'PYTHONPATH'))}
                environment.update(OPENAI_API_KEY='offline-fixture', OPENAI_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1')
                if model is not None:
                    environment['OPENAI_MODEL'] = model
                result = subprocess.run([sys.executable, 'solution.py', 'task.md'], cwd=root, env=environment,
                    capture_output=True, text=True, timeout=180)
                work = next((root / '.agent_work').iterdir())
                saved = json.loads((work / 'run_report.json').read_text())
                with sqlite3.connect(work / 'journal.sqlite') as database:
                    limits = json.loads(database.execute("SELECT value FROM metadata WHERE key='limits'").fetchone()[0])
                with (root / 'submission.csv').open() as stream:
                    reader = csv.DictReader(stream)
                    rows, columns = list(reader), reader.fieldnames
                csv_ok = columns == ['id', 'y'] and [r['id'] for r in rows] == [str(i) for i in range(100, 112)] and all(math.isfinite(float(r['y'])) for r in rows)
                row = {'case': label, 'exit_code': result.returncode, 'calls': len(requests), 'limits': limits,
                    'state': saved['state'], 'control': saved['control_status'], 'refit': saved['refit_status'],
                    'csv_valid': csv_ok, 'efforts': [r['reasoning']['effort'] for r in requests],
                    'transports': ['native' if r.get('tools') else 'json' for r in requests]}
                row['passed'] = result.returncode == 0 and all(row[k] == 'completed' for k in ('state', 'control', 'refit')) and csv_ok
                row['passed'] &= limits['seconds'] == 10800 and all(limits[k] is None for k in ('max_calls', 'max_candidates', 'max_llm_seconds', 'max_debug'))
                row['passed'] &= row['efforts'] == (['xhigh', 'xhigh', 'xhigh', 'medium'] if fallback else ['xhigh', 'xhigh', 'medium'])
                row['passed'] &= all(r['model'] == 'qwen/qwen3.8-27b' and r['temperature'] == 1 and r['reasoning']['enabled'] for r in requests)
                directory = output / label
                directory.mkdir()
                (directory / 'REPORT.json').write_text(json.dumps(saved, indent=2))
                (directory / 'requests.json').write_text(json.dumps(requests))
                (directory / 'run.log').write_text(result.stdout + result.stderr)
                shutil.copyfile(root / 'submission.csv', directory / 'submission.csv')
                report['cases'].append(row)
                print(json.dumps(row), flush=True)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        (output / 'REPORT.json').write_text(json.dumps(report, indent=2))
    report['passed'] = all(r['passed'] for r in report['cases'])
    (output / 'REPORT.json').write_text(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
