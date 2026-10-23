"""Exact ZIP, public data, recorded Qwen decisions; no new paid generation."""
import argparse
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import threading
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.diagnostic_3h import Store, validate_csv, write
from automl_agent.qwen_stream import completion_from_sse


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    task = args.output / 'task'; task.mkdir()
    with zipfile.ZipFile(args.archive) as package:
        assert all('/' not in n and (n.endswith('.py') or n == 'requirements.txt') for n in package.namelist())
        package.extractall(task)
    source = ROOT / 'data_for_participants/Регрессия'
    hashes = {}
    for name in ('task.md', 'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv'):
        shutil.copyfile(source / name, task / name)
        hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
    contract_response = json.loads(args.contract_response.read_text())
    proposal = json.loads(contract_response['choices'][0]['message']['content'])
    store = Store(args.source)
    item = next(a for a in store.index()['artifacts'].values()
        if a.get('identity') == args.identity and a['name'] == 'qwen_response.txt')
    raw = gzip.decompress((store.root / item['path']).read_bytes())
    action_response = json.loads(raw) if raw.lstrip().startswith(b'{') else completion_from_sse(raw.decode().splitlines())
    requests, errors = [], []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                context = json.loads(body['messages'][-1]['content'])
                requests.append({'native': bool(body.get('tools')), 'effort': body['reasoning']['effort']})
                if len(requests) == 1:
                    assert context['request_id'] == proposal['request_id'] and context['source_hash'] == proposal['source_hash']
                    response = contract_response
                elif len(requests) == 2:
                    assert body.get('tools')
                    response = action_response
                elif len(requests) == 3:
                    cards = [json.loads(o['content']) for o in context['observations'] if o['kind'] == 'training_card']
                    assert len(cards) == 1 and cards[0]['status'] == 'completed'
                    command = {'operation_id': 'replay_finish', 'tool': 'finish', 'arguments': {
                        'reason': 'Recorded native training replay completed',
                        'result_ids': ['baseline', cards[0]['experiment_id']],
                        'imputation_skip_reason': 'This transport replay does not add a new imputation experiment'}}
                    assert body.get('tools')
                    response = {'choices': [{'finish_reason': 'tool_calls', 'message': {'tool_calls': [
                        {'id': 'fixture_finish', 'type': 'function', 'function': {'name': 'finish',
                            'arguments': json.dumps({'operation_id': command['operation_id'], **command['arguments']})}}]}}]}
                else:
                    raise AssertionError('Unexpected extra generation')
                response = {**response, 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
                status = 200
            except Exception as error:
                errors.append(type(error).__name__ + ': ' + str(error))
                status, response = 402, {'error': 'Offline fixture protocol failed; no external provider used'}
            encoded = json.dumps(response).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded))); self.end_headers(); self.wfile.write(encoded)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('OPENAI_', 'AGENT_', 'PYTHONPATH'))}
    environment.update(OPENAI_API_KEY='offline-recorded-replay', OPENAI_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1')
    try:
        result = subprocess.run([sys.executable, 'solution.py', 'task.md'], cwd=task, env=environment,
            capture_output=True, text=True, timeout=360)
        (args.output / 'run.log').write_text(result.stdout + result.stderr)
    finally:
        server.shutdown(); server.server_close(); thread.join()
    work = next((task / '.agent_work').iterdir())
    saved = json.loads((work / 'run_report.json').read_text())
    with sqlite3.connect(work / 'journal.sqlite') as db:
        meta = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
        operation = json.loads(db.execute("SELECT record FROM operations WHERE id='step_2'").fetchone()[0])
        trained = json.loads(db.execute('SELECT result FROM experiments WHERE id=?', (operation['experiment_id'],)).fetchone()[0])
        receipts = db.execute('SELECT COUNT(*) FROM cc_receipts').fetchone()[0]
    csv = validate_csv({'kind': 'regression', 'directory': str(task)})
    publication = saved.get('publication') or meta.get('publication') or {}
    report = {'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(), 'input_hashes': hashes,
        'decision_source_identity': args.identity, 'recorded_action_sha256': hashlib.sha256(raw).hexdigest(),
        'scope': 'Public data fit using recorded terminal Qwen contract/action plus synthetic finish; not a fresh autonomous live run or model-quality comparison',
        'organizer_requests': 0, 'request_count': len(requests), 'requests': requests, 'fixture_errors': errors,
        'exit_code': result.returncode, 'state': saved['state'], 'control': saved['control_status'], 'refit': saved['refit_status'],
        'native_operation_status': operation['status'], 'trained_status': trained['status'], 'receipts': receipts,
        'csv': csv, 'publication': publication, 'limits': meta['limits']}
    report['passed'] = (not errors and result.returncode == 0 and len(requests) == 3 and receipts == 2
        and all(report[k] == 'completed' for k in ('state', 'control', 'refit', 'trained_status'))
        and operation['status'] == 'succeeded' and publication.get('scope') == 'full_train_refit'
        and csv.get('passed') is True and meta['limits']['seconds'] == 10800
        and all(meta['limits'][k] is None for k in ('max_calls', 'max_candidates', 'max_debug', 'max_llm_seconds')))
    write(args.output / 'REPORT.json', report)
    print(json.dumps(report))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--identity', required=True)
    p.add_argument('--contract-response', type=Path, required=True)
    a = p.parse_args()
    for name in ('archive', 'output', 'source', 'contract_response'):
        setattr(a, name, getattr(a, name).resolve())
    raise SystemExit(run(a))
