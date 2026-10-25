"""Sequential frozen A/B development runs; labels opened only by explicit evaluate.

The human removed the monetary series cap. Account snapshots, original deadlines,
worker limits and unknown-outcome stops remain. No other model is used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_agent.util import atomic_json, file_hash

TASKS = ('adult', 'bank', 'bike', 'har')
FILES = ('task.md', 'train.csv', 'test.csv', 'data_dictionary.csv', 'sample_submission.csv')


def prepare(output, candidate):
    output.mkdir(parents=True, exist_ok=False)
    original = ROOT / 'outputs/eda_autonomy_20261024'
    original_manifest = json.loads((original / 'manifest.json').read_text())
    commits = {'A': '6b0c0cd', 'B': candidate}
    resolved = {arm: subprocess.check_output(['git', 'rev-parse', ref], cwd=ROOT, text=True).strip()
        for arm, ref in commits.items()}
    for arm, sha in resolved.items():
        runtime = output / 'runtimes' / arm
        runtime.mkdir(parents=True)
        archive = output / (arm + '.tar')
        with archive.open('wb') as target:
            subprocess.run(['git', 'archive', sha, 'solution.py', 'requirements.txt', 'automl_agent',
                'context_compressor', 'ml_knowledge'], cwd=ROOT, stdout=target, check=True)
        with tarfile.open(archive) as bundle:
            # git archive contains trusted tracked paths; exclude nested links.
            members = [m for m in bundle.getmembers() if m.isfile() or m.isdir()]
            bundle.extractall(runtime, members=members, filter='data')
        archive.unlink()
        corpus = runtime / 'ml_knowledge/corpus'
        if corpus.exists():
            shutil.rmtree(corpus)
        corpus.symlink_to(ROOT / 'ml_knowledge/corpus', target_is_directory=True)
    runs = []
    for index, task in enumerate(TASKS):
        expected = original_manifest[task]['prepared_sha256']
        hashes = {}
        for arm in (('A', 'B') if index % 2 == 0 else ('B', 'A')):
            case = output / 'runs' / (task + '_' + arm) / 'task'
            case.mkdir(parents=True)
            for name in FILES:
                source = original / 'tasks' / task / name
                if file_hash(source) != expected[name]:
                    raise ValueError('Prepared input changed: ' + task + '/' + name)
                shutil.copyfile(source, case / name)
                hashes[name] = file_hash(source)
            runs.append({'task': task, 'arm': arm, 'id': task + '_' + arm,
                'state': 'registered', 'seconds': 10800, 'cpu': 7, 'memory_mb': 28000,
                'max_calls': None, 'max_candidates': None, 'seed': 42,
                'input_hashes': hashes.copy(), 'outer_labels_opened': False})
    manifest = {'version': 1, 'commits': resolved, 'runs': runs, 'real_model': 'qwen/qwen3.8-27b',
        'monetary_series_cap': None, 'monetary_override': 'Human explicitly removed test budget cap',
        'datasets_known_to_development': True, 'labels_source_outside_tasks': str(original / 'sealed')}
    atomic_json(output / 'protocol.json', manifest)
    print(json.dumps({'registered': len(runs), 'commits': resolved}))


def snapshot(output, phase, before=None):
    command = [sys.executable, str(ROOT / 'skills/alpha-api-limits/scripts/check_limits.py'), phase,
        '--output', str(output)]
    if before:
        command.extend(['--before', str(before)])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=40)
    if result.returncode:
        return None
    return json.loads(output.read_text())


def live(output):
    from dotenv import dotenv_values
    config = {**dotenv_values(ROOT / '.env'), **os.environ}
    if not config.get('OPENAI_API_KEY'):
        raise ValueError('Local organizer credential missing')
    protocol = json.loads((output / 'protocol.json').read_text())
    if protocol.get('stopped_reason') or protocol.get('evaluation_frozen_at'):
        raise ValueError('Series stopped or evaluated; inspect evidence before registering a new series')
    expected_key = hashlib.sha256(config['OPENAI_API_KEY'].encode()).hexdigest()[:16]
    period = protocol.get('account_period')
    for row in protocol['runs']:
        if row['state'] != 'registered':
            continue
        directory = output / 'runs' / row['id']
        marker = directory / 'started.json'
        if marker.exists():
            raise ValueError('Prior attempt exists; no blind repeat')
        before_path, after_path = directory / 'api_limits_before.json', directory / 'api_limits_after.json'
        before = snapshot(before_path, 'before')
        if (before is None or before.get('remaining_budget') is None
                or before.get('key_fingerprint') != expected_key
                or before.get('budget_exhausted') or before['limits'].get('blocked')
                or (period is not None and period != before['limits'].get('budget_reset_at'))):
            row.update(state='blocked_account_snapshot')
            protocol['stopped_reason'] = 'account_snapshot_not_usable'
            atomic_json(output / 'protocol.json', protocol)
            break
        period = before['limits'].get('budget_reset_at')
        protocol['account_period'] = period
        print(json.dumps({'run': row['id'], 'before': before['remaining_budget']}, ensure_ascii=False), flush=True)
        atomic_json(marker, {'started_at': time.time(), 'commit': protocol['commits'][row['arm']]})
        runtime = output / 'runtimes' / row['arm']
        env = {**os.environ, **{k: v for k, v in config.items() if k.startswith('OPENAI_') and v is not None},
            'OPENAI_MODEL': protocol['real_model'], 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
        cmd = [sys.executable, str(runtime / 'solution.py'), str(directory / 'task/task.md'),
            '--time-limit', str(row['seconds']), '--cpu', str(row['cpu']),
            '--memory-mb', str(row['memory_mb']), '--seed', '42', '--run-id', 'research_memory_ab']
        started = time.monotonic()
        process = None
        try:
            with (directory / 'agent.log').open('w') as log:
                process = subprocess.Popen(cmd, cwd=runtime, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    process.wait(timeout=row['seconds'] + 30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            row.update(state='exited', returncode=process.returncode, wall_seconds=time.monotonic()-started)
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            after = snapshot(after_path, 'after', before_path)
            row['api_after_confirmed'] = after is not None
            row['gateway_increase'] = after.get('comparison', {}).get('increase') if after else None
            row['remaining_after'] = after.get('remaining_budget') if after else None
            journal = directory / 'task/.agent_work/research_memory_ab/journal.sqlite'
            unknown = None
            if journal.exists():
                with sqlite3.connect(journal.as_uri() + '?mode=ro', uri=True) as db:
                    found = db.execute("SELECT value FROM metadata WHERE key='qwen_unknown_outcome'").fetchone()
                    unknown = json.loads(found[0]) if found else None
            row['unknown_model_outcome'] = bool(unknown) or process is None or process.returncode < 0
            atomic_json(output / 'protocol.json', protocol)
            print(json.dumps({'run': row['id'], 'state': row['state'], 'increase': row['gateway_increase'],
                'remaining': row['remaining_after'], 'unknown': bool(unknown)}, ensure_ascii=False), flush=True)
        if after is None or row['gateway_increase'] is None or row['unknown_model_outcome'] or after.get('budget_exhausted') or after['limits'].get('blocked'):
            protocol['stopped_reason'] = 'unknown_outcome_or_account_unavailable'
            atomic_json(output / 'protocol.json', protocol)
            break


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'live'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidate', help='Frozen candidate commit for prepare')
    a = p.parse_args()
    if a.action == 'prepare':
        if not a.candidate:
            p.error('--candidate is required')
        prepare(a.output.resolve(), a.candidate)
    else:
        live(a.output.resolve())
