"""One bounded native organizer-task probe; development only, never a submission.

Copies the supplied task, freezes source/input hashes, launches the unchanged
solution entry point, observes only its process tree/journal, and checks API
limits again after the child is terminal. It never prints connection values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
LIMITS_SCRIPT = ROOT / 'skills/alpha-api-limits/scripts/check_limits.py'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sources():
    paths = [ROOT / 'solution.py']
    for name in ['automl_agent', 'context_compressor', 'ml_knowledge']:
        paths += sorted((ROOT / name).glob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def limits(phase, args, before=None):
    destination = args.output / ('api_limits_before_start.json' if phase == 'before' else 'api_limits_after.json')
    command = [sys.executable, str(LIMITS_SCRIPT), phase, '--env-file', str(args.env_file), '--output', str(destination)]
    if before:
        command += ['--before', str(before)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, timeout=45)
    if destination.exists():
        return json.loads(destination.read_text())
    return dict(status='unknown', check_exit_code=result.returncode)


def snapshot(task):
    paths = list((task / '.agent_work').glob('*/journal.sqlite'))
    if not paths:
        return {}
    db = sqlite3.connect(paths[0].as_uri() + '?mode=ro', uri=True, timeout=1)
    try:
        data = {k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
        experiments = []
        for identifier, spec, result in db.execute('SELECT id,spec,result FROM experiments'):
            s, r = json.loads(spec), json.loads(result) if result else {}
            experiments.append(dict(id=identifier, model=s['pipeline']['model'], status=r.get('status', 'running'),
                score=r.get('score'), scope=r.get('scope')))
        return dict(state=data.get('state'), llm_calls=data.get('llm_calls', 0),
            llm_seconds=data.get('llm_seconds', 0), candidate_count=data.get('candidate_count', 0),
            control_status=data.get('control_status'), refit_status=data.get('refit_status'),
            publication_scope=(data.get('publication') or {}).get('scope'), experiments=experiments,
            failure_present=bool(data.get('failure')))
    finally:
        db.close()


def rss_tree(pid):
    result = subprocess.run(['ps', '-axo', 'pid=,ppid=,rss='], capture_output=True, text=True, timeout=5)
    if result.returncode:
        return None
    rows = [tuple(map(int, line.split())) for line in result.stdout.splitlines() if len(line.split()) == 3]
    selected, pending = set(), {pid}
    while pending:
        selected.update(pending)
        pending = {child for child, parent, _ in rows if parent in pending and child not in selected}
    return sum(rss for child, _, rss in rows if child in selected) / 1024


def run(args):
    from dotenv import dotenv_values
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    task = args.output / 'task'
    task.mkdir(exist_ok=False)
    values = dotenv_values(args.env_file)
    connection = {k:os.environ.get(k) or values.get(k) for k in ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL']}
    assert connection['OPENAI_API_KEY'] and connection['OPENAI_BASE_URL'], 'Missing organizer connection'
    assert urlparse(connection['OPENAI_BASE_URL']).hostname == 'changellenge.up.railway.app', 'Unexpected organizer endpoint'
    input_hashes = {}
    for name in ['task.md', 'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv']:
        source = args.source / name
        if source.exists():
            shutil.copyfile(source, task / name)
            input_hashes[name] = sha(source)
    frozen_sources = sources()
    corpus = ROOT / 'ml_knowledge/corpus'
    lock = json.loads((corpus / 'knowledge.lock.json').read_text())
    corpus_sha = sha(corpus / 'knowledge.sqlite')
    assert corpus_sha == lock['files']['knowledge.sqlite'], 'Local corpus is not materialized or does not match its lock'
    command = [sys.executable, '-u', str(ROOT / 'solution.py'), str(task / 'task.md'),
        '--time-limit', str(args.seconds), '--cpu', '1', '--memory-mb', '3072',
        '--max-calls', str(args.max_calls), '--max-candidates', str(args.max_candidates),
        '--run-id', 'bounded_autonomy', '--seed', '42']
    freeze = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=frozen_sources, driver_sha256=sha(Path(__file__)), input_sha256=input_hashes,
        corpus_sha256=corpus_sha, corpus_version=lock['corpus'], requirements_sha256=sha(ROOT / 'requirements.txt'),
        seconds=args.seconds, cpu=1, memory_mb=3072, max_model_calls=args.max_calls,
        max_candidates=args.max_candidates, seed=42,
        task_spec_supplied=False, transport='auto', model_or_hyperparameters_prescribed=False,
        generation_overrides_supplied=False, independent_quality_claimed=False,
        limitation='Previously viewed organizer data; native partial development environment; controller internal control can select finalists, not an independent model-quality confirmation.')
    save(args.output / 'freeze.json', freeze)
    before, after, process = None, None, None
    started = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = dict(status='preparing', real_model_calls_started=False)
    try:
        before = limits('before', args)
        assert before.get('remaining_budget', -1) >= 30 and not before.get('budget_exhausted') and before.get('limits', {}).get('blocked') is not True, 'Unknown, exhausted or insufficient API remainder'
        assert hashlib.sha256(connection['OPENAI_API_KEY'].encode()).hexdigest()[:16] == before['key_fingerprint'], 'Connection/key fingerprint mismatch'
        environment = {k:v for k,v in os.environ.items() if not k.startswith(('OPENAI_', 'AGENT_', 'PYTHONPATH'))}
        environment.update({k:v for k,v in connection.items() if v})
        environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        print(json.dumps(dict(status='starting', remaining=before['remaining_budget'], maximum=before['limits']['max_budget'])), flush=True)
        with (args.output / 'runtime.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=task, env=environment, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            report.update(status='running', child_pid=process.pid, real_model_calls_started=True)
            peak, last_printed = 0., 0.
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > args.seconds + 60:
                    report['watchdog_stop'] = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                    break
                memory = rss_tree(process.pid)
                peak = max(peak, memory or 0)
                try:
                    current = snapshot(task)
                except (OSError, sqlite3.Error, ValueError):
                    current = {'observation':'temporarily_unavailable'}
                report.update(wall_seconds=elapsed, sampled_peak_process_tree_rss_mib=peak, current=current)
                save(args.output / 'progress.json', report)
                if elapsed - last_printed >= 30:
                    print(json.dumps(dict(elapsed=round(elapsed, 1), **current)), flush=True)
                    last_printed = elapsed
                time.sleep(2)
            report.update(status='terminal', returncode=process.returncode, wall_seconds=time.monotonic()-started,
                current=snapshot(task), sampled_peak_process_tree_rss_mib=peak)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        # The only after check happens after confirmed terminal process state.
        after = limits('after', args, args.output / 'api_limits_before_start.json' if before and 'key_fingerprint' in before else None)
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        report.update(api_limits_after=after, sources_unchanged=frozen_sources == sources(),
            child_cpu_seconds=usage_after.ru_utime+usage_after.ru_stime-usage_before.ru_utime-usage_before.ru_stime,
            child_terminal=process is None or process.poll() is not None)
        save(args.output / 'result.json', report)
        print(json.dumps(dict(status=report['status'], remainder_after=after.get('remaining_budget'),
            account_counter_increase=after.get('comparison', {}).get('increase'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--max-calls', type=int, default=6)
    parser.add_argument('--max-candidates', type=int, default=2)
    args = parser.parse_args()
    if min(args.seconds, args.max_calls, args.max_candidates) <= 0:
        parser.error('Time, call and candidate caps must be positive')
    run(args)
