"""Bounded real-agent quality probe; external tooling, never in the submission."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import diagnostic_3h as d
from tools.shared_quality_budget import SharedQualityBudget
from automl_agent.qwen_stream import ProviderStreamError, completion_from_sse, validate_completion


def decode_response(response):
    if 'text/event-stream' in response.headers.get('content-type', '').lower():
        try:
            return completion_from_sse(response.text.splitlines())
        except ProviderStreamError as error:
            # Preserve the original explicit failure downstream; no gateway retry.
            return {'error': {'code': error.code}, 'usage': {}}
    return validate_completion(response.json())


class PeriodGuard(d.Guard):
    """External experiments cannot silently enter a new API budget period."""
    def __init__(self, before, fetch, directory, *, wall_time=time.time, **kwargs):
        reset = datetime.fromisoformat(before['limits']['budget_reset_at'])
        if reset.tzinfo is None:
            raise ValueError('API reset must include a timezone')
        super().__init__(before, fetch, directory, **kwargs)
        # Leave a ten-minute gap for requests already in flight and accounting.
        # This is NOT a claim that delayed provider billing has a proven bound.
        self.period_end = self.clock() + reset.timestamp() - wall_time() - 600
        if self.period_end <= self.clock():
            raise ValueError('Too close to API reset; do not start a new series')

    def allowed(self):
        if self.clock() >= self.period_end:
            self.close('api_period_admission_closed')
            return False
        allowed = super().allowed()
        if self.clock() >= self.period_end:
            self.close('api_period_admission_closed')
            return False
        return allowed

    def response_deadline(self, body, job):
        end, reserve, source = d.response_deadline(body, job)
        return min(end, self.period_end), reserve, source + '+api_period'


class SharedPeriodGuard(PeriodGuard):
    def __init__(self, *args, accounting, **kwargs):
        self.accounting = accounting
        super().__init__(*args, **kwargs)

    def allowed(self):
        allowed = super().allowed()
        if not self.accounting.observe(self.last):
            self.close(self.accounting.state['closed'])
            return False
        if self.clock() >= self.period_end:
            self.close('api_period_admission_closed')
            return False
        return allowed


def guest_report(job):
    code = """import pathlib,sqlite3,json
p=pathlib.Path('/task/.agent_work/diagnostic/journal.sqlite')
if not p.exists():print('{}');raise SystemExit
c=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True,timeout=1)
meta={k:json.loads(v) for k,v in c.execute('SELECT key,value FROM metadata')}
rows=[{'id':i,'spec':json.loads(s),'result':json.loads(r) if r else None} for i,s,r in c.execute('SELECT id,spec,result FROM experiments')]
print(json.dumps({'state':meta.get('state'),'calls':meta.get('llm_calls',0),'candidate_count':meta.get('candidate_count',0),'remaining':max(0,meta.get('deadline',0)-__import__('time').time()),'leader':meta.get('leader_id'),'score':meta.get('raw_best'),'control':meta.get('control_status'),'refit':meta.get('refit_status'),'last_result':meta.get('last_tool_result'),'experiments':rows}))
"""
    result = d.docker('exec', job['container'], 'python', '-c', code, check=False, timeout=6)
    return json.loads(result.stdout) if result.returncode == 0 else {'observation_unavailable': True}


def run(args):
    with ExitStack() as cleanup:
        return run_with_resources(args, cleanup)


def run_with_resources(args, cleanup):
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    args.output.mkdir(parents=True, exist_ok=False)
    key = d.limits.load_key(ROOT / '.env')
    before = d.limits.snapshot(d.limits.fetch_info(key), key, 'before')
    d.write(args.output / 'api_limits_before.json', before)
    if before['budget_exhausted'] or before['limits'].get('blocked') or before['remaining_budget'] is None:
        raise RuntimeError('Account unavailable; no live probe started')
    period_anchor = json.loads(args.period_anchor.read_text())
    if period_anchor['key_fingerprint'] != before['key_fingerprint']:
        raise RuntimeError('API key changed since authorization snapshot')
    same_period = period_anchor['limits'].get('budget_reset_at') == before['limits'].get('budget_reset_at')
    accounting = None
    if not same_period:
        # One persistent ledger and one process lock for every series using
        # this key/period. Do not give each job or restarted run its own 400.
        from hashlib import sha256
        identity = before['key_fingerprint'] + before['limits']['budget_reset_at']
        ledger = ROOT / 'outputs/quality_goal_20261023/shared_budget' / (sha256(identity.encode()).hexdigest()[:24] + '.json')
        accounting = SharedQualityBudget(ledger, before)
        cleanup.callback(accounting.release)
        if accounting.state['closed']:
            raise RuntimeError('Shared development budget closed: ' + accounting.state['closed'])
        period_cap, used = 400., accounting.spent
    else:
        comparison = d.limits.compare(period_anchor, before)
        if comparison.get('increase') is None:
            raise RuntimeError('Budget counters cannot be compared')
        period_cap, used = period_anchor['remaining_budget'], max(0., comparison['increase'])
    allowance = min(args.api_cap, before['remaining_budget'], period_cap - used)
    if allowance <= 5:
        raise RuntimeError('Insufficient authorized budget for another probe')
    d.register_secret(key)
    token = secrets.token_urlsafe(24); d.register_secret(token)
    archive = args.archive.resolve(); jobs = {}
    for kind, source in [('regression', 'Регрессия'), ('classification', 'Многоклассовая классификация')]:
        if args.task and kind not in args.task:
            continue
        task = args.output / kind / 'task'; task.mkdir(parents=True)
        hashes = {}
        for name in ('task.md', 'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv'):
            original = ROOT / 'data_for_participants' / source / name
            shutil.copyfile(original, task / name); hashes[name] = d.digest(original)
        with zipfile.ZipFile(archive) as package:
            assert all('/' not in n and (n.endswith('.py') or n == 'requirements.txt') for n in package.namelist())
            package.extractall(task)
        jobs[kind] = {'kind': kind, 'directory': str(task.resolve()), 'container': 'alpha-goal-' + args.output.name[-24:].replace('_','-') + '-' + kind,
            'input_hashes': hashes, 'started': False, 'finished': False, 'deadline_mono': None}
    fetch = lambda: d.limits.snapshot(d.limits.fetch_info(key), key, 'status')
    guard = SharedPeriodGuard(before, fetch, args.output, budget_cap=allowance, accounting=accounting) if accounting else PeriodGuard(before, fetch, args.output, budget_cap=allowance)
    # User-authorized goal budget replaces the historical 30%-of-account cap.
    guard.budget = allowance; guard.threshold = max(1., allowance - min(20., .1 * allowance))
    server = d.Gateway(('0.0.0.0', 0), guard, jobs, os.environ['OPENAI_BASE_URL'].rstrip('/'), key,
        deadline_reader=guard.response_deadline, client_key=token, response_decoder=decode_response, accounting=accounting)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    d.write(args.output / 'gateway_ready.json', {'port': server.server_port})
    done = threading.Event()
    def gateway_beat():
        while not done.is_set():
            d.write(args.output / 'gateway_heartbeat.json', {'mono':time.monotonic(),'utc':d.utc()})
            done.wait(2)
    heartbeat = threading.Thread(target=gateway_beat, daemon=True); heartbeat.start()
    manifest = {'archive': str(archive), 'archive_sha256': d.digest(archive), 'seconds_per_task': args.seconds,
        'cpu_per_task': args.cpu, 'memory_gib_per_task': args.memory_gib, 'api_allowance': allowance,
        'api_stop_threshold': guard.threshold, 'user_period_cap': period_cap, 'already_used_in_period': used,
        'same_period_as_initial_anchor': same_period, 'image': d.IMAGE, 'jobs': jobs,
        'api_admission_deadline_mono': guard.period_end,
        'new_period_generations_enabled': accounting is not None,
        'shared_budget_ledger': str(accounting.path) if accounting else None,
        'shared_admission_threshold': 300 if accounting else None,
        'shared_inflight_safety_margin': 100 if accounting else None,
        'shared_margin_is_published_price_bound': False,
        'shared_accounting_policy': 'account_counter_high_water_plus_reported_cost_conservative_double_count' if accounting else None,
        'development_max_output_tokens': accounting.max_output_tokens if accounting else None,
        'development_max_request_bytes': accounting.max_request_bytes if accounting else None,
        'development_guard_sources': {name: d.digest(ROOT / name) for name in
            ('tools/goal_quality_run.py', 'tools/shared_quality_budget.py', 'tools/diagnostic_3h.py')},
        'scope': 'Previously viewed public development tasks; no hidden score claim',
        'network': 'Provided model routed through authenticated local budget gateway; worker network guard retained',
        'sse': 'Upstream receives stream=true; gateway buffers the complete original SSE body, verifies termination, and preserves bytes/content-type downstream',
        'stock_submission_default_seconds_unchanged': 10800, 'paid_request_retries': False}
    d.write(args.output / 'MANIFEST.json', manifest)
    watchdog = None
    try:
        for kind, job in jobs.items():
            local = f'http://host.docker.internal:{server.server_port}/{kind}/v1'
            command = ['create', '--name', job['container'], '--cpus', str(args.cpu), '--memory', str(args.memory_gib)+'g',
                '--memory-swap', str(args.memory_gib)+'g', '-v', job['directory']+':/task', '-w','/task',
                '--env','PYTHONDONTWRITEBYTECODE=1','--env','OPENAI_API_KEY='+token,'--env','OPENAI_BASE_URL='+local,
                ]
            command += [d.IMAGE,'python','solution.py','task.md','--time-limit',str(args.seconds),'--cpu',str(args.cpu),
                '--memory-mb',str(args.memory_gib*1024),'--run-id','diagnostic']
            d.docker(*command)
            job.update(started=True,entry_mono=time.monotonic(),entry_utc=d.utc())
            # The inherited watchdog reserves 30s for the kill command. The
            # agent's original journal deadline still limits every API request.
            job['deadline_mono']=job['entry_mono']+args.seconds+30
            d.write(args.output/'CONFIG.json',{'jobs':jobs}); d.docker('start',job['container'])
        d.write(args.output/'observer_heartbeat.json',{'mono':time.monotonic(),'utc':d.utc()})
        with (args.output/'watchdog.log').open('w') as log:
            watchdog=subprocess.Popen([sys.executable,str(ROOT/'tools/diagnostic_3h.py'),'watchdog','--output',str(args.output)],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while not all(j['finished'] for j in jobs.values()):
            d.write(args.output/'observer_heartbeat.json',{'mono':time.monotonic(),'utc':d.utc()})
            progress={}
            for kind,job in jobs.items():
                if job['finished']:continue
                state=json.loads(d.docker('inspect','--format','{{json .State}}',job['container']).stdout)
                if not state['Running']:
                    job.update(finished=True,docker_state=state,finished_utc=d.utc())
                    log=d.docker('logs',job['container'],check=False)
                    (args.output/kind/'container.log').write_text(log.stdout+log.stderr)
                else:
                    progress[kind]=guest_report(job)
                    d.write(args.output/kind/'progress.json',progress[kind])
            d.write(args.output/'CONFIG.json',{'jobs':jobs})
            d.write(args.output/'PROGRESS.json',{'utc':d.utc(),'jobs':progress,'budget_used':guard.increase,'gate_closed':guard.reason})
            if watchdog.poll() is not None and not all(j['finished'] for j in jobs.values()):
                raise RuntimeError('Independent watchdog exited while an agent was running')
            time.sleep(2)
        results=[]
        for kind,job in jobs.items():
            path=Path(job['directory'])/'.agent_work/diagnostic/run_report.json'
            report=json.loads(path.read_text()) if path.is_file() else None
            csv=d.validate_csv(job)
            results.append({'kind':kind,'report':report,'csv':csv,'docker':job.get('docker_state')})
        d.write(args.output/'REPORT.json',{'archive_sha256':d.digest(archive),'results':results,
            'gate_reason':guard.reason,'upstream_outcome_unknown':server.unknown,
            'upstream_active_request':server.active,'quality_advantage_established':False,'jobs':jobs})
        print(json.dumps({'status':'finished','output':str(args.output),'jobs':[(x['kind'],(x['report'] or {}).get('control_status'),(x['report'] or {}).get('refit_status')) for x in results]}),flush=True)
    finally:
        for job in jobs.values():
            if job['started'] and not job['finished']:d.docker('kill',job['container'],check=False)
        d.write(args.output/'RUN_FINISHED',{'utc':d.utc()}); done.set()
        if watchdog and watchdog.poll() is None:watchdog.terminate();watchdog.wait(timeout=10)
        server.shutdown();server.server_close();thread.join(timeout=5);heartbeat.join(timeout=5)
        try:
            after=d.limits.snapshot(d.limits.fetch_info(key),key,'after');after['comparison']=d.limits.compare(before,after)
            d.write(args.output/'api_limits_after.json',after)
            if accounting: accounting.observe(after)
            print(json.dumps({'remaining_before':before['remaining_budget'],'remaining_after':after['remaining_budget'],
                'increase':after['comparison'].get('increase')}),flush=True)
        finally:
            if accounting: accounting.release()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--period-anchor',type=Path,required=True);p.add_argument('--seconds',type=int,default=1800)
    p.add_argument('--api-cap',type=float,default=160);p.add_argument('--cpu',type=int,default=5);p.add_argument('--memory-gib',type=float,default=4.5)
    p.add_argument('--task',action='append',choices=['regression','classification'])
    a=p.parse_args();a.output=a.output.resolve();run(a)
