"""Four full public runs of two immutable ZIPs, under one new API allowance.

Development supervisor only; never included in either delivered agent.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
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
from tools.goal_quality_run import SharedPeriodGuard, decode_response, guest_report
from tools.shared_quality_budget import SharedQualityBudget


class Allowance(SharedQualityBudget):
    """Known request charges and account deltas are two independent lower bounds.

    Take their maximum; do not charge our own settled requests twice. An unknown
    charge closes all four arms. The reserve is an estimate, not a price bound.
    """
    def __init__(self, path, before, cap):
        self.cap = before['limits']['spend'] + cap
        self.reserve = 20.
        self.arm_limit = (cap - self.reserve) / 4
        self.request_arms = {}
        super().__init__(path, before)

    @property
    def spent(self):
        initial = self.state['initial_spend']
        return initial + max(self.state['counter_spend'] - initial, self.state['reported_cost'])

    def arm_spend(self, arm):
        return sum(row['cost'] for row in self.state['requests'].values() if row.get('arm') == arm)

    def complete(self, identifier, status, usage):
        allowed = super().complete(identifier, status, usage)
        if identifier in self.state['requests']:
            self.state['requests'][identifier]['arm'] = self.request_arms[identifier]
            self.save()
        return allowed


class Handler(d.GatewayHandler):
    def link_request(self, identifier, task, raw):
        super().link_request(identifier, task, raw)
        self.server.accounting.request_arms[identifier] = task

    def handle_post(self):
        arm = self.path.strip('/').split('/')[0]
        if self.path.endswith('/chat/completions') and self.server.accounting.arm_spend(arm) >= self.server.accounting.arm_limit:
            return self.reply(402, {'error': {'code': 'equal_arm_api_allowance_exhausted'}})
        return super().handle_post()


def response_deadline(body, job):
    # The inherited reader understands strings; extract the pinned text block
    # for reserve checks while forwarding the original multimodal bytes intact.
    messages = []
    for message in body.get('messages', []):
        content = message.get('content')
        if isinstance(content, list):
            message = {**message, 'content': content[0]['text']}
        messages.append(message)
    return d.response_deadline({**body, 'messages': messages}, job)


def validate_pair(base, visual):
    def sources(path):
        import ast
        with zipfile.ZipFile(path) as package:
            tree = ast.parse(package.read('agent_bundle.py'))
            files = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'FILES' for target in node.targets))
            outer = {name: hashlib.sha256(package.read(name)).hexdigest() for name in package.namelist() if name != 'agent_bundle.py'}
        return files, outer
    a, outer_a = sources(base)
    b, outer_b = sources(visual)
    expected = {'automl_agent/context_memory.py', 'automl_agent/contracts.py', 'automl_agent/controller.py',
        'automl_agent/knowledge.py', 'automl_agent/qwen.py', 'automl_agent/research.py', 'automl_agent/tool_results.py',
        'automl_agent/visual_diagnostics.py', 'automl_agent/worker.py', 'context_compressor/contracts.py',
        'context_compressor/store.py', 'context_compressor/core.py', 'context_compressor/transport.py',
        'context_compressor/assembler.py'}
    changed = {name for name in a.keys() | b.keys() if a.get(name) != b.get(name)}
    assert changed == expected and outer_a == outer_b, ('Unexpected non-diagnostic changes', changed ^ expected)
    d.verify_archive(base)
    d.verify_archive(visual)
    return sorted(changed)


def finish_job(job, output):
    task = Path(job['directory'])
    report = d.read(task / '.agent_work/diagnostic/run_report.json', {})
    csv = d.validate_csv({**job, 'kind': job['task_kind']})
    result = {'arm': job['kind'], 'task': job['task_kind'], 'version': job['version'], 'report': report,
        'csv': csv, 'docker': job['docker_state'], 'elapsed_seconds': job['finished_mono'] - job['entry_mono']}
    result['technical_passed'] = (job['docker_state']['ExitCode'] == 0 and report.get('state') == 'completed'
        and report.get('control_status') == report.get('refit_status') == 'completed' and csv['passed'])
    d.write(output / job['kind'] / 'RESULT.json', result)
    return result


def run(args):
    from dotenv import load_dotenv
    changed = validate_pair(args.base, args.visual)
    args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as cleanup:
        load_dotenv(ROOT / '.env', override=False)
        key = d.limits.load_key(ROOT / '.env')
        before = d.limits.snapshot(d.limits.fetch_info(key), key, 'before')
        d.write(args.output / 'api_limits_before.json', before)
        if before['budget_exhausted'] or before['limits'].get('blocked') or before['remaining_budget'] is None or before['remaining_budget'] < args.cap:
            raise RuntimeError('Known unblocked account remainder of at least 200 required; no paid call sent')
        accounting = Allowance(args.output / 'budget_ledger.json', before, args.cap)
        cleanup.callback(accounting.release)
        token = secrets.token_urlsafe(32)
        d.register_secret(key)
        d.register_secret(token)
        info = json.loads(d.docker('info', '--format', '{{json .}}').stdout)
        assert info['NCPU'] >= 2 * args.cpu and info['MemTotal'] >= (2 * args.memory_gib + 1) * d.GIB
        assert shutil.disk_usage(args.output).free >= 15 * d.GIB
        assert d.docker('network', 'inspect', args.network, '--format', '{{json .Internal}}').stdout.strip() == 'true'
        image_id = d.docker('image', 'inspect', d.IMAGE, '--format', '{{.Id}}').stdout.strip()
        jobs = {}
        for task_kind, source in [('regression', 'Регрессия'), ('classification', 'Многоклассовая классификация')]:
            for version, archive in [('base', args.base), ('visual', args.visual)]:
                arm = task_kind + '_' + version
                directory = args.output / arm / 'task'
                directory.mkdir(parents=True)
                hashes = {}
                for name in ('task.md', 'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv'):
                    original = ROOT / 'data_for_participants' / source / name
                    shutil.copyfile(original, directory / name)
                    hashes[name] = d.digest(original)
                with zipfile.ZipFile(archive) as package:
                    package.extractall(directory)
                jobs[arm] = {'kind': arm, 'task_kind': task_kind, 'version': version, 'directory': str(directory),
                    'container': 'alpha-visual-ab-' + args.output.name[-8:].replace('_', '-') + '-' + arm.replace('_', '-'), 'input_hashes': hashes,
                    'archive_sha256': d.digest(archive), 'started': False, 'finished': False, 'deadline_mono': None}
        fetch = lambda: d.limits.snapshot(d.limits.fetch_info(key), key, 'status')
        guard = SharedPeriodGuard(before, fetch, args.output, budget_cap=args.cap, accounting=accounting)
        guard.budget, guard.threshold = args.cap, args.cap - accounting.reserve
        def deadline(body, job):
            end, reserve, source = response_deadline(body, job)
            return min(end, guard.period_end), reserve, source + '+api_period'
        server = d.Gateway(('0.0.0.0', 0), guard, jobs, os.environ['OPENAI_BASE_URL'].rstrip('/'), key,
            deadline_reader=deadline, client_key=token, response_decoder=decode_response, accounting=accounting)
        server.RequestHandlerClass = Handler
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        d.write(args.output / 'gateway_ready.json', {'port': server.server_port})
        relay = 'alpha-visual-relay-' + args.output.name[-8:].replace('_', '-')
        d.docker('create', '--name', relay, '--network', args.network, '--cpus', '.25', '--memory', '192m',
            '-v', str(ROOT / 'tools/visual_ab_relay.py') + ':/relay.py:ro', '--env', 'GATEWAY_PORT=' + str(server.server_port),
            d.IMAGE, 'python', '/relay.py')
        d.docker('network', 'connect', 'bridge', relay)
        d.docker('start', relay)
        def close_relay():
            d.docker('stop', '-t', '1', relay, check=False)
            d.docker('rm', relay, check=False)
        cleanup.callback(close_relay)
        done = threading.Event()
        def beat():
            while not done.is_set():
                for owner in ('observer', 'gateway'):
                    d.write(args.output / (owner + '_heartbeat.json'), {'mono': time.monotonic(), 'utc': d.utc()})
                done.wait(2)
        heartbeat = threading.Thread(target=beat, daemon=True)
        heartbeat.start()
        manifest = {'base_archive': str(args.base), 'visual_archive': str(args.visual), 'only_changed_sources': changed,
            'image_id': image_id, 'network': args.network, 'external_internet': False,
            'relay': {'name': relay, 'provider_key': False, 'only_destination': 'host_budget_gateway',
                'source_sha256': d.digest(ROOT / 'tools/visual_ab_relay.py')},
            'seconds_per_task': args.seconds, 'cpu_per_task': args.cpu, 'memory_gib_per_task': args.memory_gib,
            'max_calls': None, 'max_candidates': None, 'research_focus_override': None, 'model': 'qwen/qwen3.8-27b',
            'api_cap': args.cap, 'new_generation_stop_threshold': guard.threshold,
            'inflight_reserve': accounting.reserve, 'reserve_is_published_price_bound': False,
            'arm_request_admission_cost_limit': accounting.arm_limit,
            'accounting': 'max(sum_known_response_costs, account_counter_increase); stop all on unknown billing',
            'request_order': 'FIFO serialized across concurrent arms', 'waves': ['regression', 'classification'],
            'seed': 42, 'quality_scope': 'Previously inspected public data, independent closed controls; one run per arm',
            'old_ledgers_reopened': False, 'source_sha256': d.digest(Path(__file__)), 'started_utc': d.utc(), 'jobs': jobs}
        d.write(args.output / 'MANIFEST.json', manifest)
        shutil.copyfile(__file__, args.output / 'benchmark_source.py')
        d.write(args.output / 'CONFIG.json', {'jobs': jobs})
        results, watchdog = [], None
        print(json.dumps({'status': 'ready', 'remaining_before': before['remaining_budget'], 'account_limit': before['limits']['max_budget'],
            'series_cap': args.cap, 'seconds_per_run': args.seconds}), flush=True)
        try:
            # Check the complete isolated path before starting either task clock.
            probe = d.docker('run', '--rm', '--network', args.network, '--cpus', '.25', '--memory', '192m',
                d.IMAGE, 'python', '-c', 'import urllib.request;print(urllib.request.urlopen("http://' + relay + ':18767/health",timeout=10).read().decode())')
            assert json.loads(probe.stdout)['alive'] is True
            d.write(args.output / 'network_preflight.json', {'passed': True, 'paid_requests': 0,
                'isolated_container_to_relay_to_accounted_gateway': True})
            shutil.copyfile(ROOT / 'tools/visual_ab_relay.py', args.output / 'relay_source.py')
            with (args.output / 'watchdog.log').open('w') as log:
                watchdog = subprocess.Popen([sys.executable, str(ROOT / 'tools/diagnostic_3h.py'), 'watchdog', '--output', str(args.output)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            for task_kind in ('regression', 'classification'):
                if guard.closed:
                    break
                wave = [job for job in jobs.values() if job['task_kind'] == task_kind]
                # Reverse launch order in the second pair; both use the same FIFO gateway.
                for job in wave if task_kind == 'regression' else reversed(wave):
                    environment = {**os.environ, 'OPENAI_API_KEY': token,
                        'OPENAI_BASE_URL': f"http://{relay}:18767/{job['kind']}/v1"}
                    command = ['docker', 'create', '--name', job['container'], '--network', args.network,
                        '--cpus', str(args.cpu), '--memory', str(args.memory_gib) + 'g', '--memory-swap', str(args.memory_gib) + 'g',
                        '-v', job['directory'] + ':/task', '-w', '/task', '--env', 'PYTHONDONTWRITEBYTECODE=1',
                        '--env', 'OPENAI_API_KEY', '--env', 'OPENAI_BASE_URL', d.IMAGE,
                        'python', 'solution.py', 'task.md', '--cpu', str(args.cpu), '--memory-mb', str(args.memory_gib * 1024),
                        '--run-id', 'diagnostic']
                    created = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
                    if created.returncode:
                        raise RuntimeError('Container creation failed: ' + created.stderr[:300])
                    job.update(started=True, entry_mono=time.monotonic(), entry_utc=d.utc())
                    job['deadline_mono'] = job['entry_mono'] + args.seconds + 30
                    d.write(args.output / 'CONFIG.json', {'jobs': jobs})
                    d.docker('start', job['container'])
                last_detail = 0
                while not all(job['finished'] for job in wave):
                    progress = {}
                    for job in wave:
                        if job['finished']:
                            continue
                        state = json.loads(d.docker('inspect', '--format', '{{json .State}}', job['container']).stdout)
                        if not state['Running']:
                            job.update(finished=True, docker_state=state, finished_utc=d.utc(), finished_mono=time.monotonic())
                            logs = d.docker('logs', job['container'], check=False)
                            (args.output / job['kind'] / 'container.log').write_text(logs.stdout + logs.stderr)
                            results.append(finish_job(job, args.output))
                        elif time.monotonic() - last_detail > 20:
                            progress[job['kind']] = guest_report(job)
                    if progress:
                        last_detail = time.monotonic()
                        d.write(args.output / 'PROGRESS.json', {'utc': d.utc(), 'jobs': progress,
                            'known_response_cost': accounting.state['reported_cost'], 'account_increase': guard.increase,
                            'gate_closed': guard.reason, 'arm_spend': {arm: accounting.arm_spend(arm) for arm in jobs}})
                    d.write(args.output / 'CONFIG.json', {'jobs': jobs})
                    if watchdog.poll() is not None and not all(job['finished'] for job in wave):
                        raise RuntimeError('Independent watchdog exited')
                    if time.monotonic() - last_detail > 60:
                        raise RuntimeError('Progress observation unavailable')
                    time.sleep(2)
                d.write(args.output / 'REPORT.json', {'results': results, 'jobs': jobs, 'gate_reason': guard.reason,
                    'all_four_finished': len(results) == 4, 'quality_advantage_established': False})
                print(json.dumps({'status': 'wave_finished', 'task': task_kind, 'known_cost': accounting.state['reported_cost'],
                    'gate_reason': guard.reason}), flush=True)
        finally:
            for job in jobs.values():
                if job['started'] and not job['finished']:
                    d.docker('kill', job['container'], check=False)
            d.write(args.output / 'RUN_FINISHED', {'utc': d.utc()})
            done.set()
            if watchdog and watchdog.poll() is None:
                watchdog.terminate()
                watchdog.wait(timeout=10)
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            heartbeat.join(timeout=5)
            after = d.limits.snapshot(d.limits.fetch_info(key), key, 'after')
            after['comparison'] = d.limits.compare(before, after)
            d.write(args.output / 'api_limits_after.json', after)
            print(json.dumps({'remaining_before': before['remaining_budget'], 'remaining_after': after['remaining_budget'],
                'counter_increase': after['comparison'].get('increase'), 'known_cost': accounting.state['reported_cost']}), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--visual', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.base, args.visual, args.output = args.base.resolve(), args.visual.resolve(), args.output.resolve()
    args.cap, args.seconds, args.cpu, args.memory_gib, args.network = 200., 10800, 5, 5, 'alpha-v010-llm'
    raise SystemExit(run(args))
