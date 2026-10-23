"""Exact agent + real watchdog recovery after a locally lost model response."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnostic_3h as d
from tools.mock_task_resolution import fixture_resolution
from tools.test_diagnostic_3h import gateway
from tools.verify_api_v2_flat import fixture


def run(output, archive):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task = output / 'task'
    task.mkdir()
    with zipfile.ZipFile(archive) as package:
        package.extractall(task)
    fixture(task)
    name = 'alpha-exact-local-finish-' + uuid.uuid4().hex[:8]
    job = {'kind': 'regression', 'directory': str(task), 'container': name,
           'started': True, 'finished': False, 'deadline_mono': time.monotonic() + 150}

    def provider(handler, body, number):
        semantic = fixture_resolution(body)
        if semantic is not None:
            return 200, semantic
        handler.connection.shutdown(socket.SHUT_RDWR)
        handler.connection.close()
        return None

    watcher = None
    pulse = threading.Event()
    started = time.monotonic()
    try:
        with gateway(output / 'gateway', provider, exact=True, jobs={'regression': job}) as (server, guard, calls):
            root = guard.directory
            d.write(root / 'CONFIG.json', {'jobs': {'regression': job}})
            d.write(root / 'gateway_ready.json', {'port': server.server_port})

            def heartbeat():
                while not pulse.is_set():
                    d.write(root / 'gateway_heartbeat.json', {'mono': time.monotonic()})
                    pulse.wait(.05)

            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            code = 'from pathlib import Path;from tools.diagnostic_3h import watchdog_main;import sys;watchdog_main(Path(sys.argv[1]),poll=.05,stale=1,grace=1)'
            watcher = subprocess.Popen([sys.executable, '-c', code, str(root)], cwd=d.ROOT)
            try:
                d.docker('run', '-d', '--name', name, '--cpus', '1', '--memory', '2g', '--memory-swap', '2g',
                    '-v', str(task) + ':/task', '-w', '/task', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                    '-e', 'OPENAI_API_KEY=offline-fixture',
                    '-e', f'OPENAI_BASE_URL=http://host.docker.internal:{server.server_port}/regression/v1',
                    '-e', f'AGENT_DIAGNOSTICS_URL=http://host.docker.internal:{server.server_port}/regression/diagnostics',
                    d.IMAGE, 'python', 'solution.py', 'task.md', '--cpu', '1', '--memory-mb', '2048',
                    '--time-limit', '120', '--run-id', 'diagnostic')
                d.docker('wait', name, timeout=140)
                state = json.loads(d.docker('inspect', '--format', '{{json .State}}', name).stdout)
                log = d.docker('logs', name, check=False)
                (output / 'container.log').write_text(log.stdout + log.stderr)
                report = d.read(task / '.agent_work/diagnostic/run_report.json', {})
                failure = d.read(root / 'regression_watchdog_killed.json')
                unknown = [event for event in d.events(root) if event['kind'] == 'request_unknown']
                semantics = sum(fixture_resolution(call['body']) is not None for call in calls)
                passed = (state['ExitCode'] == 0 and failure is None and len(unknown) == 1 and
                    len(calls) - semantics == 1 and guard.closed and
                    report.get('control_status') == report.get('refit_status') == 'completed' and
                    report.get('publication', {}).get('scope') == 'full_train_refit')
                result = {'passed': passed, 'organizer_requests': 0, 'archive_sha256': d.digest(archive),
                    'supervisor_sources': d.supervisor_sources(), 'docker_state': state,
                    'watchdog': failure, 'watchdog_grace_seconds': 1, 'model_requests': len(calls),
                    'semantic_requests': semantics, 'generation_requests_with_lost_response': len(calls) - semantics,
                    'run_report': report, 'elapsed_seconds': time.monotonic() - started,
                    'scope': 'Synthetic lost response; confirms local full refit survives API closure, not live acceptance'}
                d.write(output / 'REPORT.json', result)
                print(json.dumps({key: result[key] for key in ('passed', 'model_requests', 'semantic_requests', 'elapsed_seconds')}))
                assert passed, result
            finally:
                (root / 'WATCHDOG_DONE').touch()
                watcher.wait(timeout=5)
                pulse.set()
                thread.join(timeout=2)
    finally:
        d.docker('rm', '-f', name, check=False)
        with zipfile.ZipFile(archive) as package:
            for member in package.namelist():
                path = task / member
                if path.is_file() and path.read_bytes() == package.read(member):
                    path.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.archive.resolve())
