"""Real-container distinction between model admission closure and diagnostic loss."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnostic_3h as d


def case(root, reason, fault=None):
    root.mkdir()
    name = 'alpha-gate-reason-' + uuid.uuid4().hex[:8]

    class Health(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            raw = json.dumps({'closed': True, 'reason': reason}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    health = ThreadingHTTPServer(('127.0.0.1', 0), Health)
    threading.Thread(target=health.serve_forever, daemon=True).start()
    done = threading.Event()

    def heartbeat():
        while not done.is_set():
            now = time.monotonic()
            d.write(root / 'observer_heartbeat.json', {'mono': now - (1 if fault == 'heartbeat' else 0)})
            d.write(root / 'gateway_heartbeat.json', {'mono': now})
            done.wait(.03)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    d.write(root / 'gateway_ready.json', {'port': health.server_port})
    d.write(root / 'CLOSED.json', {'reason': reason})
    (root / 'CLOSE').touch()
    if fault == 'agent_latch':
        signal = root / 'signals/test'
        signal.mkdir(parents=True)
        (signal / 'ARMED').touch()  # No HEALTHY marker: independent diagnostic failure.
    watcher = None
    try:
        d.docker('run', '-d', '--name', name, '--network', 'none', '--cpus', '1', '--memory', '128m',
                 '-v', str(root) + ':/evidence', d.IMAGE, 'python', '-c',
                 'import time,pathlib;time.sleep(1.2);pathlib.Path("/evidence/LOCAL_FINISHED").touch()')
        d.write(root / 'CONFIG.json', {'jobs': {'test': {'kind': 'test', 'container': name,
            'started': True, 'finished': False, 'deadline_mono': time.monotonic() + 4}}})
        code = 'from pathlib import Path;from tools.diagnostic_3h import watchdog_main;import sys;watchdog_main(Path(sys.argv[1]),poll=.02,stale=.3,grace=.6)'
        watcher = subprocess.Popen([sys.executable, '-c', code, str(root)], cwd=d.ROOT)
        until = time.monotonic() + 6
        while True:
            state = json.loads(d.docker('inspect', '--format', '{{json .State}}', name).stdout)
            if not state['Running']:
                break
            assert time.monotonic() < until, 'container_not_stopped'
            time.sleep(.05)
        killed = d.read(root / 'test_watchdog_killed.json')
        expected_stop = fault is not None or reason == 'required_diagnostics_failed'
        passed = (state['ExitCode'] != 0 and not (root / 'LOCAL_FINISHED').exists()) if expected_stop else (
            state['ExitCode'] == 0 and (root / 'LOCAL_FINISHED').exists() and killed is None)
        return {'passed': passed, 'closure': reason, 'independent_fault': fault,
                'expected_infrastructure_stop': expected_stop, 'exit_code': state['ExitCode'],
                'local_finalization_finished': (root / 'LOCAL_FINISHED').exists(), 'watchdog': killed}
    finally:
        (root / 'WATCHDOG_DONE').touch()
        if watcher is not None:
            watcher.wait(timeout=5)
        done.set()
        thread.join(timeout=2)
        health.shutdown()
        health.server_close()
        d.docker('rm', '-f', name, check=False)


def run(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for index, (reason, fault) in enumerate([
        ('upstream_outcome_unknown_at_deadline', None),
        ('upstream_outcome_unknown', None),
        ('budget_threshold_or_blocked', None),
        ('required_diagnostics_failed', None),
        ('upstream_outcome_unknown_at_deadline', 'heartbeat'),
        ('upstream_outcome_unknown_at_deadline', 'agent_latch'),
    ]):
        row = case(output / ('case_' + str(index)), reason, fault)
        cases.append(row)
        d.write(output / 'REPORT.json', {'cases': cases, 'organizer_requests': 0,
                'passed': False, 'supervisor_sources': d.supervisor_sources()})
        print(json.dumps(row), flush=True)
    result = {'passed': all(row['passed'] for row in cases), 'cases': cases,
              'organizer_requests': 0, 'supervisor_sources': d.supervisor_sources(),
              'scope': 'Real child containers and production watchdog, shortened clocks; full archive recovery is separate'}
    d.write(output / 'REPORT.json', result)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output))
