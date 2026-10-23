"""External diagnostic protocol v2 supervisor. Never included in the submission ZIP."""
from __future__ import annotations

import argparse
import ast
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import hmac
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
import os
from pathlib import Path
import select
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zipfile

import httpx
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diagnostic_store import Store, register_secret, scrub, scan_secrets, ARTIFACT_MAX, SEGMENT_MAX, STORE_MAX

ROOT = Path(__file__).resolve().parents[1]
SHA = 'ef92e88e4c6164f006bfd9b03741cdce08f03205cfb14ebec6f792ed6dcee0d4'
IMAGE = 'qwen-ml-agent:exact'
GIB = 1024**3
spec = importlib.util.spec_from_file_location('alpha_limits', ROOT / 'skills/alpha-api-limits/scripts/check_limits.py')
limits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(limits)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


_log_lock = threading.Lock()


def event(directory, kind, **fields):
    Store(directory).append(kind, task=fields.pop('task', None), source='supervisor_gateway_watchdog', **fields)


def events(directory):
    return [{**record, **record['payload']} for record in Store(directory).records()]


def capture_exception(directory,source,*,task=None):
    import traceback
    saved=Store(directory).put(traceback.format_exc(),name=source+'_traceback.txt',task=task)
    event(directory,source+'_exception',task=task,artifact=saved['path'],outcome='failed')
    return saved


def docker(*args, check=True, timeout=25):
    result = subprocess.run(['docker', *map(str, args)], capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError('docker ' + str(args[0]) + ': ' + result.stderr[:500])
    return result


def supervisor_sources():
    paths=('tools/diagnostic_3h.py','tools/diagnostic_store.py','skills/alpha-api-limits/scripts/check_limits.py')
    return {name:digest(ROOT/name) for name in paths}


_metadata_cache = {}


def metadata(directory, container=None):
    # Never open a guest's live WAL/shm with the host SQLite VFS. Docker Desktop
    # shares bytes, but macOS and Linux do not share SQLite's locking domain.
    if container:
        cached = _metadata_cache.get(container)
        if cached and time.monotonic()-cached[0] < .5:
            return cached[1]
        code = "import sqlite3,json,pathlib; p=pathlib.Path('/task/.agent_work/diagnostic/journal.sqlite'); c=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True,timeout=1) if p.exists() else None; print(json.dumps({k:json.loads(v) for k,v in c.execute('SELECT key,value FROM metadata')} if c else {}))"
        result = docker('exec', container, 'python', '-c', code, check=False, timeout=5)
        if result.returncode:
            state = json.loads(docker('inspect', '--format', '{{json .State}}', container).stdout)
            if state['Running']:
                raise RuntimeError('guest_metadata_failed:'+result.stderr[:200])
            result = docker('run', '--rm', '--network', 'none', '-v', str(directory)+':/task', IMAGE,
                            'python', '-c', code, timeout=10)
        value = json.loads(result.stdout)
        _metadata_cache[container] = (time.monotonic(), value)
        return value
    path = Path(directory) / '.agent_work/diagnostic/journal.sqlite'
    if not path.exists():
        return {}
    with sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True, timeout=1) as db:
        return {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}


def backup_sqlite(directory,job,*,reason):
    started=time.monotonic()
    target_root=Path(directory)/job['kind']/'sqlite_backups'
    target_root.mkdir(parents=True,exist_ok=True)
    rows=[]
    export_root = None
    if job.get('container'):
        export_root = target_root/('guest_export_'+uuid.uuid4().hex)
        export_root.mkdir()
        code = """import sqlite3,pathlib,time,json
root=pathlib.Path('/task/.agent_work/diagnostic'); end=time.monotonic()+110
def progress(*_):
 if time.monotonic()>end: raise TimeoutError('guest_backup_deadline')
for source in root.rglob('*.sqlite'):
 target=pathlib.Path('/export')/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
 with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True,timeout=1) as db:
  with sqlite3.connect(target) as out:
   db.backup(out,pages=128,sleep=.01,progress=progress)
   if out.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('guest_backup_integrity')
"""
        # A separate helper stays in the guest's kernel even when the agent exits.
        # Host sanitation starts only after this helper closes every exported DB.
        docker('run', '--rm', '--network', 'none', '--cpus', '.5', '--memory', '256m',
               '-v', job['directory']+':/task', '-v', str(export_root.resolve())+':/export',
               IMAGE, 'python', '-c', code, timeout=115)
    source_root = export_root or (Path(job['directory'])/'.agent_work/diagnostic')
    sources=list(source_root.rglob('*.sqlite'))
    if not sources and reason in {'experiment_finished','agent_finished'}:
        raise ValueError('mandatory_sqlite_source_missing')
    copy_deadline=started+120
    def progress(*_):
        if time.monotonic()>=copy_deadline:raise TimeoutError('sqlite_backup_deadline')
    for source in sources:
        name=str(source.relative_to(source_root)).replace('/','_')
        target=target_root/(str(time.time_ns())+'_'+name)
        pending=target.with_suffix('.pending')
        with closing(sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True,timeout=1)) as db:
            with closing(sqlite3.connect(pending)) as backup:
                db.backup(backup,pages=128,sleep=.01,progress=progress)
                backup.set_progress_handler(lambda: int(time.monotonic()>=copy_deadline), 1000)
                backup.execute('PRAGMA secure_delete=ON')
                backup.create_function('diagnostic_scrub',1,lambda text:scrub(text).decode('utf-8'))
                def scrub_blob(value):
                    try:value.decode('utf-8')
                    except UnicodeDecodeError:return value
                    return scrub(value)
                backup.create_function('diagnostic_scrub_blob',1,scrub_blob)
                for (table,) in backup.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall():
                    quoted='"'+table.replace('"','""')+'"'
                    for column in backup.execute('PRAGMA table_info('+quoted+')').fetchall():
                        sql_name='"'+column[1].replace('"','""')+'"'
                        backup.execute('UPDATE '+quoted+' SET '+sql_name+'=diagnostic_scrub('+sql_name+') WHERE typeof('+sql_name+")='text'")
                        backup.execute('UPDATE '+quoted+' SET '+sql_name+'=diagnostic_scrub_blob('+sql_name+') WHERE typeof('+sql_name+")='blob'")
                backup.commit()
                backup.execute('VACUUM')
                backup.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                backup.execute('PRAGMA journal_mode=DELETE')
                assert backup.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        os.replace(pending,target)
        saved=Store(directory).put(target.read_bytes(),name=name,task=job['kind'],identity='sqlite_backup_'+reason,binary=True)
        rows.append(saved)
        with closing(sqlite3.connect(target)) as snapshot_db:
            tables={row[0] for row in snapshot_db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'operations' in tables:
                for (raw_record,) in snapshot_db.execute('SELECT record FROM operations'):
                    record=json.loads(raw_record)
                    # Content identity makes unchanged observations idempotent across backups.
                    Store(directory).receive(job['kind'], {'kind':'operation_record','payload':{'record':record},
                        'clock_domain':'external_sqlite_snapshot'})
        target.unlink()
    if export_root:
        shutil.rmtree(export_root)
    event(directory,'sqlite_backup',task=job['kind'],reason=reason,seconds=time.monotonic()-started,files=rows)
    return rows



def emergency_reason(directory):
    directory=Path(directory)
    for marker in (directory/'signals').glob('*/ARMED'):
        root=marker.parent
        if not (root/'HEALTHY').exists() or (root/'FAILURE.json').exists():
            return 'agent_diagnostics_failed:'+root.name
    ledger=directory/'history/ledger.sqlite'
    if ledger.exists():
        with closing(sqlite3.connect('file:'+str(ledger)+'?mode=ro',uri=True,timeout=.2)) as db:
            if db.execute("SELECT 1 FROM jobs WHERE status='failed' OR (status IN ('queued','running') AND deadline<?) LIMIT 1",(time.time(),)).fetchone():
                return 'required_backup_failed_or_overdue'
    return None


def safe_event(directory,kind,**fields):
    try:event(directory,kind,**fields)
    except Exception:
        # Auditing is secondary to a latched protective action.
        pass


class BackupExecutor:
    """One durable bounded queue and one copier per task, shared by all producers."""
    def __init__(self,directory,job,guard):
        self.directory,self.job,self.guard=Path(directory),job,guard
        self.store=Store(directory);self.done=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def run(self):
        try:
            # A previous copier's running job has unknown outcome; never call it ready.
            for old in self.store.jobs():
                if old['task']==self.job['kind'] and old['status']=='running':
                    self.store.finish_job(old['id'],'failed',{'reason':'copier_interrupted'})
                    raise RuntimeError('backup_copier_interrupted')
            while not self.done.is_set():
                item=self.store.claim(self.job['kind'])
                if not item:
                    self.done.wait(.1);continue
                if time.time()>=item['deadline']:raise RuntimeError('backup_overdue')
                try:
                    rows=backup_sqlite(self.directory,self.job,reason=item['reason'])
                    if time.time()>=item['deadline']:raise RuntimeError('backup_overdue')
                    self.store.finish_job(item['id'],'completed' if rows else 'skipped',{'files':rows,'missing_reason':None if rows else 'source_not_created'})
                    event(self.directory,'backup_ready' if rows else 'backup_skipped',task=self.job['kind'],job_id=item['id'],files=rows,missing_reason=None if rows else 'source_not_created')
                except Exception as error:
                    self.guard.close('required_backup_failed')
                    self.store.finish_job(item['id'],'failed',{'error_type':type(error).__name__})
                    raise
        except Exception:
            self.guard.close('required_backup_failed')
            try:capture_exception(self.directory,'backup',task=self.job['kind'])
            except Exception:pass


def response_deadline(body, job):
    """Read the exact stock reserve, without adding headers or editing v022."""
    meta = metadata(job['directory'], job.get('container'))
    seconds = meta.get('limits', {}).get('seconds')
    deadline = meta.get('deadline')
    if not isinstance(seconds, (int, float)) or not isinstance(deadline, (int, float)):
        raise ValueError('missing_agent_deadline')
    reserve = min(300., seconds * 5 / 180)
    source = 'stock_Budget.final_reserve_initial'
    for message in body.get('messages', []):
        if message.get('role') != 'user':
            continue
        try:
            content = json.loads(message['content'])
        except (ValueError, TypeError, KeyError):
            continue
        resources = content.get('core', {}).get('policies_resources', {})
        if resources:
            reserve = max(reserve, float(resources['reserve_seconds']))
            if abs(float(resources['deadline_epoch']) - deadline) > .01:
                raise ValueError('context_deadline_mismatch')
            source = 'wire_core.policies_resources'
        elif 'remaining' in content and isinstance(content['remaining'], dict):
            reserve = max(reserve, float(content['remaining'].get('reserve_seconds', 0)))
            source = 'wire_remaining'
    # One second conservatism covers journal/HTTP setup before the stock alarm.
    end = min(job['deadline_mono'], time.monotonic() + deadline - time.time()) - reserve - 1
    return end, reserve, source


class Guard:
    def __init__(self, before, fetch, directory, *, clock=time.monotonic, disk=None, budget_cap=180):
        self.before, self.fetch, self.directory, self.clock = before, fetch, Path(directory), clock
        self.directory.mkdir(parents=True,exist_ok=True)
        self.disk = disk or (lambda: shutil.disk_usage(self.directory).free)
        self.budget = min(float(budget_cap), .30 * before['remaining_budget'])
        self.threshold = .90 * self.budget
        self.last, self.last_at, self.failure_at = before, clock(), None
        self.closed, self.reason, self.increase = False, None, 0.
        self.lock = threading.RLock()
        self.sample_no = 0

    def close(self, reason):
        with self.lock:
            if not self.closed:
                self.closed, self.reason = True, reason
                # Close memory and the independent latch before any audit I/O.
                try:(self.directory / 'CLOSE').touch()
                except OSError:pass
                try:write(self.directory / 'CLOSED.json', {'schema_version':2,'reason': reason, 'utc': utc(), 'mono':self.clock()})
                except OSError:pass
                safe_event(self.directory, 'gate_closed', reason=reason)

    def refresh(self, force=False):
        with self.lock:
            if self.closed:
                return False
            now = self.clock()
            if not force and now - self.last_at <= 15 and self.failure_at is None:
                return True
            try:
                fresh = self.fetch()
                comparison = limits.compare(self.before, fresh)
                self.sample_no += 1
                write(self.directory / 'limits' / f'{self.sample_no:06d}.json', fresh)
                if comparison.get('increase') is None:
                    self.close(comparison.get('reason', 'incomparable_counters'))
                    return False
                # A reset never grows the frozen budget; changing periods closes it.
                old_reset = self.last['limits'].get('budget_reset_at')
                new_reset = fresh['limits'].get('budget_reset_at')
                if old_reset != new_reset:
                    self.close('period_changed')
                    return False
                if limits.compare(self.last, fresh).get('increase') is None:
                    self.close('counter_reset')
                    return False
                self.last, self.last_at, self.failure_at = fresh, self.clock(), None
                self.increase = comparison['increase']
                if fresh['limits'].get('blocked') or fresh.get('budget_exhausted') or self.increase >= self.threshold:
                    self.close('budget_threshold_or_blocked')
                    return False
                return True
            except Exception as exc:
                self.failure_at = now if self.failure_at is None else self.failure_at
                event(self.directory, 'limits_error', error=type(exc).__name__)
                if self.clock() - self.failure_at >= 60:
                    self.close('accounting_unavailable_60s')
                return False

    def allowed(self):
        if self.closed:
            return False
        try:failure=emergency_reason(self.directory)
        except Exception:failure='diagnostic_state_unavailable'
        if failure:
            self.close(failure)
            return False
        if (self.directory / 'CLOSE').exists():
            self.close('normal_agents_finished' if (self.directory/'RUN_FINISHED').exists() else 'watchdog_or_supervisor_closed')
            return False
        heartbeat = read(self.directory / 'observer_heartbeat.json', {})
        if self.clock() - heartbeat.get('mono', -1e20) > 15:
            self.close('observer_heartbeat_expired')
            return False
        if self.disk() < 5 * GIB:
            self.close('disk_below_5gib')
            return False
        return self.refresh()


def disconnected(sock):
    try:
        ready, _, _ = select.select([sock], [], [], 0)
        return bool(ready) and sock.recv(1, socket.MSG_PEEK) == b''
    except (OSError, ValueError):
        return True


class Gateway(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, guard, jobs, upstream, key, *, deadline_reader=response_deadline, client_key=None, response_decoder=None, accounting=None):
        self.guard, self.jobs, self.upstream, self.key = guard, jobs, upstream, key
        self.deadline_reader = deadline_reader
        self.client_key = client_key
        self.response_decoder = response_decoder
        self.accounting = accounting
        self.queue, self.condition = [], threading.Condition()
        self.active, self.unknown = None, False
        self.upstream_response_reserve = 0.
        self.store, self.links, self.task_sequence, self.body_attempts = Store(guard.directory), {}, {}, {}
        super().__init__(address, GatewayHandler)
        self.backups={task:BackupExecutor(guard.directory,job,guard) for task,job in jobs.items()}

    def server_close(self):
        for executor in self.backups.values():executor.done.set()
        for executor in self.backups.values():executor.thread.join(timeout=2)
        super().server_close()


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def reply(self, status, body, *, content_type='application/json'):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        if getattr(self,'gateway_request_id',None):
            saved=self.server.store.put(data,name='gateway_http_reply.txt',task=self.gateway_task,identity=self.gateway_request_id)
            event(self.server.guard.directory,'gateway_http_reply',task=self.gateway_task,request=self.gateway_request_id,
                  status=status,artifact=saved['path'],sha256=saved['sha256'])
        try:
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        self.close_connection = True

    def link_request(self,identifier,task,raw):
        server,guard=self.server,self.server.guard
        wire_hash = hashlib.sha256(raw).hexdigest()
        server.task_sequence[task] = server.task_sequence.get(task,0)+1
        attempt_key=(task,wire_hash)
        server.body_attempts[attempt_key]=server.body_attempts.get(attempt_key,0)+1
        matches=server.links.get(attempt_key,[])
        link=matches.pop(0) if len(matches)==1 else {}
        request_artifact=server.store.put(raw,name='qwen_request.json',task=task,identity=identifier)
        event(guard.directory,'request_link',task=task,request=identifier,wire_sha256=wire_hash,
            task_sequence=server.task_sequence[task],body_attempt=server.body_attempts[attempt_key],
            call_id=link.get('call_id'),request_id=link.get('request_id'),
            missing_reason=None if link else 'no_unique_agent_exchange_start',
            request_id_missing_reason=link.get('request_id_missing_reason'),request_artifact=request_artifact['path'])

    def do_GET(self):
        self.reply(200, {'alive': True, 'closed': self.server.guard.closed, 'reason': self.server.guard.reason, 'active': self.server.active, 'unknown': self.server.unknown})

    def do_POST(self):
        try:return self.handle_post()
        except Exception as error:
            self.server.guard.close('required_diagnostics_failed')
            try:capture_exception(self.server.guard.directory,'gateway',task=getattr(self,'gateway_task',None))
            except Exception:pass
            self.gateway_request_id=None
            return self.reply(507,{'error':{'code':'diagnostic_infrastructure_failed','type':type(error).__name__}})

    def handle_post(self):
        server, guard = self.server, self.server.guard
        if server.client_key and not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+server.client_key):
            return self.reply(403, {'error': {'code': 'local_auth_required'}})
        if self.path.endswith('/diagnostics'):
            task = self.path.strip('/').split('/')[0]
            if task not in server.jobs:
                return self.reply(404, {'error': {'code':'unknown_task'}})
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 2*ARTIFACT_MAX:
                    raise ValueError('diagnostic_body_limit')
                body = json.loads(self.rfile.read(length))
                receipt=server.store.receive(task,body)
                if body.get('kind') == 'qwen_exchange_start' and not receipt.get('duplicate'):
                    fields=body['payload']
                    server.links.setdefault((task,fields['wire_sha256']),[]).append(fields)
                return self.reply(200,receipt)
            except Exception as exc:
                import traceback
                guard.close('required_diagnostics_failed')
                try:server.store.put(traceback.format_exc(),name='collector_traceback.txt',task=task)
                except Exception:pass
                return self.reply(507,{'error':{'code':'diagnostic_capture_failed','type':type(exc).__name__}})
        task = self.path.strip('/').split('/')[0]
        if task not in server.jobs or not self.path.endswith('/chat/completions'):
            return self.reply(404, {'error': {'code': 'unknown_route'}})
        identifier = uuid.uuid4().hex
        self.gateway_request_id,self.gateway_task=identifier,task
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4 * 1024**2:
                raise ValueError('invalid_length')
            raw = self.rfile.read(length)
            self.link_request(identifier,task,raw)
            body = json.loads(raw)
            # The gateway starts before containers receive their own deadlines.
            # Read the committed registry, rather than the pre-launch snapshot.
            current=read(guard.directory/'CONFIG.json',{}).get('jobs',{}).get(task)
            if current is not None:
                server.jobs[task]=current
            deadline, reserve, source = server.deadline_reader(body, server.jobs[task])
        except Exception:
            guard.close('deadline_unavailable')
            trace=capture_exception(guard.directory,'request_admission',task=task)
            event(guard.directory,'request_rejected',task=task,request=identifier,reason=guard.reason,artifact=trace['path'])
            return self.reply(402, {'error': {'code': guard.reason}})
        queued = time.monotonic()
        event(guard.directory, 'request_queued', request=identifier, task=task, response_deadline_mono=deadline,
              reserve=reserve, deadline_source=source, effort=body.get('reasoning', {}).get('effort'),
              max_tokens=body.get('max_tokens'), temperature=body.get('temperature'))
        with server.condition:
            server.queue.append(identifier)
        try:
            while True:
                if disconnected(self.connection):
                    event(guard.directory, 'cancelled_before_send', request=identifier, task=task, reason='client_disconnected')
                    return
                if time.monotonic() >= deadline:
                    event(guard.directory, 'cancelled_before_send', request=identifier, task=task, reason='deadline')
                    return self.reply(402, {'error': {'code': 'response_deadline'}})
                if not guard.allowed():
                    if guard.closed:
                        event(guard.directory,'request_rejected',task=task,request=identifier,reason=guard.reason,outcome='rejected')
                        return self.reply(402, {'error': {'code': guard.reason}})
                    time.sleep(.1)
                    continue
                remaining = deadline-time.monotonic()
                if remaining < server.upstream_response_reserve:
                    event(guard.directory, 'cancelled_before_send', request=identifier, task=task,
                          reason='insufficient_response_budget', remaining_seconds=remaining,
                          measured_response_reserve_seconds=server.upstream_response_reserve)
                    return self.reply(402, {'error': {'code': 'insufficient_response_budget'}})
                with server.condition:
                    if server.queue[0] == identifier and server.active is None and not server.unknown:
                        # No time-consuming action between these final checks and send.
                        if disconnected(self.connection) or time.monotonic() >= deadline or not guard.allowed():
                            continue
                        server.active = identifier
                        server.queue.pop(0)
                        break
                time.sleep(.1)
            sent = time.monotonic()
            event(guard.directory, 'request_sent', request=identifier, task=task, queue_seconds=sent-queued,
                  response_deadline_mono=deadline,certainty='admitted_before_final_forward_check')
            outcome, done = {}, threading.Event()

            def forward():
                try:
                    # No generation retries. Wall deadline is also enforced below.
                    transport = httpx.HTTPTransport(retries=0)
                    with httpx.Client(transport=transport, trust_env=False, follow_redirects=False) as client:
                        # Recheck after client setup and all potentially slow audit I/O.
                        remaining = deadline-time.monotonic()
                        if not guard.allowed() or remaining<=0 or remaining<server.upstream_response_reserve:
                            reason=guard.reason or ('response_deadline' if remaining<=0 else 'insufficient_response_budget')
                            outcome.update(not_sent=True,status=402,content=json.dumps({'error':{'code':reason}}).encode())
                            event(guard.directory,'cancelled_before_send',request=identifier,task=task,reason=reason)
                            return
                        if server.accounting and not server.accounting.begin(identifier, body, raw):
                            guard.close(server.accounting.state['closed'])
                            outcome.update(not_sent=True,status=402,content=json.dumps({'error':{'code':guard.reason}}).encode())
                            return
                        # Durable admission can be slow. No blocking accounting
                        # or refresh between this last clock check and post.
                        if disconnected(self.connection) or time.monotonic() >= deadline or guard.closed:
                            if server.accounting:
                                server.accounting.cancel_before_send(identifier)
                            outcome.update(not_sent=True,status=402,content=json.dumps({'error':{'code':guard.reason or 'response_deadline'}}).encode())
                            return
                        outcome['sent_mono']=time.monotonic()
                        response = client.post(server.upstream + '/chat/completions', content=raw,
                            headers={'Authorization': 'Bearer ' + server.key, 'Content-Type': 'application/json'},
                            timeout=httpx.Timeout(max(.01, deadline-time.monotonic()), connect=min(15, max(.01, deadline-time.monotonic()))))
                        outcome['upstream_received_mono']=time.monotonic()
                        with server.condition:
                            server.upstream_response_reserve=max(server.upstream_response_reserve,
                                1.25*(outcome['upstream_received_mono']-outcome['sent_mono']))
                        saved=server.store.put(response.content,name='qwen_response.txt',task=task,identity=identifier)
                        headers={name:response.headers[name] for name in ('x-request-id','request-id','retry-after') if name in response.headers}
                        if server.response_decoder is not None and response.status_code == 200:
                            parsed = server.response_decoder(response)
                        else:
                            try:parsed=response.json()
                            except ValueError:parsed={}
                        if server.accounting and not server.accounting.complete(identifier, response.status_code, parsed.get('usage')):
                            guard.close(server.accounting.state['closed'])
                        event(guard.directory,'response_saved',task=task,request=identifier,artifact=saved['path'],
                            response_sha256=saved['sha256'],response_bytes=saved['bytes'],status=response.status_code,
                            diagnostic_headers=headers,provider_response_id=parsed.get('id'),
                            provider_id_missing_reason=None if parsed.get('id') else 'provider_did_not_return_id',
                            usage=parsed.get('usage'),usage_missing_reason=None if parsed.get('usage') else 'provider_did_not_return_usage')
                        outcome.update(status=response.status_code, content=response.content,
                            content_type=response.headers.get('content-type', 'application/json'), usage=parsed.get('usage', {}))
                        if server.unknown:
                            event(guard.directory, 'late_response_after_unknown', request=identifier, task=task,
                                  status=response.status_code, upstream_seconds=time.monotonic()-sent)
                except Exception as exc:
                    import traceback
                    diagnostic=server.store.put(traceback.format_exc(),name='upstream_traceback.txt',task=task,identity=identifier)
                    event(guard.directory,'upstream_exception',task=task,request=identifier,artifact=diagnostic['path'],error_type=type(exc).__name__)
                    outcome['unknown_error'] = type(exc).__name__
                finally:
                    done.set()

            threading.Thread(target=forward, daemon=True).start()
            client_gone = False
            while not done.wait(.1):
                client_gone |= disconnected(self.connection)
                if time.monotonic() >= deadline:
                    server.unknown = True
                    guard.close('upstream_outcome_unknown_at_deadline')
                    event(guard.directory, 'request_unknown', request=identifier, task=task, client_disconnected=client_gone)
                    return self.reply(402, {'error': {'code': guard.reason}})
            if outcome.get('not_sent'):
                return self.reply(outcome['status'],outcome['content'])
            sent=outcome.get('sent_mono',sent)
            received = outcome.get('upstream_received_mono',time.monotonic())
            client_gone |= disconnected(self.connection)
            if 'unknown_error' in outcome:
                server.unknown = True
                guard.close('upstream_outcome_unknown')
                event(guard.directory, 'request_unknown', request=identifier, task=task, error=outcome['unknown_error'])
                return self.reply(402, {'error': {'code': guard.reason}})
            usage = outcome.get('usage', {})
            event(guard.directory, 'request_response', request=identifier, task=task, status=outcome['status'],
                  upstream_seconds=received-sent, queue_seconds=sent-queued, client_disconnected=client_gone,
                  late=client_gone or received >= deadline, usage=usage)
            self.reply(outcome['status'], outcome['content'], content_type=outcome.get('content_type', 'application/json'))
            event(guard.directory,'response_delivery',task=task,request=identifier,delivered=not client_gone,
                  certainty='socket_write_without_peer_ack' if not client_gone else 'client_disconnected',
                  diagnostic_and_forwarding_delay_seconds=time.monotonic()-received)
            return
        finally:
            with server.condition:
                if identifier in server.queue:
                    server.queue.remove(identifier)
                if server.active == identifier and not server.unknown:
                    server.active = None
                server.condition.notify_all()


def gateway_main(directory, port):
    directory = Path(directory)
    config = read(directory / 'CONFIG.json')
    load_dotenv(ROOT / '.env', override=False)
    key = limits.load_key(ROOT / '.env')
    before = read(directory / 'api_limits_before.json')
    fetch = lambda: limits.snapshot(limits.fetch_info(key), key, 'monitor')
    guard = Guard(before, fetch, directory, budget_cap=config.get('api_budget_cap',180))
    server = Gateway(('0.0.0.0', port), guard, config['jobs'], os.environ['OPENAI_BASE_URL'].rstrip('/'), key,
                     client_key=os.environ['DIAGNOSTIC_GATEWAY_TOKEN'])

    def pulse():
        while True:
            write(directory / 'gateway_heartbeat.json', {'mono': time.monotonic(), 'utc': utc(), 'closed': guard.closed})
            time.sleep(1)

    def account():
        last = 0
        while True:
            guard.allowed()
            if time.monotonic() - last >= 30:
                guard.refresh(force=True)
                last = time.monotonic()
            time.sleep(1)
    threading.Thread(target=pulse, daemon=True).start()
    threading.Thread(target=account, daemon=True).start()
    write(directory / 'gateway_ready.json', {'pid': os.getpid(), 'port': port})
    server.serve_forever()


def infrastructure_gate_closure(reason):
    # Model/budget admission closes permanently, but local control/refit may
    # consume its already reserved time. Diagnostic loss still stops all work.
    return reason not in {'budget_threshold_or_blocked', 'upstream_outcome_unknown',
        'upstream_outcome_unknown_at_deadline', 'api_period_admission_closed',
        'previous_request_billing_unknown', 'period_changed', 'incomparable_budget_counters',
        'account_unavailable', 'shared_300_admission_threshold', 'development_request_size_limit',
        'concurrent_or_duplicate_paid_admission', 'billing_request_identity_mismatch',
        'response_billing_unknown', 'observed_charge_exceeds_safety_margin',
        'request_still_inflight_at_release'}


def watchdog_main(directory, *, poll=.25, stale=15., grace=60., free_disk=None):
    directory = Path(directory)
    failure = None
    safe_event(directory, 'watchdog_started', pid=os.getpid(), stale=stale, grace=grace)
    write(directory / 'watchdog_ready.json', {'pid': os.getpid()})
    while not (directory / 'WATCHDOG_DONE').exists():
        config = read(directory / 'CONFIG.json', {'jobs': {}})
        now = time.monotonic()
        failure_reason = None
        for owner in ('observer', 'gateway'):
            heartbeat = read(directory / f'{owner}_heartbeat.json', {})
            if now - heartbeat.get('mono', -1e20) > stale:
                failure_reason = owner + '_heartbeat_expired'
        ready = read(directory / 'gateway_ready.json', {})
        if ready.get('port'):
            try:
                health = httpx.get(f'http://127.0.0.1:{ready["port"]}/health', timeout=1, trust_env=False)
                if health.status_code != 200:
                    failure_reason = 'gateway_health_failed'
                elif (health.json().get('closed') and not (directory/'RUN_FINISHED').exists()
                      and infrastructure_gate_closure(health.json().get('reason'))):
                    failure_reason = 'gateway_closed:'+str(health.json().get('reason'))
            except Exception:
                failure_reason = 'gateway_health_failed'
        if (free_disk() if free_disk else shutil.disk_usage(directory).free) < 2 * GIB:
            failure_reason = 'disk_below_2gib'
            failure = now - grace
        try:failure_reason=emergency_reason(directory) or failure_reason
        except Exception:failure_reason='diagnostic_state_unavailable'
        if (directory/'CLOSE').exists() and not (directory/'RUN_FINISHED').exists():
            closed_reason=read(directory/'CLOSED.json',{}).get('reason','diagnostic_gate_closed')
            if infrastructure_gate_closure(closed_reason):
                failure_reason=failure_reason or closed_reason
        if failure_reason:
            try:(directory / 'CLOSE').touch()
            except OSError:pass
            if failure is None:
                failure = now
                safe_event(directory, 'watchdog_infrastructure_failure', reason=failure_reason)
        stops=[]
        for job in config['jobs'].values():
            if not job.get('started') or job.get('finished') or job.get('watchdog_killed'):
                continue
            margin = min(30., grace * .5)
            reason = 'hard_deadline' if now >= job['deadline_mono']-margin else 'infrastructure_grace' if failure is not None and now-failure >= grace-margin else None
            if reason:
                if (directory / (job['kind'] + '_watchdog_killed.json')).exists():
                    continue
                stops.append((job,reason))
        def stop(item):
            job,reason=item
            end=min(job['deadline_mono'],failure+grace if failure is not None else job['deadline_mono'])
            try:
                result=docker('kill',job['container'],check=False,timeout=max(.1,min(25.,end-time.monotonic())))
                code=result.returncode
            except subprocess.TimeoutExpired:
                code='kill_command_timeout'
            # Capture the stop before auditing so an audit error cannot repeat it.
            stopped=time.monotonic()
            try:write(directory / (job['kind'] + '_watchdog_killed.json'), {'reason': reason, 'mono': stopped, 'result': code,
                'detection_mono':failure,'stop_latency_seconds':stopped-failure if failure is not None else None})
            except OSError:pass
            safe_event(directory, 'watchdog_kill', task=job['kind'], container=job['container'], reason=reason, exit_code=code,
                detection_mono=failure, stop_latency_seconds=stopped-failure if failure is not None else None)
        if stops:
            with ThreadPoolExecutor(max_workers=len(stops)) as pool:list(pool.map(stop,stops))
        write(directory / 'watchdog_heartbeat.json', {'mono': now, 'utc': utc()})
        time.sleep(poll)


def verify_archive(archive):
    manifest = read(archive.with_suffix('.manifest.json'))
    assert digest(archive) == manifest['sha256'], 'archive_sha256'
    with zipfile.ZipFile(archive) as package:
        assert len(package.namelist()) == len(manifest['members'])
        assert set(package.namelist()) == {m['path'] for m in manifest['members']}
        for item in manifest['members']:
            assert '/' not in item['path'] and (item['path'].endswith('.py') or item['path'] == 'requirements.txt')
            content = package.read(item['path'])
            assert len(content) == item['bytes'] and hashlib.sha256(content).hexdigest() == item['sha256']
        assert package.read('requirements.txt') == (ROOT / 'data_for_participants/requirements.txt').read_bytes()
    return manifest


def validate_csv(job):
    directory = Path(job['directory'])
    path = directory / 'submission.csv'
    if not path.exists():
        return {'passed': False, 'reason': 'missing_csv'}
    def rows(name):
        with (directory / name).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)
    columns, output = rows('submission.csv')
    expected, _ = rows('sample_submission.csv')
    _, test = rows('test.csv')
    _, train = rows('train.csv')
    target = 'site_eui' if job['kind'] == 'regression' else 'attack_cat'
    checks = {'schema': columns == expected, 'row_count': len(output) == len(test),
              'ids_order': [r.get('id') for r in output] == [r['id'] for r in test],
              'unique_ids': len({r.get('id') for r in output}) == len(output),
              'no_missing': all(all(v not in ('', None) for v in r.values()) for r in output)}
    try:
        classes = {t[target] for t in train}
        checks['values'] = all(math.isfinite(float(r[target])) for r in output) if target == 'site_eui' else all(r[target] in classes for r in output)
    except (KeyError, ValueError):
        checks['values'] = False
    return {'passed': all(checks.values()), 'checks': checks, 'rows': len(output), 'sha256': digest(path)}


def export_job(job, directory):
    work = Path(job['directory']) / '.agent_work/diagnostic'
    report = read(work / 'run_report.json', {})
    meta = metadata(job['directory'], job.get('container'))
    snapshot = directory / job['kind'] / 'analysis'
    snapshot.mkdir(parents=True, exist_ok=True)
    journal = work / 'journal.sqlite'
    experiments = []
    if journal.exists():
        with sqlite3.connect('file:' + str(journal) + '?mode=ro', uri=True) as source:
            with sqlite3.connect(snapshot / 'journal.sqlite') as target:
                source.backup(target)
            experiments = [{'spec': json.loads(a), 'result': json.loads(b) if b else None} for a,b in source.execute('SELECT spec,result FROM experiments ORDER BY rowid')]
            with (snapshot / 'actions.jsonl').open('w') as stream:
                for seq, timestamp, kind, payload in source.execute('SELECT seq,time,kind,payload FROM events ORDER BY seq'):
                    stream.write(json.dumps({'seq': seq, 'epoch': timestamp, 'kind': kind, 'payload': json.loads(payload)}, ensure_ascii=False) + '\n')
            write(snapshot / 'operations.json', [json.loads(row[0]) for row in source.execute('SELECT record FROM operations')])
    write(snapshot / 'metadata.json', meta)
    write(snapshot / 'experiments.json', experiments)
    with (snapshot / 'experiments.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id','engine','model','status','scope','protocol','score','seconds','error','hypothesis'])
        writer.writeheader()
        for row in experiments:
            s, r = row['spec'], row['result'] or {}
            writer.writerow({'id': s['experiment_id'], 'engine': s['pipeline']['engine'], 'model': s['pipeline']['model'],
                'status': r.get('status'), 'scope': r.get('scope'), 'protocol': r.get('validation_hash'), 'score': r.get('score'),
                'seconds': r.get('seconds'), 'error': r.get('error'), 'hypothesis': json.dumps(s.get('hypothesis'), ensure_ascii=False)})
    result = {'kind': job['kind'], 'docker': job.get('docker_state'), 'report': report, 'csv': validate_csv(job),
              'watchdog': read(directory / (job['kind'] + '_watchdog_killed.json'))}
    publication = report.get('publication') or meta.get('publication') or {}
    result['technical_passed'] = (job.get('docker_state', {}).get('ExitCode') == 0 and report.get('state') == 'completed'
        and report.get('control_status') == report.get('refit_status') == 'completed'
        and publication.get('scope') == 'full_train_refit' and result['csv']['passed']
        and publication.get('csv_sha256') == result['csv'].get('sha256') and not result['watchdog'])
    csv_path = Path(job['directory']) / 'submission.csv'
    if result['csv']['passed']:
        shutil.copyfile(csv_path, directory / (job['kind'] + '_submission.csv'))
    result['baseline_id'] = meta.get('baseline_id')
    write(snapshot / 'RESULT.json', result)
    return result


def diagnostic_report(directory):
    store=Store(directory)
    records=list(store.records())
    agent=[]
    for row in records:
        if row['source']=='agent' and row.get('artifact') and row['kind']!='candidate_artifact':
            body=read_compressed_json(store.root/row['artifact'])
            if body:
                agent.append((row,body.get('payload',{})))
    operations={}
    for row,payload in agent:
        if row['kind']=='operation_record':
            op=payload['record'];operations[(row['task_id'],op['operation_id'])]=op
    for row in records:
        if row['kind']=='operation_observed':
            payload=read_compressed_json(store.root/row['artifact'])
            if payload:
                op=payload['record'];operations[(row['task_id'],op['operation_id'])]=op
    parsed={}
    for row,payload in agent:
        if row['kind']=='qwen_parse_outcome':parsed[(row['task_id'],payload['call_id'])]=payload
    registry=[]
    totals={}
    for row in records:
        if row['kind']!='request_link':continue
        p=row['payload'];task=row['task_id'];identifier=p['request']
        response=next((x for x in records if x['kind']=='response_saved' and x['payload'].get('request')==identifier),None)
        sent=next((x for x in records if x['kind']=='request_sent' and x['payload'].get('request')==identifier),None)
        returned=next((x for x in records if x['kind']=='request_response' and x['payload'].get('request')==identifier),None)
        terminal=next((x for x in reversed(records) if x['kind'] in {'request_unknown','request_rejected','cancelled_before_send'} and x['payload'].get('request')==identifier),None)
        parse=parsed.get((task,p.get('call_id')), {})
        command=parse.get('command',{})
        operation=operations.get((task,command.get('operation_id')), {})
        usage=response['payload'].get('usage') if response else None
        outcome='completed' if response else terminal['kind'] if terminal else 'interrupted'
        entry={**p,'task':task,'gateway_id':identifier,'provider_response_id':response['payload'].get('provider_response_id') if response else None,
            'provider_id_missing_reason':None if response and response['payload'].get('provider_response_id') else 'not_returned_or_no_provider_response',
            'response_artifact':response.get('artifact') if response else None,'operation_id':command.get('operation_id'),
            'experiment_id':operation.get('experiment_id'),'operation_missing_reason':None if command.get('operation_id') else 'no_accepted_command',
            'experiment_missing_reason':None if operation.get('experiment_id') else 'operation_did_not_create_experiment',
            'parse_outcome':parse or None,'outcome':outcome,'usage':usage,'usage_missing_reason':None if usage else 'not_returned_or_no_provider_response',
            'queue_seconds':returned['payload']['queue_seconds'] if returned else None,
            'upstream_seconds':returned['payload']['upstream_seconds'] if returned else None,
            'sent':sent is not None and not (terminal and terminal['kind']=='cancelled_before_send')}
        registry.append(entry)
        total=totals.setdefault(task,{'known_usage_requests':0,'unknown_usage_requests':0,'tokens_known':{},'queue_seconds':0.,'upstream_seconds':0.})
        total['queue_seconds']+=entry['queue_seconds'] or 0;total['upstream_seconds']+=entry['upstream_seconds'] or 0
        total['known_usage_requests' if usage else 'unknown_usage_requests']+=1
        if usage:
            for key in ('prompt_tokens','completion_tokens','total_tokens'):
                if isinstance(usage.get(key),int):total['tokens_known'][key]=total['tokens_known'].get(key,0)+usage[key]
            reasoning=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
            if isinstance(reasoning,int):total['tokens_known']['reasoning_tokens']=total['tokens_known'].get('reasoning_tokens',0)+reasoning
    write(directory/'REQUEST_REGISTRY.json',registry)
    write(directory/'TOKEN_TIMING_TOTALS.json',totals)
    # Each candidate snapshot can be captured repeatedly. Read the final version
    # per task and identity, then account for each trial exactly once.
    histories={}
    for row in records:
        if row['kind']=='candidate_artifact' and row['payload'].get('name')=='diagnostic_trials.jsonl':
            histories[(row['task_id'],row['payload'].get('identity'))]=row
    trials=[]
    for (task,identity),row in histories.items():
        content=gzip.decompress((store.root/row['artifact']).read_bytes()).decode()
        starts={};finished={}
        for line in content.splitlines():
            item=json.loads(line)
            if item['kind']=='trial_started':starts[item['trial_id']]=item
            elif item['kind']=='trial_finished':finished[item['trial_id']]=item
        for identifier,start in starts.items():
            finish=finished.get(identifier)
            trials.append({'task':task,'identity':identity,'trial_id':identifier,'start':start,'finish':finish,
                'outcome':finish['status'] if finish else 'interrupted','missing_reason':None if finish else 'worker_ended_without_trial_terminal_record'})
    write(directory/'TRIALS.json',trials)
    pending_operations=[{'task':task,**op} for (task,_),op in operations.items() if op.get('status') in {'pending','running'}]
    linked={(r['task'],r['call_id']) for r in registry}
    orphan_exchanges=[{'task':row['task_id'],'call_id':payload.get('call_id'),'artifact':row['artifact'],'reason':'agent_exchange_without_gateway_receipt'} for row,payload in agent if row['kind']=='qwen_exchange_start' and (row['task_id'],payload.get('call_id')) not in linked]
    complete=not orphan_exchanges and all(item['call_id'] is not None and (item['request_id'] is not None or item.get('request_id_missing_reason')) for item in registry)
    missing=[{'gateway_id':item['gateway_id'],'reason':('missing_call_id' if item['call_id'] is None else item.get('request_id_missing_reason') or 'missing_request_id')} for item in registry if item['call_id'] is None or item['request_id'] is None]
    result={'history':store.verify(),'registry_requests':len(registry),'links_complete':complete,'missing_links':missing,
        'orphan_exchanges':orphan_exchanges,'pending_operations':pending_operations,'trials':len(trials),'timing_scope':'Queue/upstream use host durations; worker durations use container-local clocks. No host/container monotonic subtraction.',
        'financial_completeness':'not_inferred_from_token_usage'}
    write(directory/'DIAGNOSTIC_ACCEPTANCE.json',result)
    return result


def read_compressed_json(path):
    try:return json.loads(gzip.decompress(Path(path).read_bytes()))
    except (OSError,ValueError):return None


def reconcile(directory, key, before, *, interval=30, duration=300, fetch=None):
    fetch = fetch or (lambda: limits.snapshot(limits.fetch_info(key), key, 'after'))
    samples, stable, last, first_after = [], 0, None, None
    end = time.monotonic() + duration
    while True:
        try:
            fresh = fetch()
            fresh['comparison'] = limits.compare(before, fresh)
            name = 'api_limits_after.json' if not samples else f'api_limits_reconcile_{len(samples):02d}.json'
            write(directory / name, fresh)
            if first_after is None:
                first_after = fresh
            same = limits.compare(last, fresh).get('increase') == 0 if last else False
            stable = stable + 1 if same else 1
            last = fresh
            samples.append({'file': name, 'success': True, 'remaining': fresh['remaining_budget'], 'increase': fresh['comparison'].get('increase')})
        except Exception as exc:
            stable = 0
            samples.append({'success': False, 'error': type(exc).__name__, 'utc': utc()})
        if stable >= 3 or time.monotonic() >= end:
            break
        time.sleep(min(interval, max(0, end-time.monotonic())))
    unknown = any(row['kind'] == 'request_unknown' for row in events(directory))
    final_success = samples[-1]['success']
    report = {'samples': samples, 'stable_successful_samples': stable,
        'remaining_confirmed': final_success, 'remaining': last['remaining_budget'] if final_success else None,
        'interval_increase': limits.compare(before, last).get('increase') if final_success else None,
        'additional_since_first_after': limits.compare(first_after, last).get('increase') if final_success and first_after else None,
        'unknown_upstream_outcome': unknown, 'accounting_complete': False,
        'spend_status': 'preliminary', 'reason': 'No confirmed upper bound on provider accounting lag'}
    write(directory / 'RECONCILIATION.json', report)
    return report


def prepare(directory, archive, checks, *, source=None, seconds=10800, cpu=5, memory_gib=5, api_budget=180):
    source_root=source or ROOT/'data_for_participants'
    assert 0<seconds<=10800 and 0<api_budget<=180 and cpu>=1 and memory_gib>=1
    directory.mkdir(parents=True, exist_ok=False)
    checked = read(checks / 'REPORT.json', {})
    assert checked.get('passed') is True and checked.get('archive_sha256') == digest(archive), 'preflight_not_passed'
    assert checked.get('supervisor_sha256') == digest(Path(__file__)), 'supervisor_changed_after_preflight'
    assert checked.get('supervisor_sources') == supervisor_sources(), 'supervisor_dependencies_changed_after_preflight'
    package_manifest = verify_archive(archive)
    shutil.copytree(checks, directory / 'preflight')
    shutil.copyfile(Path(__file__), directory / 'supervisor_source.py')
    info = json.loads(docker('info', '--format', '{{json .}}').stdout)
    assert info['NCPU'] >= max(10,2*cpu) and info['MemTotal'] >= (2*memory_gib+1) * GIB, 'insufficient_linux_capacity'
    clock_before=time.time()
    disk_check = docker('run', '--rm', '--network', 'none', IMAGE, 'python', '-c',
        'import shutil,json,psutil,time;print(json.dumps({"disk":[shutil.disk_usage(p).free for p in ("/","/tmp")],"memory_available":psutil.virtual_memory().available,"utc_epoch":time.time(),"clock_domain":"linux_vm"}))')
    clock_after=time.time()
    capacity = json.loads(disk_check.stdout)
    assert clock_before-1 <= capacity['utc_epoch'] <= clock_after+1,'linux_host_utc_alignment_unconfirmed'
    capacity['host_utc_bounds']=[clock_before,clock_after]
    assert capacity['memory_available'] >= (2*memory_gib+1) * GIB, 'insufficient_available_memory'
    assert min(capacity['disk'] + [shutil.disk_usage(directory).free, shutil.disk_usage('/tmp').free]) >= 15*GIB, 'insufficient_disk'
    jobs = {}
    slug = directory.name[-22:].lower().replace('_', '-')
    for kind, source in [('regression', 'Регрессия'), ('classification', 'Многоклассовая классификация')]:
        task = directory / kind / 'task'
        task.mkdir(parents=True)
        hashes = {}
        for name in ('task.md', 'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv'):
            original = Path(source_root) / source / name
            shutil.copyfile(original, task / name)
            hashes[name] = digest(original)
        with zipfile.ZipFile(archive) as package:
            package.extractall(task)
        jobs[kind] = {'kind': kind, 'directory': str(task), 'container': 'alpha-diag-' + slug + '-' + kind,
                      'input_hashes': hashes, 'started': False, 'finished': False, 'deadline_mono': None}
    write(directory / 'CONFIG.json', {'jobs': jobs,'api_budget_cap':api_budget})
    return {'archive_sha256': digest(archive), 'archive_version':archive.stem, 'base_archive_sha256':SHA,
            'diagnostic_limits':{'artifact_bytes':ARTIFACT_MAX,'segment_bytes':SEGMENT_MAX,'total_bytes':STORE_MAX},
            'archive_members': package_manifest['members'], 'cpu_per_task': cpu,
            'memory_gib_per_task': memory_gib, 'linux_cpus': info['NCPU'], 'linux_memory_bytes': info['MemTotal'],
            'system_memory_reserve_bytes': info['MemTotal'] - 2*memory_gib * GIB, 'capacity': capacity,
            'host_free_bytes': shutil.disk_usage(directory).free, 'image_id': docker('image', 'inspect', IMAGE, '--format', '{{.Id}}').stdout.strip(),
            'duration_seconds': seconds, 'api_budget_cap':api_budget,'source':str(source_root),
            'supervisor_sha256': digest(Path(__file__)), 'supervisor_sources':supervisor_sources(), 'preflight_report_sha256': digest(checks / 'REPORT.json'),
            'response_deadline': 'min(monotonic archive deadline, journal epoch deadline converted to monotonic) minus wire reserve minus 1 second',
            'api_response_budget_policy': {'version':1,'reserve':'1.25 times the longest completed upstream response in this series, including known HTTP failures',
                'cold_start':'no observed latency; ordinary deadline and unknown-outcome protection still apply',
                'insufficient_budget':'reject before sending; do not close a healthy gateway or add a generation retry',
                'scope':'conservative admission estimate, not a guaranteed response-time bound'},
            'watchdog': {'heartbeat_seconds': 5, 'stale_seconds': 15, 'grace_seconds': 60, 'kill_command_reserve_seconds': 30, 'poll_seconds': .25,
                'model_or_budget_closure':'new generations remain closed; local finalization may use the original task deadline',
                'diagnostic_failure':'independent latch/heartbeat/storage failures still enforce infrastructure grace'},
            'jobs': jobs, 'started_utc': utc()}


def run(directory, archive, checks, port, *, source=None, seconds=10800, cpu=5, memory_gib=5, api_budget=180):
    manifest = prepare(directory, archive, checks, source=source,seconds=seconds,cpu=cpu,memory_gib=memory_gib,api_budget=api_budget)
    load_dotenv(ROOT / '.env', override=False)
    key = limits.load_key(ROOT / '.env')
    before = limits.snapshot(limits.fetch_info(key), key, 'before')
    write(directory / 'api_limits_before.json', before)
    if before['remaining_budget'] is None or before['remaining_budget'] <= 0 or before['limits'].get('blocked'):
        write(directory / 'NOT_STARTED.json', {'reason': 'unknown_or_blocked_budget'})
        return 3
    manifest.update(budget=min(api_budget, .30*before['remaining_budget']), key_fingerprint=before['key_fingerprint'])
    manifest['request_stop_threshold'] = .90*manifest['budget']
    write(directory / 'MANIFEST.json', manifest)
    print(json.dumps({'status': 'before', 'remaining': before['remaining_budget'], 'budget': manifest['budget'], 'threshold': manifest['request_stop_threshold'], 'output': str(directory)}), flush=True)
    jobs = manifest['jobs']
    children = []
    gateway_token = secrets.token_urlsafe(32)
    register_secret(key)
    register_secret(gateway_token)
    write(directory / 'observer_heartbeat.json', {'mono': time.monotonic(), 'utc': utc()})
    write(directory / 'gateway_heartbeat.json', {'mono': time.monotonic(), 'utc': utc()})
    try:
        for mode in ('gateway', 'watchdog'):
            log = (directory / (mode + '.log')).open('w')
            command = [sys.executable, str(Path(__file__).resolve()), mode, '--output', str(directory)]
            if mode == 'gateway':
                command += ['--port', str(port)]
            child_env = {**os.environ, 'DIAGNOSTIC_GATEWAY_TOKEN': gateway_token}
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=child_env)
            log.close()
            children.append(process)
            timeout = time.monotonic()+10
            while not (directory / (mode + '_ready.json')).exists():
                if process.poll() is not None or time.monotonic() >= timeout:
                    raise RuntimeError(mode + '_startup_failed')
                time.sleep(.1)
        environment = {k: v for k,v in os.environ.items() if not k.startswith(('OPENAI_', 'AGENT_'))}
        environment['OPENAI_API_KEY'] = gateway_token
        environment['OPENAI_MODEL'] = os.environ.get('OPENAI_MODEL', '').strip() or 'qwen/qwen3.8-27b'
        for kind, job in jobs.items():
            environment['OPENAI_BASE_URL'] = f'http://host.docker.internal:{port}/{kind}/v1'
            environment['AGENT_DIAGNOSTICS_URL'] = f'http://host.docker.internal:{port}/{kind}/diagnostics'
            signal_root=directory/'signals'/kind
            signal_root.mkdir(parents=True,exist_ok=True)
            (signal_root/'HEALTHY').touch();(signal_root/'ARMED').touch()
            environment['AGENT_DIAGNOSTICS_FAILURE_DIR']='/diagnostic-signal'
            command = ['create', '--name', job['container'], '--cpus', str(cpu), '--memory', str(memory_gib)+'g', '--memory-swap', str(memory_gib)+'g',
                '-v', job['directory'] + ':/task', '-v', str(signal_root)+':/diagnostic-signal', '-w', '/task', '--env', 'PYTHONDONTWRITEBYTECODE=1']
            for name in ('OPENAI_API_KEY', 'OPENAI_MODEL', 'OPENAI_BASE_URL', 'AGENT_DIAGNOSTICS_URL', 'AGENT_DIAGNOSTICS_FAILURE_DIR'):
                command += ['--env', name + '=' + environment[name]]
            command += [IMAGE, 'python', 'solution.py', 'task.md', '--cpu', str(cpu), '--memory-mb', str(memory_gib*1024), '--time-limit',str(seconds),'--run-id', 'diagnostic']
            docker(*command)
            job.update(started=True, entry_mono=time.monotonic(), entry_utc=utc())
            job['deadline_mono'] = job['entry_mono'] + seconds
            write(directory / 'CONFIG.json', {'jobs': jobs})
            docker('start', job['container'])
            event(directory, 'agent_started', task=kind, container=job['container'], deadline_mono=job['deadline_mono'])
        last_resources = last_status = last_detail = 0
        last_resource_sample=None
        while not all(job['finished'] for job in jobs.values()):
            now = time.monotonic()
            write(directory / 'observer_heartbeat.json', {'mono': now, 'utc': utc()})
            if children[0].poll() is not None or children[1].poll() is not None:
                (directory / 'CLOSE').touch()
                event(directory, 'infrastructure_process_failure', gateway=children[0].poll(), watchdog=children[1].poll())
                raise RuntimeError('infrastructure_process_failed')
            for job in jobs.values():
                if job['finished']:
                    continue
                state = json.loads(docker('inspect', '--format', '{{json .State}}', job['container']).stdout)
                job['docker_state'] = state
                if not state['Running']:
                    job.update(finished=True, finished_mono=time.monotonic(), finished_utc=utc())
                    log = docker('logs', job['container'], check=False)
                    (directory / job['kind'] / 'container.log').write_text(log.stdout + log.stderr)
                    event(directory, 'agent_finished', task=job['kind'], state=state)
                    Store(directory).enqueue(job['kind'],'agent_finished')
            if now-last_resources >= 10:
                stats = docker('stats', '--no-stream', '--format', '{{json .}}', *[j['container'] for j in jobs.values()], check=False)
                measured=time.monotonic()
                interval=measured-last_resource_sample if last_resource_sample is not None else None
                missed=max(0,int(interval/10)-1) if interval is not None else None
                event(directory, 'resources', measurement_mono=measured,measurement_interval_seconds=interval,
                      target_interval_seconds=10,missed_intervals=missed,
                      missing_reason='first_measurement' if interval is None else 'target_samples_missed' if missed else None,
                      free_bytes=shutil.disk_usage(directory).free,
                      docker_stats=[json.loads(line) for line in stats.stdout.splitlines() if line.startswith('{')],
                      states={k: {'finished':j['finished'], 'metadata': {name: metadata(j['directory'], j.get('container')).get(name) for name in ('state', 'leader_id', 'llm_calls', 'candidate_count', 'active_worker', 'publication')}} for k,j in jobs.items()})
                last_resources = now
                last_resource_sample=measured
            if now-last_status >= 30:
                for job in jobs.values():
                    Store(directory).enqueue(job['kind'],'periodic_30_seconds')
                write(directory / 'STATUS.json', {'utc': utc(), 'jobs': jobs, 'closed': read(directory / 'CLOSED.json')})
                print(json.dumps({'status': 'monitor', 'utc': utc(), 'jobs': {k: {'finished': j['finished'], 'elapsed':round(now-j['entry_mono']), **{name:metadata(j['directory'], j.get('container')).get(name) for name in ('state','leader_id','llm_calls','candidate_count')}} for k,j in jobs.items()}, 'closed':read(directory / 'CLOSED.json')}, ensure_ascii=False), flush=True)
                last_status = now
            if now-last_detail >= 900:
                write(directory / 'summaries' / f'{int(now):010d}.json', {'jobs': jobs, 'metadata': {k: metadata(j['directory'], j.get('container')) for k,j in jobs.items()}})
                last_detail = now
            write(directory / 'CONFIG.json', {'jobs': jobs})
            time.sleep(3)
    except BaseException as exc:
        (directory / 'CLOSE').touch()
        try:
            trace=capture_exception(directory,'observer')
            safe_event(directory, 'supervisor_exception', artifact=trace['path'], error=type(exc).__name__, message=str(exc)[:300])
        except Exception:pass
        for job in jobs.values():
            if job['started'] and not job['finished']:
                docker('kill', job['container'], check=False)
                job['docker_state'] = json.loads(docker('inspect', '--format', '{{json .State}}', job['container']).stdout)
                job['finished'] = True
    finally:
        backup_wait=time.monotonic()+120
        while any(j['status'] in {'queued','running'} for j in Store(directory).jobs()):
            write(directory / 'observer_heartbeat.json', {'mono':time.monotonic(), 'utc':utc()})
            if time.monotonic()>=backup_wait or emergency_reason(directory):
                (directory/'CLOSE').touch();break
            time.sleep(.2)
        if all(job['finished'] for job in jobs.values()) and not any(row['kind']=='supervisor_exception' for row in events(directory)):
            (directory/'RUN_FINISHED').touch()
        (directory / 'CLOSE').touch()
        # An active request is allowed its known deadline; do not assert cancellation.
        if children:
            end = max([j.get('deadline_mono') or 0 for j in jobs.values()] + [time.monotonic()])
            while children[0].poll() is None:
                try:
                    health = httpx.get(f'http://127.0.0.1:{port}/health', timeout=2).json()
                    if not health.get('active') or health.get('unknown'):
                        break
                except Exception:
                    break
                if time.monotonic() >= end:
                    break
                write(directory / 'observer_heartbeat.json', {'mono':time.monotonic(), 'utc':utc()})
                time.sleep(1)
        (directory / 'WATCHDOG_DONE').touch()
        for process in children:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        write(directory / 'CONFIG.json', {'jobs': jobs})
    results = [export_job(job, directory) for job in jobs.values() if job['started']]
    diagnostic_acceptance=diagnostic_report(directory)
    reconciliation = reconcile(directory, key, before)
    trusted={}
    with zipfile.ZipFile(archive) as package:
        syntax=ast.parse(package.read('ml_knowledge_bundle.py'))
        knowledge=next(ast.literal_eval(node.value) for node in syntax.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='MANIFEST' for target in node.targets))
    for job in jobs.values():
        task=Path(job['directory'])
        for member in manifest['archive_members']:trusted[str((task/member['path']).resolve())]=member['sha256']
        for name,specification in knowledge['files'].items():trusted[str((task/'.ml_knowledge_runtime'/knowledge['corpus']/name).resolve())]=specification['sha256']
    secret_scan=scan_secrets(directory,trusted_files=trusted)
    history_check=Store(directory).verify()
    closed_reason=read(directory/'CLOSED.json',{}).get('reason')
    limited=closed_reason not in {None,'normal_agents_finished'}
    report = {'results':results, 'reconciliation': reconciliation, 'closed':read(directory/'CLOSED.json'), 'secret_scan':secret_scan,'diagnostic_history':history_check,'diagnostic_acceptance':diagnostic_acceptance,
        'limited_run':limited,
        'full_run_passed': len(results)==2 and all(r['technical_passed'] for r in results) and not limited and secret_scan['passed'] and history_check['passed'] and diagnostic_acceptance['links_complete'] and not diagnostic_acceptance['pending_operations'],
        'official_quality': 'unknown'}
    write(directory / 'REPORT.json', report)
    lines = ['# Результаты диагностического прогона '+archive.stem, '', f'Начало: {manifest["started_utc"]}.',
             f'Бюджет: {manifest["budget"]:.4f}; порог новых запросов: {manifest["request_stop_threshold"]:.4f}.', '',
             'Два агента делили один канал Qwen. Ожидание очереди и upstream отражены отдельно в events.jsonl.', '']
    for result in results:
        r = result['report']
        lines += [f'## {result["kind"]}', f'Техническое завершение: {result["technical_passed"]}; CSV: {result["csv"]["passed"]}.',
            f'Состояние: {r.get("state")}; контроль: {r.get("control_status")}; переобучение: {r.get("refit_status")}.',
            f'Лидер: {r.get("leader")}; контрольный победитель: {r.get("control_winner")}; baseline: {result.get("baseline_id")}.',
            f'Контрольные оценки: {json.dumps(r.get("control_scores"), ensure_ascii=False)}.',
            f'Вызовы Qwen: {r.get("model_calls")}; кандидаты: {r.get("research_candidates")}; CPU: {r.get("effective_cpu")}; память: {r.get("effective_memory_mb")} МиБ.',
            f'Причина остановки поиска: {r.get("research_stop_reason")}; ошибка: {r.get("failure")}.', '']
    lines += ['## Учёт API', f'Подтверждённый текущий остаток: {reconciliation["remaining"]}; прирост счётчика: {reconciliation["interval_increase"]}.',
              'Расход предварительный: верхняя граница задержки серверного списания неизвестна.',
              'Метрики — локальное свидетельство; ответы скрытого теста и пороги организаторов неизвестны.',
              'Подробные гипотезы, результаты, ошибки и операции сохранены в analysis каждой задачи; исходные журналы сохранены.', '']
    (directory / 'Отчёт.md').write_text('\n'.join(lines))
    print(json.dumps({'status':'finished','report':str(directory/'REPORT.json'),'passed':report['full_run_passed'],'remaining':reconciliation['remaining'],'increase':reconciliation['interval_increase']},ensure_ascii=False),flush=True)
    return 0 if report['full_run_passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run','gateway','watchdog'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, default=ROOT/'dist/solution_v022.zip')
    parser.add_argument('--checks', type=Path)
    parser.add_argument('--port', type=int, default=18842)
    parser.add_argument('--source',type=Path,default=ROOT/'data_for_participants')
    parser.add_argument('--seconds',type=float,default=10800)
    parser.add_argument('--cpu',type=int,default=5)
    parser.add_argument('--memory-gib',type=int,default=5)
    parser.add_argument('--api-budget',type=float,default=180)
    args = parser.parse_args()
    if args.mode == 'gateway':
        gateway_main(args.output, args.port)
    elif args.mode == 'watchdog':
        watchdog_main(args.output)
    else:
        raise SystemExit(run(args.output.resolve(), args.archive.resolve(), args.checks.resolve(), args.port,
            source=args.source.resolve(),seconds=args.seconds,cpu=args.cpu,memory_gib=args.memory_gib,api_budget=args.api_budget))
