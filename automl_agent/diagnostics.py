"""Optional diagnostic-version telemetry; no model or selection decisions."""
from __future__ import annotations

import base64
import gzip
from functools import wraps
import hashlib
import json
import os
import re
from pathlib import Path
import threading
import time
import traceback
import urllib.request
import uuid

from .util import redact

ARTIFACT_LIMIT = 64 * 1024**2
LOG_SEGMENT = 1024**2
LOG_TOTAL = 256 * 1024**2


class DiagnosticUnavailable(RuntimeError):
    pass


DIAGNOSTIC_PROTOCOL = 2


def fail(reason, **details):
    """Independent latch: remove health first, then try to record the reason."""
    shared = os.environ.get("AGENT_DIAGNOSTICS_FAILURE_DIR")
    if shared:
        root = Path(shared)
        try:
            (root / "HEALTHY").unlink(missing_ok=True)
        except OSError:
            pass
        finally:
            try:
                from .util import atomic_json
                if not (root / "FAILURE.json").exists():
                    atomic_json(root / "FAILURE.json", {"schema_version": 1, "reason": reason,
                        "epoch": time.time(), "pid": os.getpid(), **details})
            except OSError:
                pass


def required_capture(function):
    @wraps(function)
    def capture(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            if not os.environ.get('AGENT_DIAGNOSTICS_URL'):
                raise
            fail('diagnostic_' + function.__name__ + '_failed', error_type=type(error).__name__)
            if isinstance(error, DiagnosticUnavailable):
                raise
            raise DiagnosticUnavailable('Required diagnostic capture failed: ' + function.__name__) from error
    return capture


@required_capture
def emit(kind, **payload):
    url = os.environ.get('AGENT_DIAGNOSTICS_URL')
    if not url:
        return None
    record = json.loads(redact(json.dumps({'schema_version': DIAGNOSTIC_PROTOCOL,
        'event_id': uuid.uuid4().hex, 'kind': kind, 'payload': payload, 'pid': os.getpid(),
        'utc_epoch': time.time(), 'monotonic': time.monotonic(),
        'clock_domain': 'container_pid_' + str(os.getpid())}, ensure_ascii=False, default=str)))
    checksum = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    record['event_sha256'] = checksum
    body = json.dumps(record, ensure_ascii=False).encode()
    request = urllib.request.Request(url, data=body, headers={
        'Authorization': 'Bearer ' + os.environ.get('OPENAI_API_KEY', ''), 'Content-Type': 'application/json'})
    # Retrying telemetry is safe because the durable receipt is keyed by event ID.
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                receipt = json.loads(response.read())
            if (receipt.get('schema_version') != DIAGNOSTIC_PROTOCOL or not receipt.get('complete') or receipt.get('event_id') != record['event_id']
                    or receipt.get('event_sha256') != checksum or receipt.get('event_status') != 'committed'):
                raise ValueError('collector_did_not_confirm_identity')
            return receipt
        except Exception as exc:
            if attempt == 0:
                continue
            fail('diagnostic_receipt_failed', error_type=type(exc).__name__)
            raise DiagnosticUnavailable('Required diagnostic capture unavailable: ' + type(exc).__name__) from exc


@required_capture
def artifact(path, *, identity=None):
    if not os.environ.get('AGENT_DIAGNOSTICS_URL'):
        return
    path = Path(path)
    content = path.read_bytes()
    if path.suffix == '.gz':
        content = gzip.decompress(content)
    if len(content) > ARTIFACT_LIMIT:
        emit('diagnostic_limit', path=path.name, bytes=len(content), limit=ARTIFACT_LIMIT)
        raise DiagnosticUnavailable('diagnostic_artifact_limit')
    content = redact(content.decode('utf-8', errors='replace')).encode()
    checksum = hashlib.sha256(content).hexdigest()
    receipt = emit('artifact', name=path.name, identity=identity or str(path), bytes=len(content),
        sha256=checksum, content_b64=base64.b64encode(gzip.compress(content, mtime=0)).decode())
    if receipt and (receipt.get('sha256') != checksum or receipt.get('bytes') != len(content)):
        fail('diagnostic_artifact_receipt_mismatch')
        raise DiagnosticUnavailable('diagnostic_artifact_receipt_mismatch')


@required_capture
def snapshot(directory):
    """Synchronous verified capture before the caller may clean originals."""
    if not os.environ.get('AGENT_DIAGNOSTICS_URL'):
        return
    directory = Path(directory)
    started = time.monotonic()
    files = []
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.name.endswith(('_X.json', '_y.json')):
            continue
        if path.suffix not in {'.json', '.jsonl', '.py', '.log', '.txt', '.csv', '.gz'}:
            continue
        artifact(path, identity=str(path))
        files.append(str(path.relative_to(directory)))
    emit('snapshot_verified', directory=str(directory), files=files, seconds=time.monotonic()-started)


def trial_event(context, kind, **payload):
    """Worker-local append log, collected while HPO is running and before cleanup."""
    path = Path(context['workdir']) / 'diagnostic_trials.jsonl'
    record = {'schema_version':1,'kind':kind,'epoch':time.time(),'mono':time.monotonic(),
              'clock_domain':'container_pid_'+str(os.getpid()), **payload}
    with path.open('a') as stream:
        stream.write(redact(json.dumps(record, default=str, ensure_ascii=False))+'\n')
        stream.flush()
        os.fsync(stream.fileno())


class RotatingOutput:
    """Drain worker output without a 10 MiB single-file kill or raw secret writes."""
    def __init__(self, pipe, directory):
        self.pipe, self.directory = pipe, Path(directory)
        self.total, self.error, self.segments, self.tail = 0, None, [], b''
        self.thread = threading.Thread(target=self.drain, daemon=True)
        self.thread.start()

    def drain(self):
        index, segment, size = 0, None, 0
        pending = b''
        try:
            while True:
                chunk = self.pipe.read(32768)
                pending += chunk
                if chunk and len(pending) <= 1024:
                    continue
                cut = len(pending)-1024 if chunk else len(pending)
                if chunk:
                    spans=[match.span() for match in re.finditer(rb'\bsk-[A-Za-z0-9_-]{12,}',pending)]
                    for name,value in os.environ.items():
                        if value and len(value)>=8 and any(w in name.upper() for w in ('KEY','TOKEN','SECRET','PASSWORD')):
                            start=pending.find(value.encode())
                            if start>=0:spans.append((start,start+len(value.encode())))
                    for start,end in spans:
                        if start<cut<end:cut=start
                raw,pending=pending[:cut],pending[cut:]
                cleaned = redact(raw.decode('utf-8', errors='replace')).encode()
                self.total += len(cleaned)
                if self.total > LOG_TOTAL:
                    self.error = 'log_limit'
                    return
                if cleaned:
                    if segment is None or size + len(cleaned) > LOG_SEGMENT:
                        if segment:segment.close()
                        path=self.directory/f'worker.log.{index:06d}.gz'
                        segment=gzip.open(path,'wb');self.segments.append(path);index+=1;size=0
                    segment.write(cleaned);segment.flush();size+=len(cleaned)
                    self.tail=(self.tail+cleaned)[-65536:]
                    (self.directory/'worker.log').write_bytes(self.tail)
                if not chunk:break
        except BaseException:
            self.error='log_io_error'
            (self.directory/'diagnostic_output_traceback.txt').write_text(redact(traceback.format_exc()))
        finally:
            if segment:segment.close()

    def finish(self):
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.error='log_drain_incomplete'
        return self.error
