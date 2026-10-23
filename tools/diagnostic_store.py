"""Durable bounded diagnostic history, kept outside agent containers."""
import base64
from contextlib import contextmanager, closing
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
import time

ARTIFACT_MAX = 64 * 1024**2
SEGMENT_MAX = 1024**2
STORE_MAX = 4 * 1024**3
_secrets = set()
_thread_lock = threading.RLock()


def register_secret(value):
    if value:
        _secrets.add(value)


def scrub(content):
    if isinstance(content, str):content=content.encode()
    for name,value in os.environ.items():
        if value and len(value)>=8 and any(w in name.upper() for w in ('KEY','TOKEN','SECRET','PASSWORD')):
            _secrets.add(value)
    for secret in sorted(_secrets,key=len,reverse=True):
        content=content.replace(secret.encode(),b'[REDACTED]')
    content=re.sub(rb'\bsk-[A-Za-z0-9_-]{12,}',b'[REDACTED]',content)
    content=re.sub(rb'(?i)(?:authorization|cookie|set-cookie)\s*[:=]\s*[^\r\n"}]*',b'[REDACTED_HEADER]',content)
    return content


def checksum(content):return hashlib.sha256(content).hexdigest()


def sync_directory(path):
    if hasattr(os,'O_DIRECTORY'):
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)


def atomic(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.pending-',delete=False) as stream:
        stream.write(content);stream.flush();os.fsync(stream.fileno());name=stream.name
    os.replace(name,path)
    sync_directory(path.parent)


class Store:
    """V2 transactional ledger; content-addressed artifacts stay outside SQLite."""
    def __init__(self,directory,*,artifact_max=ARTIFACT_MAX,segment_max=SEGMENT_MAX,total_max=STORE_MAX):
        self.directory=Path(directory)
        self.root=self.directory/'history';self.root.mkdir(parents=True,exist_ok=True)
        self.artifact_max,self.segment_max,self.total_max=artifact_max,segment_max,total_max
        self.ledger=self.root/'ledger.sqlite'
        self.local=threading.local()
        pointer=self.root/'INDEX.json'
        self.legacy=pointer.exists() and not self.ledger.exists() and 'ledger' not in json.loads(pointer.read_text())
        if not self.legacy and not pointer.exists():
            with self.locked() as db:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value INTEGER);
                    INSERT OR IGNORE INTO state VALUES('seq',0),('bytes',0);
                    CREATE TABLE IF NOT EXISTS segments(path TEXT PRIMARY KEY,record TEXT);
                    CREATE TABLE IF NOT EXISTS artifacts(sha TEXT PRIMARY KEY,record TEXT);
                    CREATE TABLE IF NOT EXISTS receipts(task TEXT,event_id TEXT,sha TEXT,receipt TEXT,PRIMARY KEY(task,event_id));
                    CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,task TEXT,reason TEXT,status TEXT,
                        created REAL,deadline REAL,started REAL,ended REAL,result TEXT);
                """)
            atomic(pointer,json.dumps({'schema_version':2,'ledger':'ledger.sqlite'}).encode())

    @contextmanager
    def locked(self):
        active=getattr(self.local,'db',None)
        if active is not None:
            yield active
            return
        with _thread_lock, (self.root/'LOCK').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                with closing(sqlite3.connect(self.ledger,timeout=10)) as db:
                    with db:
                        db.execute('PRAGMA synchronous=FULL')
                        db.execute('BEGIN IMMEDIATE')
                        self.local.db=db
                        try:yield db
                        finally:self.local.db=None
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    def index(self):
        if self.legacy:return json.loads((self.root/'INDEX.json').read_text())
        with self.locked() as db:
            state=dict(db.execute('SELECT key,value FROM state'))
            return {'schema_version':2,**state,'complete':True,
                'segments':[json.loads(r[0]) for r in db.execute('SELECT record FROM segments ORDER BY path')],
                'artifacts':{sha:json.loads(raw) for sha,raw in db.execute('SELECT sha,record FROM artifacts')}}

    def limit(self,reason):
        # The independent latch precedes every fallible audit write.
        (self.directory/'CLOSE').touch()
        atomic(self.directory/'DIAGNOSTIC_LIMIT.json',json.dumps({'reason':reason,'utc':datetime.now(timezone.utc).isoformat()}).encode())
        raise ValueError('diagnostic_storage_limit:'+reason)

    def _capacity(self,db,extra):
        used=db.execute("SELECT value FROM state WHERE key='bytes'").fetchone()[0]
        if used+extra>self.total_max:self.limit('total_bytes')
        db.execute("UPDATE state SET value=value+? WHERE key='bytes'",(extra,))

    def put(self,content,*,name,task=None,identity=None,binary=False):
        if self.legacy:raise ValueError('legacy_history_read_only')
        content=content if binary else scrub(content)
        if len(content)>self.artifact_max:self.limit('artifact_bytes')
        sha=checksum(content)
        with self.locked() as db:
            old=db.execute('SELECT record FROM artifacts WHERE sha=?',(sha,)).fetchone()
            if old:return json.loads(old[0])
            packed=gzip.compress(content,mtime=0)
            self._capacity(db,len(packed))
            sealed='control' in str(identity or name).lower()
            relative=('sealed_control' if sealed else 'artifacts')+'/'+sha+'.gz'
            atomic(self.root/relative,packed)
            restored=gzip.decompress((self.root/relative).read_bytes())
            if checksum(restored)!=sha or len(restored)!=len(content):raise ValueError('artifact_verify_failed')
            item={'path':relative,'bytes':len(content),'stored_bytes':len(packed),'sha256':sha,
                'stored_sha256':checksum(packed),'schema_version':2,'name':name,'task':task,'identity':identity,'complete':True,'sealed_control':sealed}
            db.execute('INSERT INTO artifacts VALUES(?,?)',(sha,json.dumps(item)))
            return item

    def append(self,kind,*,task=None,source='host',phase=None,artifact=None,**payload):
        if self.legacy:raise ValueError('legacy_history_read_only')
        started=time.monotonic()
        with self.locked() as db:
            seq=db.execute("SELECT value FROM state WHERE key='seq'").fetchone()[0]+1
            try:manifest=json.loads((self.directory/'MANIFEST.json').read_text())
            except (OSError,ValueError):manifest={}
            record={'schema_version':2,'seq':seq,'series_id':self.directory.name,'task_id':task,
                'archive_version':manifest.get('archive_version','preflight'),'archive_sha256':manifest.get('archive_sha256'),
                'utc':datetime.now(timezone.utc).isoformat(),'mono':started,'pid':os.getpid(),
                'clock_domain':'host_pid_'+str(os.getpid()),'source':source,'kind':kind,'phase':phase,
                'artifact':artifact,'payload':payload}
            raw=scrub(json.dumps(record,ensure_ascii=False,default=str,allow_nan=False).encode())+b'\n'
            if len(raw)>self.segment_max:raise ValueError('event_exceeds_segment_limit')
            packed=gzip.compress(raw,mtime=0);self._capacity(db,len(packed))
            last=db.execute('SELECT record FROM segments ORDER BY path DESC LIMIT 1').fetchone()
            segment=json.loads(last[0]) if last else None
            if segment is None or segment['uncompressed_bytes']+len(raw)>self.segment_max:
                count=db.execute('SELECT count(*) FROM segments').fetchone()[0]
                segment={'path':f'events/{count:06d}.jsonl.gz','bytes':0,'uncompressed_bytes':0,'first_seq':seq,'last_seq':seq,'schema_version':2,'complete':True}
            path=self.root/segment['path'];path.parent.mkdir(exist_ok=True)
            # Roll back any uncommitted tail from an interrupted transaction.
            with path.open('r+b' if path.exists() else 'w+b') as stream:
                stream.truncate(segment['bytes']);stream.seek(segment['bytes'])
                stream.write(packed);stream.flush();os.fsync(stream.fileno())
            sync_directory(path.parent)
            segment.update(bytes=segment['bytes']+len(packed),uncompressed_bytes=segment['uncompressed_bytes']+len(raw),last_seq=seq)
            segment['sha256']=checksum(path.read_bytes())
            db.execute('INSERT OR REPLACE INTO segments VALUES(?,?)',(segment['path'],json.dumps(segment)))
            db.execute("UPDATE state SET value=? WHERE key='seq'",(seq,))
        return {'complete':True,'seq':seq,'write_seconds':time.monotonic()-started}

    def _snapshot(self):
        # Only copy committed ranges while locked; hashing/decompression is offline.
        with self.locked() if not self.legacy else _thread_lock:
            index=self.index()
            chunks=[(segment,(self.root/segment['path']).read_bytes()[:segment['bytes']]) for segment in index['segments']]
        return index,chunks

    def records(self):
        index,chunks=self._snapshot()
        for segment,packed in chunks:
            if len(packed)!=segment['bytes'] or checksum(packed)!=segment['sha256']:
                raise ValueError('corrupt_event_snapshot:'+segment['path'])
            for line in gzip.decompress(packed).splitlines():yield json.loads(line)

    def verify(self):
        index,chunks=self._snapshot();records=[]
        for segment,packed in chunks:
            if (self.root/segment['path']).stat().st_size!=segment['bytes']:
                raise ValueError('uncommitted_or_corrupt_event_tail:'+segment['path'])
            if checksum(packed)!=segment['sha256']:raise ValueError('corrupt_event_snapshot')
            records.extend(json.loads(line) for line in gzip.decompress(packed).splitlines())
        assert [r['seq'] for r in records]==list(range(1,index['seq']+1))
        for item in index['artifacts'].values():
            packed=(self.root/item['path']).read_bytes();content=gzip.decompress(packed)
            assert checksum(packed)==item['stored_sha256'] and len(content)==item['bytes'] and checksum(content)==item['sha256']
        pending=[str(p.relative_to(self.root)) for p in self.root.rglob('.pending-*')]
        return {'passed':not pending,'records':len(records),'artifacts':len(index['artifacts']),'bytes':index['bytes'],'incomplete_files':pending}

    def _enqueue(self,db,task,reason,job_id,deadline_seconds=120,maximum=8):
        if db.execute('SELECT 1 FROM jobs WHERE id=?',(job_id,)).fetchone():return job_id
        if db.execute("SELECT count(*) FROM jobs WHERE task=? AND status IN ('queued','running')",(task,)).fetchone()[0]>=maximum:
            raise ValueError('backup_queue_full')
        now=time.time()
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)',(job_id,task,reason,'queued',now,now+deadline_seconds,None,None,None))
        return job_id

    def enqueue(self,task,reason,*,maximum=8,deadline_seconds=120):
        with self.locked() as db:
            if reason.startswith('periodic'):
                old=db.execute("SELECT id FROM jobs WHERE task=? AND reason=? AND status IN ('queued','running')",(task,reason)).fetchone()
                if old:return old[0]
            return self._enqueue(db,task,reason,checksum(os.urandom(32)),deadline_seconds,maximum)

    def claim(self,task):
        with self.locked() as db:
            row=db.execute("SELECT id,reason,deadline FROM jobs WHERE task=? AND status='queued' ORDER BY created LIMIT 1",(task,)).fetchone()
            if row:
                db.execute("UPDATE jobs SET status='running',started=? WHERE id=?",(time.time(),row[0]))
                return {'id':row[0],'reason':row[1],'deadline':row[2]}

    def finish_job(self,job_id,status,result):
        with self.locked() as db:
            db.execute('UPDATE jobs SET status=?,ended=?,result=? WHERE id=?',(status,time.time(),json.dumps(result),job_id))

    def jobs(self):
        with self.locked() as db:
            names=[r[1] for r in db.execute('PRAGMA table_info(jobs)')]
            return [dict(zip(names,row)) for row in db.execute('SELECT * FROM jobs ORDER BY created')]

    def receive(self,task,body):
        raw=scrub(json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
        supplied=body.get('event_sha256')
        unsigned={k:v for k,v in body.items() if k!='event_sha256'}
        sha=checksum(json.dumps(unsigned,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
        if supplied and supplied!=sha:raise ValueError('event_hash_mismatch')
        event_id=body.get('event_id') or sha
        kind=body['kind'];payload=dict(body.get('payload',{}))
        with self.locked() as db:
            old=db.execute('SELECT sha,receipt FROM receipts WHERE task=? AND event_id=?',(task,event_id)).fetchone()
            if old:
                if old[0]!=sha:raise ValueError('event_identity_conflict')
                return {**json.loads(old[1]),'duplicate':True}
            if kind=='artifact':
                content=gzip.decompress(base64.b64decode(payload.pop('content_b64')))
                if len(content)!=payload['bytes'] or checksum(content)!=payload['sha256']:raise ValueError('artifact_content_mismatch')
                item=self.put(content,name=payload['name'],task=task,identity=payload.get('identity'))
                receipt={'sha256':item['sha256'],'bytes':item['bytes']}
                self.append('candidate_artifact',task=task,source='agent',artifact=item['path'],event_id=event_id,**payload)
            else:
                item=self.put(raw,name=kind+'.json',task=task,identity=payload.get('directory'))
                self.append(kind,task=task,source='agent',artifact=item['path'],event_id=event_id,agent_pid=body.get('pid'),
                    agent_clock_domain=body.get('clock_domain'),agent_mono=body.get('monotonic'))
                receipt={}
            backup=None
            if kind=='experiment_record' and payload.get('result'):
                backup=self._enqueue(db,task,'experiment_finished','event:'+task+':'+event_id)
            receipt.update(schema_version=2,complete=True,event_status='committed',event_id=event_id,event_sha256=sha,
                backup_job_id=backup,backup_status='queued' if backup else 'not_required',duplicate=False)
            db.execute('INSERT INTO receipts VALUES(?,?,?,?)',(task,event_id,sha,json.dumps(receipt)))
            return receipt


def scan_secrets(directory,*,trusted_files=None):
    """No size exemption; also decompress every committed gzip artifact."""
    hits=[];verified_patterns=[];trusted_files=trusted_files or {}
    for path in Path(directory).rglob('*'):
        if not path.is_file():continue
        # Stream all ordinary files, keeping enough overlap for exact secrets.
        overlap=max([len(s.encode()) for s in _secrets]+[512])
        carry=b''
        exact=False;pattern=False;file_hash=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024**2),b''):
                file_hash.update(chunk)
                content=carry+chunk
                exact |= any(secret.encode() in content for secret in _secrets)
                pattern |= bool(re.search(rb'\bsk-[A-Za-z0-9_-]{12,}',content))
                carry=content[-overlap:]
        if exact:hits.append(str(path))
        elif pattern:
            expected=trusted_files.get(str(path.resolve()))
            if expected and expected==file_hash.hexdigest():verified_patterns.append({'path':str(path),'sha256':expected,'reason':'unchanged_verified_delivery_or_public_reference; exact_secret_scan_still_applied'})
            else:hits.append(str(path))
        if path.suffix=='.gz':
            try:
                with gzip.open(path,'rb') as stream:
                    carry=b''
                    for chunk in iter(lambda:stream.read(1024**2),b''):
                        content=carry+chunk
                        if any(secret.encode() in content for secret in _secrets) or re.search(rb'\bsk-[A-Za-z0-9_-]{12,}',content):
                            hits.append(str(path)+':decompressed');break
                        carry=content[-overlap:]
            except (OSError,EOFError):hits.append(str(path)+':invalid_compression')
    return {'passed':not hits,'paths':hits,'verified_immutable_pattern_matches':verified_patterns}
