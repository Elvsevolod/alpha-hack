"""Bounded replay of the long rehearsal's SQLite failure path. No API traffic."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import diagnostic_3h as d
from tools.diagnostic_store import Store,scan_secrets,register_secret


def run(output):
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    task=output/'task';task.mkdir();name='alpha-sqlite-replay-'+uuid.uuid4().hex[:8]
    secret='sk-replayOnlySyntheticSecret12345678';register_secret(secret)
    code="""import sqlite3,pathlib,time,json
+p=pathlib.Path('/task/.agent_work/diagnostic');p.mkdir(parents=True)
+c=sqlite3.connect(p/'journal.sqlite');c.execute('PRAGMA journal_mode=WAL');c.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)');c.execute('CREATE TABLE load_records(id INTEGER PRIMARY KEY,record TEXT)')
+c.executemany('INSERT INTO load_records VALUES (?,?)',[(i,json.dumps({'secret':'sk-replayOnlySyntheticSecret12345678','payload':'x'*16000})) for i in range(2000)]);c.commit()
+for i in range(3000):
+ c.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('counter',json.dumps(i)));c.execute('UPDATE load_records SET record=record WHERE id=?',(i%2000,));c.commit();time.sleep(.02)
+assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
+c.close()
+""".replace('\n+','\n')
    job={'directory':str(task),'container':name,'kind':'regression'}
    rows=[]
    try:
        d.docker('run','-d','--name',name,'--network','none','--cpus','.5','--memory','256m','-v',str(task)+':/task',d.IMAGE,'python','-c',code)
        until=time.monotonic()+20
        while not (task/'.agent_work/diagnostic/journal.sqlite').exists():
            assert time.monotonic()<until;time.sleep(.1)
        while 'counter' not in d.metadata(task,name):
            assert time.monotonic()<until;time.sleep(.1)
        previous=-1
        for i in range(6):
            begin=time.monotonic();meta=d.metadata(task,name);counter=meta.get('counter',-1)
            assert counter>=previous;previous=counter
            backups=d.backup_sqlite(output,job,reason='periodic_30_seconds')
            assert backups and max(b['bytes'] for b in backups)>=17690624
            rows.append({'counter':counter,'seconds':time.monotonic()-begin,'backups':backups})
        d.docker('wait',name,timeout=90)
        state=json.loads(d.docker('inspect','--format','{{json .State}}',name).stdout)
        assert state['ExitCode']==0,state
        rows.append({'stopped_backup':d.backup_sqlite(output,job,reason='agent_finished')})
        assert Store(output).verify()['passed'] and scan_secrets(output/'history')['passed']
        report={'passed':True,'organizer_requests':0,'cycles':rows,'exit_code':state['ExitCode'],'supervisor_sources':d.supervisor_sources(),'fix':'live WAL access uses Linux SQLite; host sanitizes closed independent exports only'}
        d.write(output/'REPORT.json',report)
        print(json.dumps({'passed':True,'cycles':6,'exit_code':0}))
    finally:d.docker('rm','-f',name,check=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);run(p.parse_args().output)
