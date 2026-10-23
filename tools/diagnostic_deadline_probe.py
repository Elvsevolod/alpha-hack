"""Offline container deadline test with one upstream and one queued request."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import diagnostic_3h as d
from tools.test_diagnostic_3h import gateway,uuid_short


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    release=threading.Event()
    names=['alpha-deadline-'+uuid_short() for _ in range(2)]
    jobs={k:{'kind':k,'container':n,'started':False,'finished':False,'deadline_mono':time.monotonic()+30} for k,n in zip(('regression','classification'),names)}
    watcher=None
    with gateway(output/'gate',lambda *args:(release.wait(20) and 200 or 200,{})) as (server,guard,calls):
        gate=guard.directory
        d.write(gate/'CONFIG.json',{'jobs':jobs})
        d.write(gate/'gateway_heartbeat.json',{'mono':time.monotonic()+100})
        watcher=subprocess.Popen([sys.executable,str(d.ROOT/'tools/diagnostic_3h.py'),'watchdog','--output',str(gate)],start_new_session=True)
        try:
            end=time.monotonic()+5
            while not (gate/'watchdog_ready.json').exists():assert time.monotonic()<end;time.sleep(.05)
            for kind,job in jobs.items():
                body=json.dumps({'deadline':time.monotonic()+30})
                code='import subprocess,urllib.request;subprocess.Popen(["python","-c","while True: pass"]);urllib.request.urlopen(urllib.request.Request('+repr(f'http://host.docker.internal:{server.server_port}/{kind}/v1/chat/completions')+',data='+repr(body.encode())+',headers={"Content-Type":"application/json"}),timeout=30).read()'
                d.docker('create','--name',job['container'],'--cpus','1','--memory','256m',d.IMAGE,'python','-c',code)
            for job in jobs.values():
                job.update(started=True,deadline_mono=time.monotonic()+10)
                d.write(gate/'CONFIG.json',{'jobs':jobs});d.docker('start',job['container'])
            end=time.monotonic()+4
            while not(server.active and server.queue):assert time.monotonic()<end;time.sleep(.05)
            assert len(calls)==1
            while not all((gate/(k+'_watchdog_killed.json')).exists() for k in jobs):
                assert time.monotonic()<end+10;time.sleep(.1)
            for kind,job in jobs.items():
                killed=d.read(gate/(kind+'_watchdog_killed.json'))
                assert killed['reason']=='hard_deadline' and killed['mono']<=job['deadline_mono']
                assert not json.loads(d.docker('inspect','--format','{{json .State}}',job['container']).stdout)['Running']
            assert len(calls)==1
            result={'passed':True,'active_training_child_stopped':True,'queued_client_stopped':True,'upstream_sends':len(calls),'containers':len(jobs)}
            d.write(output/'REPORT.json',result);print(json.dumps(result),flush=True)
        finally:
            release.set();(gate/'WATCHDOG_DONE').touch()
            if watcher:watcher.wait(timeout=5)
            for name in names:d.docker('rm','-f',name,check=False)


if __name__=='__main__':run(Path(sys.argv[1]).resolve())
