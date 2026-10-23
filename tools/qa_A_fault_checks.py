"""Independent emergency latch with live heartbeats and broken audit writes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import diagnostic_3h as d


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    name='alpha-qa-A-signal-'+uuid.uuid4().hex[:8]
    signal=output/'signals'/'regression';signal.mkdir(parents=True)
    (signal/'HEALTHY').touch();(signal/'ARMED').touch()
    job={'kind':'regression','container':name,'started':True,'finished':False,'deadline_mono':time.monotonic()+180}
    d.write(output/'CONFIG.json',{'jobs':{'regression':job}})
    d.docker('run','-d','--name',name,'--network','none','--cpus','1','--memory','128m',d.IMAGE,
        'python','-c','import subprocess,time;subprocess.Popen(["python","-c","while True: pass"]);time.sleep(300)')
    producer=subprocess.Popen([sys.executable,'-c',
        'import sys,time;from pathlib import Path;from tools.diagnostic_3h import write;p=Path(sys.argv[1]);\nwhile True:\n write(p/"observer_heartbeat.json",{"mono":time.monotonic()});write(p/"gateway_heartbeat.json",{"mono":time.monotonic()});time.sleep(.1)',str(output)],cwd=d.ROOT,start_new_session=True)
    code='from pathlib import Path;from tools import diagnostic_3h as d;import sys;d.event=lambda *a,**k:(_ for _ in ()).throw(OSError("synthetic audit failure"));d.watchdog_main(Path(sys.argv[1]))'
    watcher=subprocess.Popen([sys.executable,'-c',code,str(output)],cwd=d.ROOT,start_new_session=True)
    try:
        until=time.monotonic()+10
        while not (output/'watchdog_ready.json').exists():
            assert time.monotonic()<until;time.sleep(.1)
        detected=time.monotonic();(signal/'HEALTHY').unlink()
        until=detected+60
        while not (output/'regression_watchdog_killed.json').exists():
            assert watcher.poll() is None and time.monotonic()<until;time.sleep(.1)
        stopped=time.monotonic();state=json.loads(d.docker('inspect','--format','{{json .State}}',name).stdout)
        assert not state['Running']
        assert not d.docker('top',name,check=False).stdout.strip().startswith('UID')
        assert time.monotonic()-d.read(output/'observer_heartbeat.json')['mono']<1
        assert time.monotonic()-d.read(output/'gateway_heartbeat.json')['mono']<1
        result={'passed':True,'organizer_requests':0,'live_heartbeats':True,'all_audit_events_failed':True,
            'independent_agent_latch':True,'container_and_descendants_stopped':True,
            'actual_stop_latency_seconds':stopped-detected,'watchdog':d.read(output/'regression_watchdog_killed.json'),
            'supervisor_sha256':d.digest(Path(d.__file__))}
        d.write(output/'REPORT.json',result);print(json.dumps(result),flush=True)
    finally:
        (output/'WATCHDOG_DONE').touch()
        watcher.wait(timeout=5);producer.kill();producer.wait();d.docker('rm','-f',name,check=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output.resolve())
