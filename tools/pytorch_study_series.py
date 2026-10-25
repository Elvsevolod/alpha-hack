"""Sequential research workers with a measured RSS/deadline watchdog."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from automl_agent.util import atomic_json,file_hash
from automl_agent.executor import terminate_tree

def run(output,tasks,seed=42,candidates=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    for name in tasks:
        destination=output/name
        if destination.exists():raise ValueError('Never overwrite research results: '+str(destination))
        command=[sys.executable,str(ROOT/'tools/pytorch_agent_study.py'),'--task',name,'--output',str(destination),'--seed',str(seed)]
        if candidates:command+=['--candidates',*candidates]
        env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','TMPDIR'}}
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/private/tmp/pytorch-mpl')
        began=time.monotonic();peak=0;stop=None
        with (output/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while process.poll() is None:
                    try:peak=max(peak,psutil.Process(process.pid).memory_info().rss)
                    except psutil.NoSuchProcess:break
                    if peak>4*1024**3:stop='rss_limit'
                    if time.monotonic()-began>1200:stop='worker_deadline'
                    if stop:terminate_tree(process);break
                    time.sleep(.2)
                process.wait()
            finally:
                if process.poll() is None:terminate_tree(process)
        if destination.exists():
            for file in ['pytorch_agent_study.py','torch_research_core.py']:
                (destination/file).write_bytes((ROOT/'tools'/file).read_bytes())
        atomic_json(output/(name+'_watchdog.json'),dict(returncode=process.returncode,stop=stop,peak_rss_bytes=peak,seconds=time.monotonic()-began,cpu_threads=1,rss_limit_bytes=4*1024**3,worker_timeout=1200,api_calls=0))
        print(name,process.returncode,stop,round(time.monotonic()-began,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',nargs='+',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--candidates',nargs='+');a=p.parse_args();run(a.output,a.tasks,a.seed,a.candidates)
