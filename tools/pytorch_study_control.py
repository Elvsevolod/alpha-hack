"""One frozen research-control evaluation; never uses external test labels.

Invoke only with a final control freeze from completed three-seed development.
No recipes/policies are selected after evaluation. Canonical CSVs stay untouched.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import numpy as np
import pandas as pd
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from automl_agent.util import atomic_json,file_hash
from automl_agent.executor import terminate_tree
from tools import pytorch_agent_study as study


def worker(name,out,freeze):
    q=json.loads(Path(freeze).read_text());chosen=q['selected'][name]
    for path,expected in q['source_hashes'].items():
        if file_hash(ROOT/path)!=expected:raise ValueError('Control recipe source changed after freeze: '+path)
    original=study.prepare
    def control_prepare(task_name,directory):
        data,X,y,classes,templates,task,strategy,columns,cat=original(task_name,directory)
        path=directory/'splits.npz';old_hash=file_hash(path);old=np.load(path)
        train=np.sort(np.r_[old['fit'],old['validation']]);evaluation=old['control']
        if set(train)&set(evaluation):raise ValueError('Control overlaps training rows')
        target=task.targets[0]
        a,b=study.study_inner(data.train.iloc[train],pd.DataFrame({target:y[train]},index=data.train.iloc[train].index),task,strategy,42)
        oa,ob=train[a],train[b]
        na,nb=study.study_inner(data.train.iloc[oa],pd.DataFrame({target:y[oa]},index=data.train.iloc[oa].index),task,strategy,42)
        np.savez(path,fit=train,validation=evaluation,control=evaluation,oof_fit=oa,oof_calibration=ob,nested_fit=oa[na],nested_stop=oa[nb])
        protocol=json.loads((directory/'protocol.json').read_text())
        protocol.update(phase='research_control_once',control_previously_viewed_in_project='known_for_organizers; possible_for_parallel_extra_task_runs',
            control_labels_used_for_fit=False,test_labels_read=False,original_split_sha256=old_hash,split_hash=file_hash(path),
            control_freeze_sha256=file_hash(Path(freeze)),frozen_candidate=chosen,seed=q['seed'])
        atomic_json(directory/'protocol.json',protocol)
        return data,X,y,classes,templates,task,strategy,columns,cat
    study.prepare=control_prepare
    study.run(name,out,q['seed'],[chosen])
    report=json.loads((out/'results.json').read_text());report.update(control_opened=True,quality_scope='reused research control; not external/official quality',control_freeze_sha256=file_hash(Path(freeze)))
    atomic_json(out/'results.json',report)


def run(freeze,out):
    freeze=Path(freeze).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    q=json.loads(freeze.read_text())
    if q.get('control_already_opened') is not False:raise ValueError('A pre-control freeze is required')
    for name in q['selected']:
        destination=out/name;env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','TMPDIR'}}
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/private/tmp/pytorch-mpl')
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',name,'--freeze',str(freeze),'--output',str(destination)]
        start=time.monotonic();peak=0;stop=None
        with (out/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while process.poll() is None:
                    try:peak=max(peak,psutil.Process(process.pid).memory_info().rss)
                    except psutil.NoSuchProcess:break
                    if peak>4*1024**3:stop='rss_limit'
                    if time.monotonic()-start>1200:stop='deadline'
                    if stop:terminate_tree(process);break
                    time.sleep(.2)
                process.wait()
            finally:
                if process.poll() is None:terminate_tree(process)
        atomic_json(out/(name+'_watchdog.json'),dict(returncode=process.returncode,stop=stop,seconds=time.monotonic()-start,peak_rss_bytes=peak,api_calls=0))
        print(name,process.returncode,stop,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker');a=p.parse_args()
    if a.worker:worker(a.worker,a.output,a.freeze)
    else:run(a.freeze,a.output)
