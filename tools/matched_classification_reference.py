"""Fixed historical tree procedure on the autonomous agent's exact control split."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def worker(out):
    import joblib
    import numpy as np
    import pandas as pd
    from automl_agent.util import file_hash
    from tools import realmlp_equal_budget as study
    from tools.pytorch_agent_study import load_dataset

    freeze=json.loads((out/'freeze.json').read_text())
    def prepare(name,directory):
        assert name=='classification'
        d=load_dataset(name);assert d.source_hash==freeze['data_hashes'][name]
        refs=study.references();task=d.task;target=task.targets[0]
        classes=sorted(task.classes[target]);y=pd.Categorical(d.train[target],categories=classes).codes
        assert (y>=0).all()
        templates=[str(ROOT/p) for p in freeze['templates']]
        states=[joblib.load(path) for path in templates]
        state=states[0]
        assert all(s['feature_plan']==state['feature_plan'] and s['columns']==state['columns'] for s in states)
        assert states[1]['normal_index']==classes.index('Normal')
        X=refs.engineer(d.train,state['feature_plan'])
        split=out/'agent_split.npz';assert file_hash(split)==freeze['split_sha256']
        with np.load(split) as values:
            np.savez_compressed(directory/'splits.npz',**{key:values[key] for key in ['fit','validation','control']})
        return d,X,y,classes,templates,task,'exact_group',state['columns'],state['categorical']
    # Reuse the previously audited tree/decision procedure; replace only its
    # research loader so it receives the agent split rather than the old split.
    study.prepare=prepare
    study.worker(out/'reference/request.json')


def run(out):
    import psutil
    import numpy as np
    from automl_agent.executor import terminate_tree
    from automl_agent.util import atomic_json,file_hash
    from tools.pytorch_agent_study import load_dataset

    out.mkdir(parents=True,exist_ok=False);directory=out/'reference';directory.mkdir()
    old=ROOT/'outputs/agent_autonomy_budget_20261026/task/.agent_work/bounded_autonomy/journal.sqlite'
    db=sqlite3.connect(old.as_uri()+'?mode=ro',uri=True)
    plan=json.loads(db.execute("select value from metadata where key='validation'").fetchone()[0]);db.close()
    d=load_dataset('classification');assert d.source_hash==plan['data_version']
    np.savez_compressed(out/'agent_split.npz',**{key:np.asarray(plan[key]) for key in ['fit','validation','control']})
    templates=['research/pytorch_agent_20261026/reference_templates/'+name+'.joblib' for name in ['current_weighted','hierarchy_balanced']]
    paths=[Path(__file__),ROOT/'tools/realmlp_equal_budget.py',ROOT/'tools/realmlp_td_s.py',ROOT/'tools/pytorch_agent_study.py',ROOT/'tools/torch_research_core.py',
        *sorted((ROOT/'automl_agent').glob('*.py')),*sorted((ROOT/'research/pytorch_agent_20261026/reference_sources').glob('*.py')),
        *[ROOT/p for p in templates]]
    atomic_json(out/'freeze.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in paths},data_hashes=dict(classification=d.source_hash),
        split_sha256=file_hash(out/'agent_split.npz'),original_agent_protocol=plan,templates=templates,
        seed=42,arm_wall_seconds=900,memory_mb=3072,threads=1,
        source_recipe='unchanged author-study fixed tree arm, fit-only grouped decision selection; only outer split replaced',
        control_previously_viewed=True,independent_confirmation=False,production_promotion=False,api_calls=0))
    atomic_json(directory/'request.json',dict(freeze=str(out/'freeze.json'),task='classification',arm='trees',phase='control'))
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MPLCONFIGDIR=str(out/'mpl'))
    begin=time.monotonic();peak=0;reason=None;usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    with (directory/'worker.log').open('w') as log:
        process=subprocess.Popen([sys.executable,__file__,str(out),'--worker'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    parent=psutil.Process(process.pid)
                    peak=max(peak,sum(p.memory_info().rss for p in [parent,*parent.children(recursive=True)] if p.is_running()))
                except psutil.NoSuchProcess:pass
                if time.monotonic()-begin>=900:reason='timeout'
                if peak>3072*1024**2:reason='memory_limit'
                if reason:terminate_tree(process)
                time.sleep(.1)
            process.wait()
        finally:
            if process.poll() is None:terminate_tree(process)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    cost=dict(returncode=process.returncode,stop=reason,cold_wall_seconds=time.monotonic()-begin,
        child_cpu_seconds=after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime,
        sampled_tree_peak_mib=peak/1024**2,limit_seconds=900,cpu_threads=1,memory_mb=3072,api_calls=0)
    atomic_json(out/'cost.json',cost);print(json.dumps(cost),flush=True)
    if process.returncode!=0:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    worker(args.output.resolve()) if args.worker else run(args.output.resolve())
