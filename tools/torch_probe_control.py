"""Freeze the DL pilot's selection, then audit once without automatic promotion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def freeze(source,decisions,output):
    from automl_agent.util import atomic_json,file_hash
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    jobs={}
    for kind in ['regression','classification']:
        run=Path(source).resolve()/kind;d=Path(decisions).resolve()/kind
        protocol=json.loads((run/'protocol.json').read_text());results=json.loads((run/'results.json').read_text())
        selection=json.loads((d/'results.json').read_text());best=selection['results'][0]
        prefix='lightgbm_plus_'
        if not best['model'].startswith(prefix):raise ValueError('This follow-up audits a selected simple tree/neural blend only')
        member,weight=best['model'][len(prefix):].rsplit('_',1)
        config=next(c for c in protocol['architectures'] if c['name']==member)
        record=next(r for r in results['results'] if r['model']==member)
        jobs[kind]=dict(source=str(run),decisions=str(d),data_hash=protocol['data_hash'],
            split_sha256=file_hash(run/'splits.npz'),neural_config=config,frozen_epochs=record['best_epoch'],
            neural_weight=float(weight),selection=best,
            baseline_state=selection.get('policies',{}).get('lightgbm'),
            selected_state=selection.get('policies',{}).get(best['model']),
            seconds_per_refit=90,source_result_hash=file_hash(d/'results.json'))
    atomic_json(out/'frozen.json',dict(jobs=jobs,created_at=time.time(),code_sha256=file_hash(Path(__file__)),
        controls_previously_used_in_project=True,control_not_used_to_choose_these_recipes=True,
        primary_comparison='same 200-tree baseline with its frozen fit-only decision policy',
        baseline_csv_promotion=False,api_calls=0))
    env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','TMPDIR'}}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    failed=[]
    for kind in jobs:
        with (out/f'{kind}.log').open('w') as log:
            try:
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',kind,'--output',str(out)],
                    env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                print(kind,'return code',result.returncode,flush=True)
                if result.returncode:
                    atomic_json(out/f'{kind}_failure.json',dict(returncode=result.returncode))
                    failed.append(kind)
            except subprocess.TimeoutExpired:
                atomic_json(out/f'{kind}_failure.json',dict(reason='worker_timeout',seconds=300))
                failed.append(kind)
    if failed:raise RuntimeError('Control audit failed: '+', '.join(failed))


def worker(kind,output):
    import joblib
    import numpy as np
    import pandas as pd
    from tools.plan_development_benchmark import tasks
    from tools.torch_tabular_probe import TorchMLP
    from automl_agent.contracts import PipelineSpec
    from automl_agent.pipelines import make_estimator
    from automl_agent.evaluation import score_predictions
    from automl_agent.decision import decide
    from automl_agent.util import atomic_json,file_hash

    out=Path(output);q=json.loads((out/'frozen.json').read_text())['jobs'][kind]
    path=out/kind;path.mkdir(exist_ok=False)
    d=tasks()[kind];task=d.task;target=task.targets[0];classes=task.classes.get(target) if task.classes else None
    if d.source_hash!=q['data_hash']:raise ValueError('Data changed since selection')
    split_path=Path(q['source'])/'splits.npz'
    if file_hash(split_path)!=q['split_sha256']:raise ValueError('Split changed since selection')
    # Original protocol records validation; retrieve exact control from current
    # deterministic plan, then assert its development rows match saved lineage.
    from automl_agent.validation import choose_validation
    from automl_agent.protocols import initial_protocol
    plan=initial_protocol(d,choose_validation(d,seed=42));splits=np.load(split_path)
    np.testing.assert_array_equal(plan.fit,splits['fit']);np.testing.assert_array_equal(plan.validation,splits['validation'])
    development=np.asarray(sorted(plan.fit+plan.validation));control=np.asarray(plan.control)
    assert not set(development)&set(control)
    X=d.train[task.feature_columns];y=d.train[[target]]
    config={k:v for k,v in q['neural_config'].items() if k!='name'}
    config.update(epochs=q['frozen_epochs'],seconds=q['seconds_per_refit'])
    model=TorchMLP(task.feature_columns,task.categorical_columns,classes,**config)
    model.fit(X.iloc[development],y.iloc[development][target])
    if model.epochs_run!=q['frozen_epochs']:raise RuntimeError('Frozen neural refit did not complete; do not score a different recipe')
    neural=model.predict(X.iloc[control]);joblib.dump(model,path/'neural.joblib')
    frame=X.copy()
    for c in task.categorical_columns:
        frame[c]=pd.Categorical(frame[c].astype(str),categories=sorted(frame.iloc[development][c].astype(str).unique()))
    tree=make_estimator(PipelineSpec(model='lightgbm',hyperparameters={'n_estimators':200,'num_leaves':31,'learning_rate':.05}),bool(classes),{'cpu':1})
    labels=pd.Categorical(y.iloc[development][target],categories=classes).codes if classes else y.iloc[development][target]
    kwargs={}
    if classes:
        frequency=np.bincount(labels,minlength=len(classes));kwargs['sample_weight']=np.sqrt(frequency.max()/frequency)[labels]
    tree.fit(frame.iloc[development],labels,**kwargs);joblib.dump(tree,path/'tree.joblib')
    baseline=tree.predict_proba(frame.iloc[control]) if classes else tree.predict(frame.iloc[control])
    combined=(1-q['neural_weight'])*baseline+q['neural_weight']*neural
    left=decide(baseline,classes,q['baseline_state']) if classes else baseline
    right=decide(combined,classes,q['selected_state']) if classes else combined
    ptype='labels' if classes else 'values'
    a=score_predictions(task,y.iloc[control],{target:left},ptype)
    b=score_predictions(task,y.iloc[control],{target:right},ptype)
    np.save(path/'baseline.npy',left);np.save(path/'candidate.npy',right)
    result=dict(baseline=a,candidate=b,signed_improvement=b-a if classes else a-b,
        control_rows=len(control),fit_rows=len(development),neural_epochs=model.epochs_run,
        neural_fit_seconds=model.seconds_used,base_model_fits=2,
        previously_viewed_project_control=True,official_score_unknown=True,
        live_agent_run=False,working_csv_not_changed=True,final_test_not_run=True)
    atomic_json(path/'result.json',result);print(kind,result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path);p.add_argument('--decisions',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',choices=['regression','classification']);a=p.parse_args()
    if a.worker:worker(a.worker,a.output)
    else:freeze(a.source,a.decisions,a.output)
