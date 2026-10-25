"""Fit-only label policies and exploratory blends for the offline DL pilot.

All external scores below use development validation, never closed control.
No automatic promotion. This is not a live Qwen run.
"""
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
import numpy as np
import pandas as pd


def study(source,output,kind):
    import joblib
    from tools.plan_development_benchmark import tasks
    from tools.torch_tabular_probe import TorchMLP,metric
    from automl_agent.contracts import PipelineSpec
    from automl_agent.pipelines import make_estimator
    from automl_agent.decision import fit_decision,decide
    from automl_agent.evaluation import score_predictions
    from automl_agent.util import atomic_json,file_hash

    source=Path(source)/kind;out=Path(output)/kind;out.mkdir(parents=True,exist_ok=False)
    original=json.loads((source/'protocol.json').read_text());summary=json.loads((source/'results.json').read_text())
    d=tasks()[kind];task=d.task;target=task.targets[0]
    if d.source_hash!=original['data_hash']:raise ValueError('Source data changed')
    split=np.load(source/'splits.npz');fit=split['fit'];val=split['validation'];a=split['inner_fit'];b=split['inner_stop']
    X=d.train[task.feature_columns];y=d.train[[target]];classes=task.classes.get(target) if task.classes else None
    atomic_json(out/'protocol.json',dict(scope='offline_development_only',source_study=str(source.relative_to(ROOT)),
        closed_control_used=False,api_calls=0,ensemble_neural_weights=[.25,.5],decision_method='regularized',
        epoch_and_policy_share_inner_holdout=True,selection_is_exploratory=True,
        source_hash=file_hash(Path(__file__)),data_hash=d.source_hash,
        split_hash=file_hash(source/'splits.npz'),automatic_promotion=False))
    psets={};sp={};fits=0;results=[];excluded=[]
    configs={c['name']:c for c in original['architectures']}
    for record in summary['results']:
        name=record['model']
        if name=='lightgbm':continue
        if record.get('refit_truncated'):
            excluded.append(dict(model=name,reason='incomplete_frozen_refit'))
            continue
        p=np.load(source/f'{name}_prediction.npy');psets[name]=p
        results.append(dict(model=name,policy='raw',score=metric(y.iloc[val][target],p,classes)))
        if classes:
            kwargs={k:v for k,v in configs[name].items() if k!='name'};kwargs['epochs']=record['best_epoch']
            model=TorchMLP(task.feature_columns,task.categorical_columns,classes,
                seconds=original['seconds_per_fit'],**kwargs).fit(X.iloc[a],y.iloc[a][target])
            if model.epochs_run!=record['best_epoch']:
                raise RuntimeError('Incomplete calibration refit; no decision policy is accepted')
            fits+=1;sp[name]=model.predict(X.iloc[b]);joblib.dump(model,out/f'{name}_calibration.joblib')
    frame=X.copy()
    for c in task.categorical_columns:
        frame[c]=pd.Categorical(frame[c].astype(str),categories=sorted(frame.iloc[fit][c].astype(str).unique()))
    specification=PipelineSpec(model='lightgbm',hyperparameters={'n_estimators':200,'num_leaves':31,'learning_rate':.05})
    def fit_tree(rows):
        model=make_estimator(specification,bool(classes),{'cpu':1})
        labels=pd.Categorical(y.iloc[rows][target],categories=classes).codes if classes else y.iloc[rows][target]
        kwargs={}
        if classes:
            f=np.bincount(labels,minlength=len(classes));kwargs['sample_weight']=np.sqrt(f.max()/f)[labels]
        model.fit(frame.iloc[rows],labels,**kwargs)
        return model
    tree=fit_tree(fit);fits+=1
    psets['lightgbm']=tree.predict_proba(frame.iloc[val]) if classes else tree.predict(frame.iloc[val])
    joblib.dump(tree,out/'lightgbm.joblib')
    np.testing.assert_allclose(metric(y.iloc[val][target],psets['lightgbm'],classes),summary['results'][-1]['score'],rtol=1e-9)
    if classes:
        inner=fit_tree(a);fits+=1;sp['lightgbm']=inner.predict_proba(frame.iloc[b])
    results.append(dict(model='lightgbm',policy='raw',score=metric(y.iloc[val][target],psets['lightgbm'],classes)))
    policies={}
    def assess(name,p,calibration=None):
        if classes:
            state=fit_decision(task,y.iloc[b],calibration,X.index[b],method='regularized',scope='fit_inner_reused_stopping_and_decision')
            prediction=decide(p,classes,state);score=score_predictions(task,y.iloc[val],{target:prediction},'labels')
            policies[name]=state
            results.append(dict(model=name,policy='fit_inner_offsets',score=score))
        else:results.append(dict(model=name,policy='raw_blend',score=metric(y.iloc[val][target],p,None)))
    for name,p in psets.items():
        if classes:assess(name,p,sp[name])
    for name in configs:
        if name not in psets:continue
        for weight in [.25,.5]:
            label=f'lightgbm_plus_{name}_{weight}'
            p=(1-weight)*psets['lightgbm']+weight*psets[name]
            if classes:
                results.append(dict(model=label,policy='raw_blend',score=metric(y.iloc[val][target],p,classes)))
            assess(label,p,(1-weight)*sp['lightgbm']+weight*sp[name] if classes else None)
    results.sort(key=lambda r:(-1 if classes else 1)*r['score'])
    atomic_json(out/'results.json',dict(results=results,additional_base_fits=fits,policies=policies,
        excluded=excluded,
        control_opened=False,quality_scope='adaptive development validation; no official score or full-agent claim',
        promotion=False))
    print(kind,results[:4],flush=True)


def run(source,output):
    from automl_agent.util import atomic_json
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    failed=[]
    env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','TMPDIR'}}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for kind in ['regression','classification']:
        with (out/f'{kind}.log').open('w') as log:
            try:
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--source',str(Path(source).resolve()),
                    '--output',str(out),'--kind',kind],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                print(kind,'return code',result.returncode,flush=True)
                if result.returncode:
                    atomic_json(out/f'{kind}_failure.json',dict(returncode=result.returncode))
                    failed.append(kind)
            except subprocess.TimeoutExpired:
                atomic_json(out/f'{kind}_failure.json',dict(reason='worker_timeout',seconds=300))
                failed.append(kind)
    if failed:raise RuntimeError('Decision probe failed: '+', '.join(failed))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--kind',choices=['regression','classification'])
    args=parser.parse_args()
    if args.kind:study(args.source,args.output,args.kind)
    else:run(args.source,args.output)
