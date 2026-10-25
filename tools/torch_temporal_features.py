"""An explicit temporal feature-group ablation for the PyTorch dev probe.

Low repetition support does not prove a feature useless. This is a candidate,
never an automatic default, and uses no dataset or feature-name exceptions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd


def temporal_feature_group(frame,columns,categorical,time_column,limit=64):
    if time_column is None or time_column not in frame or frame[time_column].nunique()<2:
        raise ValueError('Temporal feature-group hypothesis needs at least two fit periods')
    if limit<1:raise ValueError('Positive finite-support engineering limit required')
    numeric=[c for c in columns if c not in categorical]
    support=frame.groupby(time_column,dropna=False)[numeric].nunique().max().to_dict()
    kept=[c for c in columns if c in categorical or c!=time_column and support[c]>limit]
    dropped=[c for c in columns if c not in kept]
    if not kept or not dropped:raise ValueError('This ablation must retain and remove features')
    return kept,dropped,support


def study(source,output):
    import joblib
    from tools.plan_development_benchmark import tasks
    from tools.torch_tabular_probe import TorchMLP,metric
    from automl_agent.contracts import PipelineSpec
    from automl_agent.pipelines import make_estimator
    from automl_agent.util import atomic_json,file_hash
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    # The development suite contains one declared temporal numeric task. The
    # transform below is role-driven and does not know its dataset/column names.
    d=next(d for d in tasks().values() if d.task.time_column and not d.task.classes)
    task=d.task;target=task.targets[0];source=Path(source)/'regression'
    original=json.loads((source/'protocol.json').read_text())
    if original['data_hash']!=d.source_hash:raise ValueError('Inputs changed')
    split=np.load(source/'splits.npz');a=split['inner_fit'];b=split['inner_stop'];fit=split['fit'];val=split['validation']
    X=d.train[task.feature_columns];y=d.train[target]
    kept,dropped,support=temporal_feature_group(X.iloc[a],task.feature_columns,task.categorical_columns,task.time_column)
    configs=[dict(name='repetition_quantile_mlp',encoding='quantile',width=256,depth=3,heads=1,epochs=100,learning_rate=.002),
        dict(name='repetition_piecewise_batchensemble',encoding='piecewise',width=128,depth=3,heads=8,epochs=100,learning_rate=.002)]
    atomic_json(out/'protocol.json',dict(scope='explicit_feature_group_development_ablation',source_hash=file_hash(Path(__file__)),
        helper_hash=file_hash(ROOT/'tools/torch_tabular_probe.py'),source_study=str(source),data_hash=d.source_hash,
        split_hash=file_hash(source/'splits.npz'),selected_on='inner_fit_X_only',numeric_period_support_limit=64,
        support=support,kept=kept,dropped=dropped,configs=configs,neural_mix_weights=[.25,.5],
        original_control_previously_inspected=True,control_not_used_for_selection=True,
        explanation='Hypothesis that heavily repeated temporal numeric coordinates encourage unstable transfer; feature repetition does not prove cause or uselessness',
        api_calls=0,fit_count_expected=5,baseline='same filtered features, 200-tree LightGBM',automatic_promotion=False))
    results=[];prediction={};entries={}
    for config in configs:
        name=config['name'];kwargs={k:v for k,v in config.items() if k!='name'}
        selector=TorchMLP(kept,task.categorical_columns,None,**kwargs).fit(X.iloc[a],y.iloc[a],(X.iloc[b],y.iloc[b]))
        model=TorchMLP(kept,task.categorical_columns,None,**dict(kwargs,epochs=selector.best_epoch,seconds=60))
        model.fit(X.iloc[fit],y.iloc[fit]);p=model.predict(X.iloc[val]);prediction[name]=p
        joblib.dump(model,out/f'{name}.joblib');np.save(out/f'{name}.npy',p)
        entry=dict(model=name,score=metric(y.iloc[val],p,None),config=config,frozen_epochs=selector.best_epoch,
            actual_epochs=model.epochs_run,refit_truncated=model.epochs_run<selector.best_epoch,
            selected_seconds=selector.seconds_used,refit_seconds=model.seconds_used)
        results.append(entry);entries[name]=entry;print(entry,flush=True)
    frame=X[kept].copy()
    for c in task.categorical_columns:frame[c]=pd.Categorical(frame[c].astype(str),categories=sorted(frame.iloc[fit][c].astype(str).unique()))
    tree=make_estimator(PipelineSpec(model='lightgbm',hyperparameters={'n_estimators':200,'num_leaves':31,'learning_rate':.05}),False,{'cpu':1})
    tree.fit(frame.iloc[fit],y.iloc[fit]);tp=tree.predict(frame.iloc[val]);np.save(out/'tree.npy',tp);joblib.dump(tree,out/'tree.joblib')
    results.append(dict(model='filtered_tree',score=metric(y.iloc[val],tp,None)))
    for name,p in prediction.items():
        for weight in [.25,.5]:results.append(dict(model=f'tree_plus_{name}_{weight}',score=metric(y.iloc[val],(1-weight)*tp+weight*p,None),member=name,neural_weight=weight))
    results.sort(key=lambda r:r['score']);chosen=results[0]
    atomic_json(out/'selection.json',dict(chosen=chosen,entries=entries,sealed_before_control=True,created=time.time()))
    atomic_json(out/'development.json',dict(results=results,closed_control_scored=False))
    print('SELECTION',chosen,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();study(a.source,a.output)
