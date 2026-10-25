"""Select completed DL recipes and estimate paired development uncertainty.

No control labels are evaluated here. Reconciles the frozen historical residual
calibrator's original categorical roles, retaining the original raw run results.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from automl_agent.util import atomic_json,file_hash
from automl_agent.identity import feature_groups
from tools.pytorch_agent_study import load_dataset,references,ResidualHalf,metric,WEIGHTS


def evaluate(directory,bootstrap=400):
    directory=Path(directory);protocol=json.loads((directory/'protocol.json').read_text());raw=json.loads((directory/'results.json').read_text())
    name=protocol['dataset'];refs=references();d=load_dataset(name);target=d.task.targets[0];classes=protocol['classes']
    y=np.asarray(__import__('pandas').Categorical(d.train[target],categories=classes).codes) if classes else d.train[target].to_numpy()
    split=np.load(directory/'splits.npz');b=split['oof_calibration'];val=split['validation'];nb=split['nested_stop']
    cp=np.load(directory/'baseline_oof.npy');vp=np.load(directory/'baseline_full.npy');baseline_raw=metric(y[val],vp,classes)
    if name=='regression':
        # Original strongest calibrator uses original categorical roles, not new crosses.
        pp=np.load(directory/'baseline_earlier.npy');cats=d.task.categorical_columns
        cp0=cp.copy();cp=ResidualHalf(cats).fit(d.train.iloc[nb],y[nb],pp).predict(d.train.iloc[b],cp)
        vp=ResidualHalf(cats).fit(d.train.iloc[np.r_[nb,b]],y[np.r_[nb,b]],np.r_[pp,cp0]).predict(d.train.iloc[val],vp)
    baseline_offset=refs.fit_offsets(cp,y[b]) if classes else None
    baseline_score=metric(y[val],vp,classes,baseline_offset)
    reference_envelope=max(baseline_score,baseline_raw) if classes else min(baseline_score,baseline_raw)
    np.save(directory/'canonical_baseline.npy',vp)
    if classes:groups=feature_groups(d.train.iloc[val][d.task.feature_columns]) if not d.task.group_column else d.train.iloc[val][d.task.group_column].to_numpy()
    else:groups=d.train.iloc[val][d.task.time_column].to_numpy() if d.task.time_column else np.arange(len(val))
    unique=np.unique(groups);units=[np.flatnonzero(groups==u) for u in unique]
    result=[]
    for row in raw['results']:
        if row.get('status')!='completed':continue
        label=row['model'];npred=np.load(directory/f'{label}_oof.npy');nval=np.load(directory/f'{label}_full.npy')
        choices=[]
        for w in WEIGHTS:
            blend=(1-w)*cp+w*npred;offset=refs.fit_offsets(blend,y[b]) if classes else None
            score=metric(y[b],blend,classes,offset);choices.append((score if classes else -score,-w,w,offset))
        _,_,weight,offset=max(choices,key=lambda r:r[:2]);p=(1-weight)*vp+weight*nval
        score=metric(y[val],p,classes,offset);gain=score-baseline_score if classes else baseline_score-score
        sample=[];rng=np.random.default_rng(42)
        if len(units)>=2:
            for _ in range(bootstrap):
                rows=np.concatenate([units[i] for i in rng.integers(len(units),size=len(units))])
                a=metric(y[val][rows],p[rows],classes,offset);base=metric(y[val][rows],vp[rows],classes,baseline_offset)
                sample.append(a-base if classes else base-a)
        np.save(directory/f'{label}_canonical_blend.npy',p)
        result.append(dict(model=label,seed=row['seed'],baseline=baseline_score,score=score,signed_gain=gain,weight=weight,
            gain_vs_reference_envelope=score-reference_envelope if classes else reference_envelope-score,reference_envelope=reference_envelope,standalone_score=row['standalone_score'],seconds=row['seconds'],base_fits=row['base_fits'],fits=row['fits'],
            paired_interval_95=np.quantile(sample,[.025,.975]).tolist() if sample else None,
            uncertainty_unit='subjects' if d.task.group_column else 'exact_feature_groups' if classes else 'time_periods' if d.task.time_column else 'rows',
            uncertainty_units=len(units),uncertainty_note='Development reuse and recipe selection are not corrected. Single time period cannot identify block uncertainty.' if len(units)<2 else 'Paired unit bootstrap; fixed fitted recipes; not retraining/selection uncertainty.',
            offset=offset.tolist() if classes else None))
    result.sort(key=lambda r:(-r['signed_gain'], -r['standalone_score'] if classes else r['standalone_score']))
    summary=dict(dataset=name,baseline=baseline_score,raw_baseline=baseline_raw,reference_envelope=reference_envelope,source_protocol_sha256=file_hash(directory/'protocol.json'),
        control_opened=False,selection_criterion='development score of OOF-selected mixture; standalone score breaks ties',results=result,selected_two=[r['model'] for r in result[:2]],
        baseline_reconciliation='original categorical roles for historical half-residual calibration; no control labels or metric-based choice')
    atomic_json(directory/'canonical_analysis.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directories',nargs='+',type=Path);p.add_argument('--bootstrap',type=int,default=400);a=p.parse_args()
    for directory in a.directories:
        s=evaluate(directory,a.bootstrap);print(s['dataset'],s['baseline'],s['selected_two'],[(r['model'],r['signed_gain'],r['weight']) for r in s['results']],flush=True)
