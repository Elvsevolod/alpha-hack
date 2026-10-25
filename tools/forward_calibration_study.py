"""Forward-only meta-learning on fresh fixed-round reference forecasts.

This stage reuses the native reference worker and production learned-blend
implementation. It never labels historical search predictions as strict OOF.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from automl_agent.util import atomic_json,file_hash,digest


def run(out,seconds=600):
    import joblib
    from tools.plan_development_benchmark import tasks
    from tools.strong_reference_study import archive,locations,jobs,baseline
    from automl_agent.residual_calibration import ResidualCalibration
    from automl_agent.learned_blend import fit_blend
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);archive(out,create=True)
    d=tasks()['regression'];task=d.task;target=task.targets[0];y=d.train[target].to_numpy()
    source,names=locations('regression');original=np.load(source/'splits.npz')
    period=d.train[task.time_column]
    values=sorted(period.dropna().unique())
    if len(values)<5:raise ValueError('This reference comparison needs three calibration origins before its saved audit')
    # Derive origins from the saved development region, not from a task name.
    development_periods=sorted(period.iloc[original['development']].unique())
    origins=development_periods[1:]
    if len(origins)<3:raise ValueError('Insufficient prior origins for forward meta-validation')
    members=[]
    for name in names:
        state=joblib.load(source/(name+'_audit')/'model.joblib')
        members.append(dict(reference=name,reference_stage='audit',spec=state['spec']))
    arrays={};folds=[]
    for i,p in enumerate(origins):
        train=np.flatnonzero(period<p);query=np.flatnonzero(period==p)
        arrays['fit'+str(i)]=train;arrays['eval'+str(i)]=query
        folds.append(dict(fit=train.tolist(),evaluation=query.tolist(),period=float(p)))
    np.savez(out/'splits.npz',**arrays)
    policies=['identity','weighted_oof','stacked_oof','residual_half','residual_full']
    atomic_json(out/'protocol.json',dict(format=1,candidates={'regression':{'strong_pair':members}},
        data_hashes={'regression':d.source_hash},split_override={'regression':str(out/'splits.npz')},
        policies=policies,forward_origins=[float(p) for p in origins],
        feature_roles=task.model_dump(mode='json'),source_hash=file_hash(Path(__file__)),
        worker_hash=file_hash(ROOT/'tools/strong_reference_study.py'),split_hash=file_hash(out/'splits.npz'),
        source_hashes={p.name:file_hash(p) for p in (out/'reference_sources').glob('*.py')},
        baseline=39.03614722027966,meta_role='strict_forward_fixed_round_oof; no search-validation relabeling',
        protected_labels_used_for_fit=False,api_calls=0,automatic_promotion=False,previous_audit_reused=True))
    shutil.copyfile(Path(__file__),out/'source_snapshot.py')
    complete=jobs(out,'regression',['strong_pair'],[str(i) for i in range(len(origins))],time.monotonic()+seconds)
    if not complete:raise RuntimeError('Incomplete fixed-round OOF; do not fit meta-models')
    predictions={i:[np.load(out/'regression'/('strong_pair_cv'+str(i))/(name+'_prediction.npy')) for name in names] for i in range(len(origins))}

    def meta(policy,prior,query,query_members):
        keys=[f'train:{int(row)}' for i in prior for row in arrays['eval'+str(i)]]
        rows=np.concatenate([arrays['eval'+str(i)] for i in prior])
        actual=pd.DataFrame({target:y[rows]},index=keys)
        mean=sum(query_members)/len(query_members)
        if policy=='identity':return mean,dict(policy=policy)
        if policy.startswith('residual_'):
            reference=np.concatenate([sum(predictions[i])/len(names) for i in prior])
            model=ResidualCalibration(task.categorical_columns,strength=.5 if policy=='residual_half' else 1)
            model.fit(d.train.iloc[rows],y[rows],reference)
            return model.predict(d.train.iloc[query],mean),dict(policy=policy,model=model,rows=len(rows))
        # The production learner correctly requires two real outer folds.
        if len(prior)<2:return mean,dict(policy=policy,fallback='insufficient_oof_support')
        reports=[]
        for j,name in enumerate(names):
            reports.append(dict(row_keys=keys,classes=task.classes,
                values={target:np.concatenate([predictions[i][j] for i in prior])},
                folds=[folds[i] for i in prior],folds_hash=digest([folds[i] for i in prior]),prediction_role='strict_oof'))
        weights,learned,evidence=fit_blend(task,reports,actual,policy,42)
        if weights is not None:p=np.average(query_members,axis=0,weights=weights)
        else:
            layer=learned[target];p=np.column_stack(query_members)@np.asarray(layer['coef'])+np.asarray(layer['intercept'])
        return p,dict(policy=policy,weights=weights,meta=learned,evidence=evidence)

    entries=[]
    for policy in policies:
        scores=[]
        for i in range(1,len(origins)):
            rows=arrays['eval'+str(i)]
            p,state=meta(policy,list(range(i)),rows,predictions[i])
            scores.append(float(np.sqrt(np.mean((y[rows]-p)**2))))
        entries.append(dict(policy=policy,scores=scores,score=float(np.mean(scores))))
    entries.sort(key=lambda r:r['score']);chosen=entries[0]
    atomic_json(out/'selection.json',dict(chosen=chosen,entries=entries,sealed_before_audit=True))
    print('FORWARD_SELECTION',chosen,flush=True)
    if chosen['policy']=='identity':
        atomic_json(out/'summary.json',dict(status='baseline_retained',chosen=chosen,control_not_reopened=True));return
    control=original['control'];reference=[]
    # Baseline reads the frozen saved models, not their final-test npy files.
    for name in names:reference.append(baseline(source,[name],'audit',None,d.train.iloc[control],'regression'))
    p,state=meta(chosen['policy'],list(range(len(origins))),control,reference)
    ref=sum(reference)/len(reference)
    a=float(np.sqrt(np.mean((y[control]-ref)**2)));b=float(np.sqrt(np.mean((y[control]-p)**2)))
    if not np.isclose(a,39.03614722027966,rtol=1e-8):raise ValueError('Reference score does not reproduce')
    joblib.dump(state,out/'frozen_meta.joblib');np.save(out/'candidate_audit.npy',p)
    atomic_json(out/'summary.json',dict(status='completed',baseline=a,candidate=b,signed_improvement=a-b,
        chosen=chosen,reused_audit=True,csv_changed=False,official_score_unknown=True,base_fits=2*len(origins),api_calls=0))
    print('FORWARD_AUDIT',a,b,flush=True)


def export(study,destination):
    """Freeze a separate CSV; final calibration may use full train OOF labels.

    Do not score that final-refit calibrator on the old audit again: its labels
    legitimately join final training only after the method was frozen.
    """
    import joblib
    from tools.plan_development_benchmark import tasks
    from tools.strong_reference_study import archive,locations,baseline
    from automl_agent.residual_calibration import ResidualCalibration
    study=Path(study).resolve();destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False);archive(study)
    q=json.loads((study/'protocol.json').read_text());summary=json.loads((study/'summary.json').read_text())
    if summary.get('status')!='completed' or summary['candidate']>=summary['baseline']:
        raise ValueError('No accepted measured research improvement to export')
    d=tasks()['regression'];task=d.task;target=task.targets[0]
    if q['data_hashes']['regression']!=d.source_hash:raise ValueError('Changed data')
    state=joblib.load(study/'frozen_meta.joblib')
    if not state['policy'].startswith('residual_'):raise ValueError('This exporter supports the checked residual recipe only')
    original=state['model'];source,names=locations('regression')
    split=np.load(study/'splits.npz');historical=np.load(source/'splits.npz')
    rows=[];predictions=[]
    for i in range(len(q['forward_origins'])):
        rows.append(split['eval'+str(i)])
        predictions.append(np.load(study/'regression'/('strong_pair_cv'+str(i))/'prediction.npy'))
    rows.append(historical['control'])
    predictions.append(baseline(source,names,'audit',None,d.train.iloc[historical['control']],'regression'))
    rows=np.concatenate(rows);p=np.concatenate(predictions)
    if len(np.unique(rows))!=len(rows):raise ValueError('Overlapping OOF keys')
    model=ResidualCalibration(original.categorical,original.shrinkage,original.strength)
    model.fit(d.train.iloc[rows],d.train.iloc[rows][target].to_numpy(),p)
    reference=baseline(source,names,'final',None,d.test,'regression')
    final=model.predict(d.test,reference)
    output=d.test[task.id_columns].copy();output[target]=final;output=output[task.output.columns]
    if len(output)!=len(d.test) or not np.isfinite(final).all() or output.isna().any().any():
        raise ValueError('Invalid final output')
    if task.id_columns and output.duplicated(task.id_columns).any():raise ValueError('Duplicate output identifiers')
    output.to_csv(destination/'submission.csv',index=False)
    restored=pd.read_csv(destination/'submission.csv')
    if list(restored)!=task.output.columns or len(restored)!=len(output):raise ValueError('CSV schema roundtrip failed')
    np.testing.assert_allclose(restored[target],final,rtol=1e-12)
    joblib.dump(model,destination/'calibrator.joblib');np.save(destination/'prediction.npy',final)
    np.testing.assert_allclose(joblib.load(destination/'calibrator.joblib').predict(d.test,reference),final,rtol=1e-12)
    atomic_json(destination/'manifest.json',dict(format=1,scope='separate_research_candidate_not_official_submission',
        data_hash=d.source_hash,study_sha256=file_hash(study/'protocol.json'),
        selection_sha256=file_hash(study/'selection.json'),frozen_policy=state['policy'],
        calibrator=dict(categorical=model.categorical,shrinkage=model.shrinkage,strength=model.strength),
        final_oof_rows=len(rows),final_fit_includes_former_control_labels=True,
        audit_score_before_final_refit=summary['candidate'],audit_not_rescored_after_final_refit=True,
        baseline_members={str((source/(name+'_final')/'model.joblib').relative_to(ROOT)):file_hash(source/(name+'_final')/'model.joblib') for name in names},
        submission_sha256=file_hash(destination/'submission.csv'),rows=len(output),columns=list(output),
        official_score_unknown=True,canonical_csv_untouched=True,api_calls=0))
    print('EXPORTED',destination/'submission.csv',len(output),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--export-to',type=Path)
    args=p.parse_args()
    if args.export_to:export(args.output,args.export_to)
    else:run(args.output,args.seconds)
