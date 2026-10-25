"""Improve the saved strongest recipes; offline, no automatic CSV replacement.

Historical feature hypotheses are retained as references. The agent may propose
semantic columns; GroupContext itself operates on explicitly supplied roles.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from automl_agent.util import atomic_json, file_hash

REFERENCE = 'df36940ef7614abee3f976c21de202c5ce865c00'


def archive(out, create=False):
    """Load exact local historical helpers required by trusted saved joblibs."""
    directory = Path(out)/'reference_sources'
    if create:directory.mkdir()
    for module in ['reference_features','reference_alternatives']:
        path = directory/(module+'.py')
        if create:
            content = subprocess.check_output(['git','show',f'{REFERENCE}:tools/{module}.py'],cwd=ROOT)
            path.write_bytes(content)
        spec = importlib.util.spec_from_file_location('tools.'+module,path)
        imported = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = imported
        spec.loader.exec_module(imported)


def locations(kind):
    if kind=='regression':
        root=ROOT/'outputs/literature_reference_20261025/regression'
        return root,['regularized_lgb','engineered_cb']
    return ROOT/'outputs/further_reference_20261025/classification',['current_weighted','hierarchy_balanced']


def prepare(raw,state,kind):
    from tools.reference_features import engineer
    frame=engineer(raw,state.get('feature_plan'))[state['columns']].copy()
    for c in state['categorical']:
        frame[c]=frame[c].fillna('__MISSING__').astype(str)
    return frame


def correction(p,counts):
    value=p*np.sqrt(counts.max()/counts)
    return value/value.sum(axis=1,keepdims=True)


def numeric_importance(state):
    """Preserve the feature map after OHE / missing indicators."""
    values=state['model'].feature_importances_
    names=state['columns'] if state['preprocessor'] is None else state['preprocessor'].get_feature_names_out()
    if len(names)!=len(values):raise ValueError('Importance mapping changed')
    numeric=set(state['columns'])-set(state['categorical'])
    pairs=[]
    for name,value in zip(names,values):
        original=name.removeprefix('numeric__')
        if original in numeric:pairs.append((original,float(value)))
    return [name for name,_ in sorted(pairs,key=lambda v:-v[1])]


def candidate_estimator(spec,classification):
    from automl_agent.contracts import PipelineSpec
    from automl_agent.pipelines import make_estimator
    config=spec.model_dump(mode='json')
    # The contest typed whitelist omits DART. Preserve that contract; this
    # explicit development candidate extends the same native estimator with
    # documentation-checked parameters and a fixed finite number of rounds.
    additional={}
    if spec.model=='lightgbm' and spec.hyperparameters.get('boosting_type')=='dart':
        for key in ['boosting_type','drop_rate','skip_drop']:
            if key in config['hyperparameters']:additional[key]=config['hyperparameters'].pop(key)
    if spec.model=='lightgbm' and spec.hyperparameters.get('linear_tree') is True:
        for key in ['linear_tree','linear_lambda']:
            if key in config['hyperparameters']:additional[key]=config['hyperparameters'].pop(key)
    if spec.model=='catboost' and spec.hyperparameters.get('boosting_type')=='Ordered':
        additional['boosting_type']=config['hyperparameters'].pop('boosting_type')
    model=make_estimator(PipelineSpec.model_validate(config),classification,{'cpu':2})
    if additional:model.set_params(**additional)
    return model


def baseline(path,names,stage,counts,raw=None,kind=None,fixed_rounds=False):
    import joblib
    if stage=='audit' and raw is None:
        raise ValueError('Audit predictions require explicit query rows; cached npy may contain final-test predictions')
    values=[]
    for name in names:
        if stage!='audit' and not (stage=='final' and raw is not None):
            forecast=path/f'{name}_{stage}'/'prediction.npy'
            if fixed_rounds and counts is None:
                mapping={'regularized_lgb':'current_lgb','engineered_cb':'current_cb'}
                forecast=ROOT/'outputs/further_reference_20261025/regression'/f'{mapping[name]}_{stage}'/'prediction.npy'
            values.append(np.load(forecast));continue
        state=joblib.load(path/f'{name}_{stage}'/'model.joblib')
        X=prepare(raw,state,kind)
        if state['spec']['model']=='lightgbm':
            for c in state['categorical']:X[c]=pd.Categorical(X[c],categories=state['vocabulary'][c])
        if state['preprocessor'] is not None:X=state['preprocessor'].transform(X)
        model=state['model']
        value=model.predict_proba(X) if counts is not None else model.predict(X)
        values.append(value)
    if counts is not None:values=[correction(p,counts) for p in values]
    return sum(values)/len(values)


def worker(out,kind,candidate,fold):
    import joblib
    from automl_agent.contracts import PipelineSpec
    from automl_agent.pipelines import make_estimator
    from automl_agent.group_context import GroupContext
    from tools.plan_development_benchmark import tasks
    archive(out)
    q=json.loads((Path(out)/'protocol.json').read_text())
    source,names=locations(kind);d=tasks()[kind];task=d.task;target=task.targets[0]
    if d.source_hash!=q['data_hashes'][kind]:raise ValueError('Data changed after freeze')
    split=np.load(q.get('split_override',{}).get(kind,source/'splits.npz'))
    stage='cv'+fold if fold!='audit' else 'audit'
    train=split['fit'+fold] if fold!='audit' else split['development']
    evaluate=split['eval'+fold] if fold!='audit' else split['control']
    classes=sorted(task.classes[target]) if kind=='classification' else None
    y=pd.Categorical(d.train[target],categories=classes).codes if classes else d.train[target].to_numpy()
    directory=Path(out)/kind/(candidate+'_'+stage);directory.mkdir(parents=True,exist_ok=True)
    configs=q['candidates'][kind][candidate]
    if configs[0].get('pair_refinement'):
        from automl_agent.pair_refinement import PairRefiner
        state=joblib.load(source/(names[0]+'_'+stage)/'model.joblib')
        X=prepare(d.train,state,kind)
        for c in state['categorical']:X[c]=pd.Categorical(X[c],categories=sorted(X.iloc[train][c].unique()))
        counts=np.bincount(y[train],minlength=len(classes))
        if fold=='audit':
            calibration=split['eval0']
            original_counts=np.bincount(y[split['fit0']],minlength=len(classes))
            cp=(baseline(source,names,'cv0',original_counts)+baseline(source,names,'cv1',original_counts))/2
        else:
            mask=np.isin(evaluate,split['calibration'+fold])
            calibration=evaluate[mask];cp=baseline(source,names,stage,counts)[mask]
        began=time.monotonic()
        model=PairRefiner().fit(X.iloc[train],y[train],y[calibration],cp)
        ref=baseline(source,names,stage,counts,d.train.iloc[evaluate],kind)
        p=model.predict_proba(X.iloc[evaluate],ref)
        np.save(directory/'prediction.npy',p)
        joblib.dump(dict(model=model,feature_plan=state['feature_plan'],columns=state['columns'],
            categorical=state['categorical']),directory/'pair_refiner.joblib')
        atomic_json(directory/'result.json',dict(status='completed',seconds=time.monotonic()-began,
            records=[dict(base_fits=len(model.pairs_),pairs=[dict(a=p['a'],b=p['b'],confusion=p['confusion']) for p in model.pairs_])],
            train_rows=len(train),query_rows=len(evaluate),control_used=fold=='audit'))
        return
    predictions=[];records=[];started=time.monotonic()
    for member in configs:
        state=joblib.load(source/(member['reference']+'_'+member.get('reference_stage',stage))/'model.joblib')
        X=prepare(d.train,state,kind);numeric=[c for c in X if c not in state['categorical']]
        transformer=None
        if member.get('context'):
            # Bound width using training-model importance, not query labels.
            chosen=numeric_importance(state)[:8]
            transformer=GroupContext(chosen,state['categorical']).fit(X.iloc[train])
            X=transformer.transform(X)
        spec=dict(member['spec'])
        spec=PipelineSpec.model_validate(spec)
        if spec.model=='lightgbm':
            for c in state['categorical']:
                X[c]=pd.Categorical(X[c],categories=sorted(X.iloc[train][c].unique()))
        scaler=None
        if member.get('scale_numeric'):
            from sklearn.preprocessing import StandardScaler
            scaler=StandardScaler().fit(X.iloc[train][numeric])
            X[numeric]=scaler.transform(X[numeric])
        preprocessor=None
        if spec.model=='xgboost':
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import OneHotEncoder
            preprocessor=ColumnTransformer([('numeric',SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=True),[c for c in X if c not in state['categorical']]),
                ('categories',OneHotEncoder(handle_unknown='ignore',sparse_output=False),state['categorical'])],sparse_threshold=0)
            tx=preprocessor.fit_transform(X.iloc[train]);vx=preprocessor.transform(X.iloc[evaluate])
        else:tx,vx=X.iloc[train],X.iloc[evaluate]
        if member.get('hierarchy'):
            from tools.reference_alternatives import HierarchicalLGBM
            previous=state['model']
            model=HierarchicalLGBM(member['spec'],previous.normal_index,previous.component_iterations_)
        else:model=candidate_estimator(spec,bool(classes))
        kwargs={}
        if spec.model=='catboost':kwargs['cat_features']=state['categorical']
        if member.get('weighted'):
            counts=np.bincount(y[train],minlength=len(classes))
            kwargs['sample_weight']=np.sqrt(counts.max()/counts)[y[train]]
        model.fit(tx,y[train],**kwargs)
        p=model.predict_proba(vx) if classes else model.predict(vx)
        if member.get('prediction_column') is not None:p=p[:,member['prediction_column']]
        if classes:
            counts=np.bincount(y[train],minlength=len(classes));p=correction(p,counts)
        if not np.isfinite(p).all():raise ValueError('Nonfinite prediction')
        predictions.append(p)
        name=member['reference']
        np.save(directory/(name+'_prediction.npy'),p)
        joblib.dump(dict(model=model,context=transformer,preprocessor=preprocessor,columns=list(X.columns),
            categorical=state['categorical'],feature_plan=state['feature_plan'],classes=classes),directory/(name+'.joblib'))
        if scaler is not None:joblib.dump(dict(scaler=scaler,columns=numeric),directory/'numeric_scaler.joblib')
        records.append(dict(reference=name,base_fits=2 if member.get('hierarchy') else 1))
    p=sum(predictions)/len(predictions);np.save(directory/'prediction.npy',p)
    atomic_json(directory/'result.json',dict(status='completed',seconds=time.monotonic()-started,records=records,
        train_rows=len(train),query_rows=len(evaluate),control_used=fold=='audit'))


def jobs(out,kind,candidates,folds,deadline):
    import psutil
    env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','TMPDIR'}}
    env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    completed=[]
    for candidate in candidates:
        successful=True
        for fold in folds:
            job=Path(out)/kind/(candidate+'_cv'+fold if fold!='audit' else candidate+'_audit')
            if job.exists():
                if (job/'result.json').exists() and (job/'prediction.npy').exists():
                    if json.loads((job/'result.json').read_text()).get('status')=='completed':continue
                successful=False;break
            job.mkdir(parents=True,exist_ok=False)
            if time.monotonic()>=deadline:
                atomic_json(job/'failure.json',dict(reason='budget_too_small'))
                successful=False;break
            with (job/'worker.log').open('w') as log:
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--output',str(out),
                    '--worker',kind,candidate,fold],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                start=time.monotonic();peak=0
                while process.poll() is None:
                    try:
                        tree=psutil.Process(process.pid)
                        # These native estimators use threads, not child workers.
                        # macOS sandbox forbids enumerating unrelated processes.
                        peak=max(peak,tree.memory_info().rss)
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                    if time.monotonic()>=deadline or time.monotonic()-start>=180 or peak>4*1024**3:
                        process.kill();process.wait()
                        atomic_json(job/'failure.json',dict(reason='resource_limit',peak_bytes=peak))
                        successful=False;break
                    time.sleep(.2)
            if process.returncode:
                if not (job/'failure.json').exists():atomic_json(job/'failure.json',dict(reason='technical_failure',returncode=process.returncode))
                successful=False;break
            print(kind,candidate,fold,'complete',round(time.monotonic()-start,2),flush=True)
        if successful:completed.append(candidate)
    return completed


def main(out,seconds=1200,extended=False):
    import joblib
    from tools.plan_development_benchmark import tasks
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);archive(out,create=True)
    from tools.reference_features import macro_f1,fit_offsets
    datasets=tasks();configs={}
    for kind in ['regression','classification']:
        source,names=locations(kind);members=[]
        for name in names:
            state=joblib.load(source/(name+'_audit')/'model.joblib')
            member=dict(reference=name,spec=state['spec'])
            member['spec']['seed']=7
            if kind=='classification':
                member['weighted']=name=='current_weighted';member['hierarchy']=name=='hierarchy_balanced'
            members.append(member)
        context=json.loads(json.dumps(members[0]));context['spec']['seed']=42;context['context']=True
        dart=dict(reference=names[0],spec=dict(model='lightgbm',seed=42,hyperparameters=dict(
            boosting_type='dart',n_estimators=600,num_leaves=31,learning_rate=.05,reg_lambda=10,
            min_child_samples=30,drop_rate=.05,skip_drop=.5)),weighted=kind=='classification')
        configs[kind]={'context':[context],'seed7':members,'dart':[dart]}
        if kind=='regression':
            context_cb=json.loads(json.dumps(members[1]));context_cb['spec']['seed']=42;context_cb['context']=True
            configs[kind]['context_catboost']=[context_cb]
        else:
            configs[kind]['native_catboost']=[dict(reference=names[0],spec=dict(model='catboost',seed=42,
                hyperparameters=dict(iterations=700,depth=7,learning_rate=.06,l2_leaf_reg=10,
                    auto_class_weights='SqrtBalanced')))]
        if extended:
            if kind=='regression':
                deep=json.loads(json.dumps(members[1]));deep['spec']['seed']=42
                deep['spec']['hyperparameters'].update(iterations=1200,depth=10,l2_leaf_reg=15)
                ordered=json.loads(json.dumps(members[1]));ordered['spec']['seed']=42
                ordered['spec']['hyperparameters']['boosting_type']='Ordered'
                uncertainty=json.loads(json.dumps(members[1]));uncertainty['spec']['seed']=42
                uncertainty['spec']['hyperparameters']['loss_function']='RMSEWithUncertainty'
                uncertainty['prediction_column']=0
                linear=json.loads(json.dumps(members[0]));linear['spec']['seed']=42
                linear['spec']['hyperparameters'].update(linear_tree=True,linear_lambda=10,
                    n_estimators=700,num_leaves=15,min_child_samples=100)
                linear['scale_numeric']=True
                configs[kind]=dict(deep_catboost=[deep],ordered_catboost=[ordered],linear_leaves=[linear],uncertainty_mean=[uncertainty])
            else:
                deep=json.loads(json.dumps(members[0]));deep['spec']['seed']=42
                deep['spec']['hyperparameters'].update(n_estimators=800,max_depth=7,min_child_weight=3,reg_lambda=10)
                pair=dict(reference=names[0],pair_refinement=True)
                configs[kind]=dict(deep_xgboost=[deep],pair_specialists=[pair])
    from importlib.metadata import version
    for kind,d in datasets.items():
        source,_=locations(kind)
        historical=json.loads((source/'protocol.json').read_text())
        actual={name:file_hash(d.task_path.parent/name) for name in historical['source_hashes']}
        if actual!=historical['source_hashes']:raise ValueError('Historical baseline inputs changed')
    atomic_json(out/'protocol.json',dict(format=1,reference_revision=REFERENCE,candidates=configs,
        data_hashes={k:d.source_hash for k,d in datasets.items()},
        code_sha256=file_hash(Path(__file__)),context_sha256=file_hash(ROOT/'automl_agent/group_context.py'),
        source_hashes={p.name:file_hash(p) for p in (out/'reference_sources').glob('*.py')},
        comparison='historical strongest saved pair on identical saved splits',api_calls=0,
        weights=[0,.2,.4,1],tau=.5,previously_viewed_audit=True,automatic_promotion=False,
        baseline_scores=dict(regression=39.03614722027966,classification=.6565257846316546),
        wall_seconds=seconds,cpu_requested=2,legacy_hierarchy_cpu=4,memory_mb=4096,timeout_per_job=180,
        extended=extended,versions={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','catboost','xgboost']},
        baseline_split_hashes={k:file_hash(locations(k)[0]/'splits.npz') for k in datasets},
        baseline_cv_policy='fixed_rounds'))
    shutil.copyfile(Path(__file__),out/'source_snapshot.py')
    deadline=time.monotonic()+seconds
    evaluate(out,datasets,configs,deadline)


def evaluate(out,datasets,configs,deadline):
    from tools.reference_features import macro_f1,fit_offsets
    protocol=json.loads((Path(out)/'protocol.json').read_text())
    for kind,d in datasets.items():
        completed=jobs(out,kind,list(configs[kind]),['0','1'],deadline)
        source,names=locations(kind);split=np.load(source/'splits.npz');target=d.task.targets[0]
        classes=sorted(d.task.classes[target]) if kind=='classification' else None
        y=pd.Categorical(d.train[target],categories=classes).codes if classes else d.train[target].to_numpy()
        entries=[]
        for candidate in ['baseline',*completed]:
            for weight in ([0] if candidate=='baseline' else [.2,.4,1]):
                scores=[];offsets=[]
                for i in ['0','1']:
                    rows=split['eval'+i];fit=split['fit'+i]
                    counts=np.bincount(y[fit],minlength=len(classes)) if classes else None
                    ref=baseline(source,names,'cv'+i,counts,
                        fixed_rounds=protocol.get('baseline_cv_policy')=='fixed_rounds')
                    p=ref if weight==0 else np.load(out/kind/(candidate+'_cv'+i)/'prediction.npy')
                    p=weight*p+(1-weight)*ref
                    if classes:
                        calibration=np.isin(rows,split['calibration'+i]);selection=~calibration
                        bias=fit_offsets(p[calibration],y[rows][calibration])
                        score=macro_f1(y[rows][selection],(np.log(np.maximum(p[selection],1e-15))+bias).argmax(axis=1),len(classes))
                        offsets.append(bias.tolist())
                    else:score=float(np.sqrt(np.mean((y[rows]-p)**2)))
                    scores.append(score)
                entries.append(dict(candidate=candidate,weight=weight,scores=scores,
                    score=float(np.mean(scores)),offsets=offsets))
        entries.sort(key=lambda r:(-1 if classes else 1)*r['score'])
        chosen=entries[0]
        atomic_json(out/kind/'selection.json',dict(chosen=chosen,entries=entries,sealed_before_audit=True,
            excluded=[name for name in configs[kind] if name not in completed]))
        print('SELECTION',kind,chosen,flush=True)
        # Only a development-selected change proceeds to the reused audit.
        if chosen['weight']==0:
            atomic_json(out/kind/'summary.json',dict(status='baseline_retained',selection=chosen));continue
        finished=jobs(out,kind,[chosen['candidate']],['audit'],deadline)
        if not finished:
            atomic_json(out/kind/'summary.json',dict(status='audit_incomplete',selection=chosen,baseline_retained=True))
            continue
        rows=split['control'];fit=split['development']
        counts=np.bincount(y[fit],minlength=len(classes)) if classes else None
        ref=baseline(source,names,'audit',counts,d.train.iloc[rows],kind)
        p=np.load(out/kind/(chosen['candidate']+'_audit')/'prediction.npy')
        combined=(1-chosen['weight'])*ref+chosen['weight']*p
        if classes:
            # Exact accepted policy for the unchanged reference, independent
            # frozen crossfit policy for the newly selected compound.
            old=json.loads((source/'selection.json').read_text())
            left=(np.log(np.maximum(ref,1e-15))+np.mean(old['learned_offsets'],axis=0)).argmax(axis=1)
            right=(np.log(np.maximum(combined,1e-15))+np.mean(chosen['offsets'],axis=0)).argmax(axis=1)
            a=macro_f1(y[rows],left,len(classes));b=macro_f1(y[rows],right,len(classes))
        else:a=float(np.sqrt(np.mean((y[rows]-ref)**2)));b=float(np.sqrt(np.mean((y[rows]-combined)**2)))
        expected=json.loads((out/'protocol.json').read_text())['baseline_scores'][kind]
        if not np.isclose(a,expected,rtol=1e-8):raise RuntimeError('Historical baseline did not reproduce')
        np.save(out/kind/'candidate_audit.npy',combined)
        atomic_json(out/kind/'summary.json',dict(status='completed',baseline=a,candidate=b,
            signed_improvement=b-a if classes else a-b,selection=chosen,
            reused_audit=True,official_score_unknown=True,csv_changed=False,live_agent_run=False))
        print('AUDIT',kind,a,b,flush=True)


def resume(out,seconds=1200):
    from tools.plan_development_benchmark import tasks
    out=Path(out).resolve();q=json.loads((out/'protocol.json').read_text());archive(out)
    datasets=tasks()
    if {k:d.source_hash for k,d in datasets.items()}!=q['data_hashes']:
        raise ValueError('Cannot resume after changing data')
    atomic_json(out/'resume.json',dict(code_sha256=file_hash(Path(__file__)),previous_code_sha256=q['code_sha256'],
        reason='Preserve completed work and skip optional failed candidates without retraining or changing recipes'))
    evaluate(out,datasets,q['candidates'],time.monotonic()+seconds)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=1200);p.add_argument('--worker',nargs=3)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--extended',action='store_true')
    args=p.parse_args()
    from tools.strong_reference_study import main,worker,resume
    if args.worker:worker(args.output,*args.worker)
    elif args.resume:resume(args.output,args.seconds)
    else:main(args.output,args.seconds,args.extended)
