"""Frozen CPU/time-budget comparison: strongest fixed trees, RealMLP-TD-S, mixture.

All training, honest OOF epoch selection, thresholds, serialization and inference
are charged to an independent fresh-process arm. No external labels or APIs.
"""
from __future__ import annotations
import argparse,json,os,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib,numpy as np,pandas as pd,psutil
from sklearn.model_selection import GroupShuffleSplit
from automl_agent.util import atomic_json,file_hash,digest
from automl_agent.identity import feature_groups
from automl_agent.executor import terminate_tree
from tools.pytorch_agent_study import prepare,study_inner,Classical,ResidualHalf,metric,references,load_dataset
from tools.realmlp_td_s import RealMLPTDS


def freeze(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'tools/realmlp_td_s.py',ROOT/'tools/pytorch_agent_study.py']
    sources += list((ROOT/'research/pytorch_agent_20261026/reference_templates').glob('*'))
    sources += list((ROOT/'research/pytorch_agent_20261026/reference_sources').glob('*.py'))
    q=dict(recipe='RealMLP-TD-S Table A.1/A.2, author defaults; no HPO',author_commit='c126ea51187c5080b91f28d352481dbd3b2194b0',
        tasks=['regression','classification','adult','bank','bike','har'],arms=['trees','neural','mixture'],phases=['development','control'],
        seed=42,threads=1,arm_wall_seconds=900,memory_mb=4096,epochs=256,batch_size=256,
        equal_budget='same upper bound for each complete arm, not equal artificial sleep/consumption; unused time is reported; no HPO in any arm',
        weights=[0,.1,.25,.5,.75,1],epoch_selection='last best own-inner classification error or MSE; complete 256-epoch selection, frozen replay with original 256-epoch LR horizon',
        classification_decision='same honest OOF calibration/selection split for argmax vs offsets, then fit selected offsets on full OOF calibration; mixture weight selected on inner selection only',
        regression_reference='strongest saved LightGBM/CatBoost recipe, original categorical half-residual calibration from every available earlier forward origin',
        numeric_missing_adaptation='fit-only median before author preprocessing; official interface rejects continuous NaN; no new indicators',
        control_status={'regression':'previously viewed in project','classification':'previously viewed in project','adult':'viewed by preceding frozen candidate evaluation','bank':'viewed by preceding frozen candidate evaluation','bike':'viewed by preceding frozen candidate evaluation','har':'not evaluated by this chat; may have been viewed by other project work'},
        independent_confirmations=False,api_calls=0,production_expansion=False,
        source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in sources},created_unix=time.time())
    q['data_hashes']={name:load_dataset(name).source_hash for name in q['tasks']}
    atomic_json(out/'freeze.json',q);return out


def worker(request):
    started=time.monotonic();r=json.loads(Path(request).read_text());out=Path(request).parent
    q=json.loads(Path(r['freeze']).read_text());name,arm,phase=r['task'],r['arm'],r['phase']
    deadline=started+q['arm_wall_seconds'];refs=references();fits=[];decisions=[]
    for p,h in q['source_hashes'].items():
        if file_hash(ROOT/p)!=h:raise ValueError('Recipe source changed after freeze: '+p)
    d,X,y,classes,templates,task,strategy,cols,cats=prepare(name,out)
    if d.source_hash!=q['data_hashes'][name]:raise ValueError('Frozen input changed')
    split=np.load(out/'splits.npz');train=split['fit'] if phase=='development' else np.sort(np.r_[split['fit'],split['validation']]);evaluate=split['validation'] if phase=='development' else split['control']
    if set(train)&set(evaluate):raise ValueError('Fit/evaluation overlap')
    target=task.targets[0]
    a,b=study_inner(d.train.iloc[train],pd.DataFrame({target:y[train]},index=d.train.iloc[train].index),task,strategy,42);a,b=train[a],train[b]
    na,nb=study_inner(d.train.iloc[a],pd.DataFrame({target:y[a]},index=d.train.iloc[a].index),task,strategy,42);na,nb=a[na],a[nb]
    np.savez_compressed(out/'charged_splits.npz',train=train,evaluation=evaluate,oof_fit=a,oof_calibration=b,nested_fit=na,nested_stop=nb)
    atomic_json(out/'charged_protocol.json',dict(task=name,arm=arm,phase=phase,freeze_sha256=file_hash(Path(r['freeze'])),data_hash=d.source_hash,
        split_sha256=file_hash(out/'charged_splits.npz'),columns=cols,categorical=cats,classes=classes,raw_transform='fit-only median; all-category one-hot/binary; median/IQR/range fallback; smooth clip',
        seed=q['seed'],all_controls_previously_viewed_or_uncertain=q['control_status'][name],total_arm_seconds=q['arm_wall_seconds'],cpu_threads=1,memory_mb=q['memory_mb']))
    def remaining():
        left=deadline-time.monotonic()
        if left<=0:raise TimeoutError('Complete arm deadline exhausted')
        return left
    def tree(rows,query,label):
        begin=time.monotonic();model=Classical(templates,classes,q['seed']).fit(X.iloc[rows],y[rows]);p=model.predict(X.iloc[query])
        fits.append(dict(component='trees',label=label,base_fits=model.base_fits,train_rows=len(rows),query_rows=len(query),seconds=time.monotonic()-begin,members=model.fit_records))
        path=out/(label+'.joblib');joblib.dump(model,path)
        np.testing.assert_allclose(joblib.load(path).predict(X.iloc[query]),p,rtol=1e-6)
        remaining();return p
    def neural(rows,query,stop_a,stop_b,label):
        begin=time.monotonic();select=RealMLPTDS(cols,cats,classes,q['seed'],epochs=256,seconds=remaining())
        select.fit(X.iloc[stop_a],d.train.iloc[stop_a][target],(X.iloc[stop_b],d.train.iloc[stop_b][target]));epoch=select.best_epoch
        fits.append(dict(component='realmlp',label=label+'_select',base_fits=1,train_rows=len(stop_a),stop_rows=len(stop_b),seconds=time.monotonic()-begin,selected_epoch=epoch,actual_epochs=select.epochs_run,curve=select.curve))
        begin=time.monotonic();model=RealMLPTDS(cols,cats,classes,q['seed'],epochs=epoch,seconds=remaining()).fit(X.iloc[rows],d.train.iloc[rows][target])
        if model.epochs_run!=epoch:raise ValueError('Partial RealMLP replay')
        p=model.predict(X.iloc[query]);path=out/(label+'.joblib');joblib.dump(model,path)
        np.testing.assert_allclose(joblib.load(path).predict(X.iloc[query]),p,rtol=1e-6)
        fits.append(dict(component='realmlp',label=label+'_replay',base_fits=1,train_rows=len(rows),query_rows=len(query),seconds=time.monotonic()-begin,selected_epoch=epoch,actual_epochs=model.epochs_run,lr_horizon=256))
        remaining();return p
    # All decisions share the same fit-only held-out calibration and selection units.
    if classes:
        groups=d.train.iloc[b][task.group_column].to_numpy() if task.group_column else feature_groups(d.train.iloc[b][task.feature_columns])
        c,sel=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=303).split(X.iloc[b],groups=groups))
        if set(groups[c])&set(groups[sel]):raise ValueError('Decision group overlap')
        def decision(p,label):
            bias=refs.fit_offsets(p[c],y[b][c]);decisions.append(dict(label=label,stage='choose_offsets',rows=len(c)))
            raw=metric(y[b][sel],p[sel],classes);adjusted=metric(y[b][sel],p[sel],classes,bias)
            use=adjusted>raw
            return (max(raw,adjusted),use)
        def refit_decision(p,use,label):
            if not use:return np.zeros(len(classes))
            decisions.append(dict(label=label,stage='refit_offsets',rows=len(b)));return refs.fit_offsets(p,y[b])
    tree_cp=tree_vp=None
    if arm in {'trees','mixture'}:
        if name=='regression':
            from automl_agent.validation import ordered_periods
            periods=ordered_periods(d.train[task.time_column]);unique=sorted(periods.iloc[train].unique());origins=[];predictions=[]
            for i,period in enumerate(unique[1:],1):
                rows=train[periods.iloc[train].to_numpy()<period];query=train[periods.iloc[train].to_numpy()==period]
                pred=tree(rows,query,'forward_'+str(i));origins.append(query);predictions.append(pred)
            positions={int(row):float(value) for rows,pred in zip(origins,predictions) for row,value in zip(rows,pred)}
            raw_cp=np.asarray([positions[int(row)] for row in b]);before=[i for i,rows in enumerate(origins) if not set(rows)&set(b)]
            prior=np.concatenate([origins[i] for i in before]);prior_p=np.concatenate([predictions[i] for i in before])
            calibrator_started=time.monotonic()
            tree_cp=ResidualHalf(task.categorical_columns).fit(d.train.iloc[prior],y[prior],prior_p).predict(d.train.iloc[b],raw_cp)
            calibration_seconds=time.monotonic()-calibrator_started
            raw_vp=tree(train,evaluate,'trees_final');all_rows=np.concatenate(origins);all_p=np.concatenate(predictions)
            calibrator_started=time.monotonic()
            tree_vp=ResidualHalf(task.categorical_columns).fit(d.train.iloc[all_rows],y[all_rows],all_p).predict(d.train.iloc[evaluate],raw_vp)
            fits.append(dict(component='residual_half',base_fits=2,train_rows=[len(prior),len(all_rows)],parameters={'shrinkage':100,'strength':.5},seconds=calibration_seconds+time.monotonic()-calibrator_started))
        else:tree_cp=tree(a,b,'trees_oof');tree_vp=tree(train,evaluate,'trees_final')
        np.save(out/'trees_oof.npy',tree_cp);np.save(out/'trees_evaluation.npy',tree_vp)
    net_cp=net_vp=None
    if arm in {'neural','mixture'}:
        if classes or arm=='mixture':net_cp=neural(a,b,na,nb,'realmlp_oof');np.save(out/'realmlp_oof.npy',net_cp)
        if arm=='neural':net_vp=neural(train,evaluate,a,b,'realmlp_final')
    if arm=='mixture':
        choices=[]
        for w in q['weights']:
            p=(1-w)*tree_cp+w*net_cp
            score,use=decision(p,'mixture_'+str(w)) if classes else (-metric(y[b],p,None),False)
            choices.append((score,-w,w,use))
        _,_,weight,use=max(choices,key=lambda t:t[:2]);cal=(1-weight)*tree_cp+weight*net_cp
        bias=refit_decision(cal,use,'mixture') if classes else None
        if weight:net_vp=neural(train,evaluate,a,b,'realmlp_final')
        prediction=tree_vp if weight==0 else (1-weight)*tree_vp+weight*net_vp
    else:
        weight=None;cal=tree_cp if arm=='trees' else net_cp
        if classes:
            _,use=decision(cal,arm);bias=refit_decision(cal,use,arm)
        else:bias=None
        prediction=tree_vp if arm=='trees' else net_vp
    np.save(out/'prediction.npy',prediction)
    value=metric(y[evaluate],prediction,classes,bias);raw_score=metric(y[evaluate],prediction,classes)
    result=dict(status='completed',task=name,arm=arm,phase=phase,score=value,raw_score=raw_score,weight=weight,offsets=bias.tolist() if classes else None,
        complete_runtime_seconds=time.monotonic()-started,base_fits=sum(f['base_fits'] for f in fits),postprocessing_fits=len(decisions),fits=fits,decisions=decisions,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        cpu_threads=1,arm_time_limit=q['arm_wall_seconds'],unused_budget_seconds=remaining(),control_labels_used_for_fit=False,external_test_labels_read=False,api_calls=0)
    atomic_json(out/'result.json',result);print(name,phase,arm,value,round(result['complete_runtime_seconds'],2),flush=True)


def run(frozen,tasks,phases):
    frozen=Path(frozen).resolve();q=json.loads(frozen.read_text());root=frozen.parent
    for phase in phases:
        for name in tasks:
            for arm in q['arms']:
                out=root/phase/name/arm;out.mkdir(parents=True,exist_ok=False)
                request=out/'request.json';atomic_json(request,dict(freeze=str(frozen),task=name,arm=arm,phase=phase))
                env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','TMPDIR'}}
                env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/private/tmp/pytorch-mpl')
                begin=time.monotonic();peak=0;stop=None;cpu=0.
                with (out/'worker.log').open('w') as log:
                    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(request)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        while process.poll() is None:
                            try:
                                p=psutil.Process(process.pid);peak=max(peak,p.memory_info().rss);times=p.cpu_times();cpu=times.user+times.system
                            except psutil.NoSuchProcess:break
                            if peak>q['memory_mb']*1024**2:stop='memory_limit'
                            if time.monotonic()-begin>=q['arm_wall_seconds']:stop='timeout'
                            if stop:terminate_tree(process);break
                            time.sleep(.1)
                        process.wait()
                    finally:
                        if process.poll() is None:terminate_tree(process)
                atomic_json(out/'cost.json',dict(returncode=process.returncode,stop=stop,wall_seconds=time.monotonic()-begin,cpu_seconds=cpu,peak_rss_bytes=peak,limit_seconds=q['arm_wall_seconds'],cpu_threads=1,memory_mb=q['memory_mb'],api_calls=0))
                print(phase,name,arm,process.returncode,stop,round(time.monotonic()-begin,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-new',type=Path);p.add_argument('--frozen',type=Path);p.add_argument('--worker',type=Path);p.add_argument('--tasks',nargs='+',default=['regression','classification']);p.add_argument('--phases',nargs='+',default=['development','control']);a=p.parse_args()
    if a.worker:
        try:worker(a.worker)
        except BaseException as error:
            atomic_json(a.worker.parent/'failure.json',dict(type=type(error).__name__,error=str(error),api_calls=0));raise
    elif a.freeze_new:freeze(a.freeze_new)
    else:run(a.frozen,a.tasks,a.phases)
