"""Offline nested tabular DL study. No API calls, test labels or promotion.

Only the benchmark loader knows dataset locations and historical reference roles.
All learners and transforms operate on passed columns and arrays. A single honest
fit-only OOF holdout estimates mixing/decisions; coverage is explicitly partial.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import resource

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from automl_agent.contracts import PipelineSpec,TaskSpec
from automl_agent.data import Dataset
from automl_agent.pipelines import make_estimator
from automl_agent.validation import choose_validation,inner_split
from automl_agent.util import atomic_json,file_hash,digest
from tools.torch_research_core import TorchMLP

CONFIGS={
 'standard_mlp':dict(encoding='standard'),
 'quantile_mlp':dict(encoding='quantile'),
 'quantile_resnet':dict(encoding='quantile',residual=True),
 'learned_piecewise_mlp':dict(encoding='learned_piecewise'),
 'quantile_shared':dict(encoding='quantile',heads=4),
 'periodic_mlp':dict(encoding='periodic'),
}
WEIGHTS=[0.,.1,.25,.5,.75,1.]


def references():
    for name in ['reference_features','reference_alternatives']:
        path=ROOT/'research/pytorch_agent_20261026/reference_sources'/(name+'.py')
        spec=importlib.util.spec_from_file_location('tools.'+name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    # Historical helper's hardcoded four threads become the shared one-thread budget.
    # Do not alter its source or the original checkout.
    module.make_estimator=None
    return sys.modules['tools.reference_features']


def load_dataset(name):
    if name in {'regression','classification'}:
        from tools.plan_development_benchmark import tasks
        return tasks()[name]
    d=Dataset(ROOT/'outputs/pytorch_inputs'/name/'task.md')
    task,unresolved=d.draft()
    if name in {'bike','har'}:
        # Explicit benchmark contracts transcribed from task.md and dictionary.
        # These are data roles, never learner exceptions.
        dictionary=pd.read_csv(d.paths['data_dictionary.csv'])
        col=dictionary['column'];role=dictionary['role']
        targets=col[role.eq('target')].tolist();features=col[role.eq('feature')].tolist()
        cats=col[role.eq('feature') & dictionary['dtype'].eq('categorical')].tolist()
        task=TaskSpec(kind='classification' if name=='har' else 'regression',targets=targets,
            feature_columns=features,categorical_columns=cats,id_columns=['id'],
            time_column='dteday' if name=='bike' else None,group_column='subject' if name=='har' else None,
            service_columns=['subject'] if name=='har' else ['dteday'],
            metric={'name':'f1' if name=='har' else 'rmse'},
            output={'columns':['id',targets[0]],'prediction_type':'labels' if name=='har' else 'values'},
            classes={targets[0]:sorted(d.train[targets[0]].unique().tolist())} if name=='har' else {},source_hash=d.source_hash,
            explicit_validation='group' if name=='har' else 'temporal')
        unresolved=[]
    if unresolved or task is None:raise ValueError(str(unresolved))
    d.activate(task);return d


def metric(y,p,classes,offset=None):
    if classes:
        from tools.reference_features import macro_f1
        return macro_f1(y,(np.log(np.maximum(p,1e-15))+(0 if offset is None else offset)).argmax(1),len(classes))
    return float(np.sqrt(np.mean((y-p)**2)))


def correction(p,counts):
    p=p*np.sqrt(counts.max()/np.maximum(counts,1));return p/p.sum(1,keepdims=True)


class Classical:
    def __init__(self,templates,classes,seed):self.templates,self.classes,self.seed=templates,classes,seed
    def fit(self,X,y):
        self.models=[];self.base_fits=0
        for template in self.templates:
            state=joblib.load(template) if isinstance(template,str) else template
            spec=PipelineSpec.model_validate(dict(state['spec'],seed=self.seed))
            data=X[state['columns']].copy();cat=state['categorical'];vocab={}
            for c in cat:
                data[c]=data[c].fillna('__MISSING__').astype(str)
                vocab[c]=sorted(data[c].unique())
                if spec.model=='lightgbm':data[c]=pd.Categorical(data[c],categories=vocab[c])
            pre=clone(state['preprocessor']) if state.get('preprocessor') is not None else None
            if pre is not None:data=pre.fit_transform(data)
            kwargs={}
            if state.get('hierarchy'):
                from tools.reference_alternatives import HierarchicalLGBM
                # Override only the trusted helper's factory resource argument.
                import automl_agent.pipelines as pipelines
                original=pipelines.make_estimator
                def factory(s,c,context):return original(s,c,{'cpu':1})
                pipelines.make_estimator=factory
                try:
                    model=HierarchicalLGBM(spec.model_dump(),state['normal_index'],state['component_iterations']).fit(data,y)
                finally:pipelines.make_estimator=original
                self.base_fits+=2
            else:
                model=make_estimator(spec,bool(self.classes),{'cpu':1})
                if self.classes and state.get('weight_exponent',.5):
                    counts=np.bincount(y,minlength=len(self.classes));kwargs['sample_weight']=(counts.max()/np.maximum(counts,1))**state.get('weight_exponent',.5);kwargs['sample_weight']=kwargs['sample_weight'][y]
                if spec.model=='catboost':kwargs['cat_features']=cat
                model.fit(data,y,**kwargs);self.base_fits+=1
            self.models.append((model,pre,vocab,spec.model,state['columns'],cat))
        self.counts=np.bincount(y,minlength=len(self.classes)) if self.classes else None
        return self
    def predict(self,X):
        result=[]
        for model,pre,vocab,kind,columns,cat in self.models:
            data=X[columns].copy()
            for c in cat:
                data[c]=data[c].fillna('__MISSING__').astype(str)
                if kind=='lightgbm':data[c]=pd.Categorical(data[c],categories=vocab[c])
            if pre is not None:data=pre.transform(data)
            value=model.predict_proba(data) if self.classes else model.predict(data)
            result.append(correction(value,self.counts) if self.classes and len(self.templates)>1 else value)
        return sum(result)/len(result)


class ResidualHalf:
    """Same frozen shrinkage recipe as the strongest historical regressor."""
    def __init__(self,categorical):self.categorical=list(categorical)
    @staticmethod
    def keys(s):return ('V:'+s.astype(str)).where(s.notna(),'M:')
    def fit(self,X,y,p):
        error=np.asarray(y)-np.asarray(p);self.mean=float(error.mean());self.tables={}
        for c in self.categorical:
            t=pd.DataFrame({'key':self.keys(X[c]).to_numpy(),'error':error}).groupby('key').error.agg(['mean','count'])
            self.tables[c]=(t['mean']-self.mean)*t['count']/(t['count']+100.)
        return self
    def predict(self,X,p):
        shifts=[self.keys(X[c]).map(t).fillna(0).to_numpy() for c,t in self.tables.items()]
        return p+.5*(self.mean+(np.mean(shifts,axis=0) if shifts else np.zeros(len(X))))


def prepare(name,out):
    refs=references();d=load_dataset(name);task=d.task;target=task.targets[0]
    classes=sorted(task.classes.get(target,[])) or None
    y=pd.Categorical(d.train[target],categories=classes).codes if classes else d.train[target].to_numpy()
    if name in {'regression','classification'}:
        root=Path('/Users/semolik/Developer/alpha-hack/outputs')
        source=root/('literature_reference_20261025/regression' if name=='regression' else 'further_reference_20261025/classification')
        names=['regularized_lgb','engineered_cb'] if not classes else ['current_weighted','hierarchy_balanced']
        split=np.load(source/'splits.npz');fit=split['fit1' if not classes else 'fit0'];val=split['eval1' if not classes else 'eval0'];control=split['control']
        templates=[]
        for n in names:
            path=source/(n+'_audit/model.joblib');state=joblib.load(path)
            model=state['model'];state['source_sha256']=file_hash(path)
            if n=='hierarchy_balanced':state.update(hierarchy=True,normal_index=model.normal_index,component_iterations=model.component_iterations_)
            state['weight_exponent']=.5 if n=='current_weighted' else 0
            state.pop('model');dest=out/(n+'_template.joblib');joblib.dump(state,dest);templates.append(str(dest))
        state=joblib.load(templates[0]);X=refs.engineer(d.train,state['feature_plan'])
        columns=state['columns'];cat=state['categorical']
        strategy='temporal' if not classes else 'exact_group'
    else:
        plan=choose_validation(d,seed=42);fit=np.asarray(plan.fit);val=np.asarray(plan.validation);control=np.asarray(plan.control)
        X=d.train.copy();columns=task.feature_columns;cat=task.categorical_columns;strategy=plan.strategy
        # A small fixed strong classical portfolio; no external labels/search.
        templates=[]
        for model,params in [('lightgbm',dict(n_estimators=600,num_leaves=31,learning_rate=.04,reg_lambda=5)),
                ('catboost',dict(iterations=600,depth=7,learning_rate=.05,l2_leaf_reg=5))]:
            templates.append(dict(spec=dict(model=model,hyperparameters=params),columns=columns,categorical=cat,preprocessor=None,weight_exponent=.5))
    # Split logic refers to raw feature roles, even when reference features are added.
    a,b=inner_split(d.train.iloc[fit],pd.DataFrame({target:y[fit]},index=d.train.iloc[fit].index),task,strategy,42)
    oa,ob=fit[a],fit[b]
    na,nb=inner_split(d.train.iloc[oa],pd.DataFrame({target:y[oa]},index=d.train.iloc[oa].index),task,strategy,42)
    np.savez(out/'splits.npz',fit=fit,validation=val,control=control,oof_fit=oa,oof_calibration=ob,nested_fit=oa[na],nested_stop=oa[nb])
    protocol=dict(dataset=name,task=task.model_dump(mode='json'),data_hash=d.source_hash,columns=columns,categorical=cat,
        strategy=strategy,split_hash=file_hash(out/'splits.npz'),templates=templates,classes=classes,
        configs=CONFIGS,weights=WEIGHTS,epochs_cap=60,width=128,depth=2,learning_rate=.002,
        oof='one honest fit-only holdout with independent nested epoch selection; partial coverage',
        nested_split_seed=42,threads=1,control_previously_viewed=name in {'regression','classification'},
        test_labels_read=False,control_labels_used=False,base_commit='1c463e9',reference_commit='df36940',
        neural_core_hash=file_hash(ROOT/'tools/torch_research_core.py'),driver_hash=file_hash(Path(__file__)),
        environment_note='shared macOS host, screening only; not isolated contest-resource measurement')
    # Template manifests retain recipe/preprocessing types; no fitted model leaks.
    protocol['template_recipe']=[{k:v for k,v in (joblib.load(t) if isinstance(t,str) else t).items() if k in {'spec','columns','categorical','feature_plan','source_sha256','normal_index','component_iterations','hierarchy','weight_exponent'}} for t in templates]
    protocol['templates']=[t if isinstance(t,str) else {k:v for k,v in t.items() if k!='preprocessor'} for t in templates]
    atomic_json(out/'protocol.json',protocol)
    return d,X,y,classes,templates,task,strategy,columns,cat


def run(name,out,seed=42,candidates=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();d,X,y,classes,templates,task,strategy,columns,cat=prepare(name,out)
    split=np.load(out/'splits.npz');fit=split['fit'];val=split['validation'];a=split['oof_fit'];b=split['oof_calibration'];na=split['nested_fit'];nb=split['nested_stop']
    refs=references();results=[]
    for region,rows,query in [('oof',a,b),('full',fit,val)]:
        began=time.monotonic();model=Classical(templates,classes,seed).fit(X.iloc[rows],y[rows]);p=model.predict(X.iloc[query]);np.save(out/f'baseline_{region}.npy',p);joblib.dump(model,out/f'baseline_{region}.joblib')
        results.append(dict(model='baseline',region=region,seed=seed,seconds=time.monotonic()-began,base_fits=model.base_fits,raw_score=metric(y[query],p,classes)))
        print(name,seed,'baseline',region,results[-1]['raw_score'],flush=True)
    cp=np.load(out/'baseline_oof.npy');vp=np.load(out/'baseline_full.npy')
    if name=='regression':
        # Forward residual calibration has a separate earlier honest origin.
        # The calibration forecast itself never sees labels from its own query.
        began=time.monotonic();earlier=Classical(templates,classes,seed).fit(X.iloc[na],y[na]);pp=earlier.predict(X.iloc[nb])
        results.append(dict(model='baseline',region='earlier_forward',seed=seed,seconds=time.monotonic()-began,base_fits=earlier.base_fits))
        np.save(out/'baseline_earlier.npy',pp)
        oof_calibrator=ResidualHalf(cat).fit(X.iloc[nb],y[nb],pp)
        final_calibrator=ResidualHalf(cat).fit(X.iloc[np.r_[nb,b]],y[np.r_[nb,b]],np.r_[pp,cp])
        cp=oof_calibrator.predict(X.iloc[b],cp);vp=final_calibrator.predict(X.iloc[val],vp)
        joblib.dump(final_calibrator,out/'baseline_residual_half.joblib')
        np.save(out/'baseline_calibrated_oof.npy',cp);np.save(out/'baseline_calibrated_full.npy',vp)
    baseline_offset=refs.fit_offsets(cp,y[b]) if classes else None
    baseline_score=metric(y[val],vp,classes,baseline_offset)
    if classes:np.save(out/'baseline_offsets.npy',baseline_offset)
    for label in candidates or CONFIGS:
        config=CONFIGS[label];began=time.monotonic();fit_records=[];pred={}
        try:
            for region,train,query,inner_a,inner_b in [('oof',a,b,na,nb),('full',fit,val,a,b)]:
                selector=TorchMLP(columns,cat,classes,epochs=60,seconds=90,seed=seed,width=128,depth=2,learning_rate=.002,**config)
                selector.fit(X.iloc[inner_a],d.train.iloc[inner_a][task.targets[0]],(X.iloc[inner_b],d.train.iloc[inner_b][task.targets[0]]))
                epochs=selector.best_epoch
                model=TorchMLP(columns,cat,classes,epochs=epochs,seconds=90,seed=seed,width=128,depth=2,learning_rate=.002,**config)
                model.fit(X.iloc[train],d.train.iloc[train][task.targets[0]])
                if model.epochs_run!=epochs:raise RuntimeError('Partial refit')
                p=model.predict(X.iloc[query]);pred[region]=p;np.save(out/f'{label}_{region}.npy',p);joblib.dump(model,out/f'{label}_{region}.joblib')
                np.testing.assert_allclose(joblib.load(out/f'{label}_{region}.joblib').predict(X.iloc[query]),p,rtol=1e-6)
                fit_records.append(dict(region=region,base_fits=2,selected_epochs=epochs,actual_epochs=model.epochs_run,
                    selection_seconds=selector.seconds_used,refit_seconds=model.seconds_used,curve=selector.curve))
            choices=[]
            for w in WEIGHTS:
                blend=(1-w)*cp+w*pred['oof'];offset=refs.fit_offsets(blend,y[b]) if classes else None
                score=metric(y[b],blend,classes,offset)
                choices.append((score if classes else -score,-w,w,offset))
            _,_,weight,offset=max(choices,key=lambda x:x[:2])
            blend=(1-weight)*vp+weight*pred['full']
            standalone=metric(y[val],pred['full'],classes,refs.fit_offsets(pred['oof'],y[b]) if classes else None)
            score=metric(y[val],blend,classes,offset)
            entry=dict(model=label,seed=seed,status='completed',raw_score=metric(y[val],pred['full'],classes),standalone_score=standalone,
                baseline_score=baseline_score,blend_score=score,weight=weight,
                signed_gain=score-baseline_score if classes else baseline_score-score,seconds=time.monotonic()-began,fits=fit_records,
                offset=offset.tolist() if classes else None,base_fits=4)
            np.save(out/f'{label}_blend.npy',blend)
        except Exception as error:
            entry=dict(model=label,seed=seed,status='failed',error=str(error),seconds=time.monotonic()-began,fits=fit_records)
        results.append(entry);atomic_json(out/'results.json',dict(results=results,baseline_score=baseline_score,seconds=time.monotonic()-started,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,api_calls=0,control_opened=False))
        print(name,seed,label,{k:entry.get(k) for k in ['status','blend_score','weight','signed_gain','error']},flush=True)
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--candidates',nargs='+')
    a=p.parse_args();run(a.task,a.output,a.seed,a.candidates)
