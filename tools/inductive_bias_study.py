"""Offline fixed-recipe comparison; no Qwen, downloads, tuning or promotion.

Each model sees identical folds. Dataset names select development fixtures only,
not production routing. This does not evaluate autonomous model selection.
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


def model(name,classification,rows,columns):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression,Ridge
    from sklearn.svm import SVC,SVR
    from sklearn.kernel_approximation import Nystroem
    from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
    from sklearn.compose import TransformedTargetRegressor
    from automl_agent.pipelines import make_estimator
    from automl_agent.contracts import PipelineSpec
    linear=lambda:LogisticRegression(C=1,max_iter=2000) if classification else Ridge(alpha=10)
    if name=='linear':estimator=make_pipeline(StandardScaler(),linear())
    elif name=='rbf':estimator=make_pipeline(StandardScaler(),SVC(C=10,gamma='scale',cache_size=128,probability=False)
        if classification else SVR(C=10,gamma='scale',epsilon=.1,cache_size=128))
    elif name=='nystroem':estimator=make_pipeline(StandardScaler(),
        Nystroem(gamma=1/columns,n_components=min(128,max(1,rows//2)),random_state=42),linear())
    elif name=='extra_trees':estimator=(ExtraTreesClassifier if classification else ExtraTreesRegressor)(
        n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=1)
    elif name=='lightgbm':estimator=make_estimator(PipelineSpec(model='lightgbm',hyperparameters=dict(
        n_estimators=200,num_leaves=15,learning_rate=.05,reg_lambda=5,min_child_samples=10)),classification,{'cpu':1})
    else:raise ValueError('Unknown fixed recipe')
    return estimator if classification else TransformedTargetRegressor(regressor=estimator,transformer=StandardScaler())


def worker(name,out):
    import resource
    import numpy as np
    from sklearn.datasets import load_wine,load_digits,load_diabetes,make_friedman1
    from sklearn.model_selection import StratifiedKFold,KFold
    from sklearn.metrics import f1_score,mean_squared_error
    from threadpoolctl import threadpool_limits
    from automl_agent.util import atomic_json,digest
    classification=name in {'wine','digits'}
    if name=='friedman':X,y=make_friedman1(n_samples=1500,n_features=20,noise=1,random_state=42)
    else:X,y={'wine':load_wine,'digits':load_digits,'diabetes':load_diabetes}[name](return_X_y=True)
    splitter=StratifiedKFold(3,shuffle=True,random_state=42) if classification else KFold(3,shuffle=True,random_state=42)
    folds=list(splitter.split(X,y));results=[];predictions={}
    with threadpool_limits(limits=1):
        for family in ['linear','rbf','nystroem','extra_trees','lightgbm']:
            scores=[];seconds=[];p=np.empty(len(y),dtype=y.dtype if classification else float)
            for fit,query in folds:
                estimator=model(family,classification,len(fit),X.shape[1]);started=time.monotonic()
                estimator.fit(X[fit],y[fit]);values=estimator.predict(X[query]);seconds.append(time.monotonic()-started)
                if not np.isfinite(values).all():raise ValueError('Invalid prediction')
                p[query]=values;scores.append(float(f1_score(y[query],values,average='macro')) if classification
                    else float(mean_squared_error(y[query],values)**.5))
            predictions[family]=p
            results.append(dict(model=family,fold_scores=scores,mean=float(np.mean(scores)),fit_predict_seconds=seconds))
    joint={}
    if classification:
        a=predictions['rbf']==y;b=predictions['lightgbm']==y
        joint=dict(rbf_correct_tree_wrong=int(np.sum(a&~b)),tree_correct_rbf_wrong=int(np.sum(b&~a)),
            both_wrong=int(np.sum(~a&~b)),disagree=int(np.sum(predictions['rbf']!=predictions['lightgbm'])),
            ensemble_gain_claimed=False)
    else:
        p=(predictions['rbf']+predictions['lightgbm'])/2
        joint=dict(uniform_mix_oof_rmse=float(mean_squared_error(y,p)**.5),weights_learned=False)
    atomic_json(Path(out)/(name+'.json'),dict(dataset=name,classification=classification,rows=len(y),columns=X.shape[1],
        results=results,joint=joint,folds_hash=digest([[a.tolist(),b.tolist()] for a,b in folds]),
        data_hash=digest([X.tolist(),y.tolist()]),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        metric='macro_f1' if classification else 'rmse',api_calls=0,agent_test=False,
        scope='fixed recipes, IID 3-fold exploratory comparison; not tuned best models or fresh contest evidence'))


def run(out):
    from importlib.metadata import version
    from automl_agent.util import atomic_json,file_hash
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'protocol.json',dict(source_sha256=file_hash(Path(__file__)),cases=['wine','digits','diabetes','friedman'],
        models=['linear','rbf','nystroem','extra_trees','lightgbm'],seed=42,folds=3,no_tuning=True,
        target_standardized_inside_fit_for_all_regressors=True,svc_probability=False,threads=1,
        versions={p:version(p) for p in ['numpy','scikit-learn','lightgbm']},api_calls=0,
        old_wdbc_holdout_not_used=True,outer_evaluation_not_used_for_recipe_selection=True))
    env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','TMPDIR'}}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for name in ['wine','digits','diabetes','friedman']:
        with (out/(name+'.log')).open('w') as log:
            r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',name,'--output',str(out)],
                env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if r.returncode:raise RuntimeError('Study failed: '+name)
        result=json.loads((out/(name+'.json')).read_text())
        print(name,[(r['model'],round(r['mean'],5)) for r in result['results']],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',choices=['wine','digits','diabetes','friedman']);args=p.parse_args()
    if args.worker:worker(args.worker,args.output)
    else:run(args.output)
