"""Frozen offline native-class work probe, never a quality benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')


def child(args):
    import cloudpickle
    import numpy as np
    import pandas as pd
    from autogluon.tabular.models.lgb.lgb_model import LGBModel
    from autogluon.tabular.models.catboost.catboost_model import CatBoostModel
    from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel
    rng=np.random.default_rng(42)
    X=pd.DataFrame(rng.normal(size=(10000,12)),columns=['x'+str(i) for i in range(12)])
    y=pd.Series(np.arange(len(X))%args.classes)
    cls={'GBM':LGBModel,'CAT':CatBoostModel,'XGB':XGBoostModel}[args.family]
    parameters={'GBM':{'num_boost_round':32,'num_leaves':15,'min_data_in_leaf':5,'seed_value':42},
        'CAT':{'iterations':32,'depth':4,'random_seed':42},
        'XGB':{'n_estimators':32,'max_depth':4,'random_state':42}}[args.family]
    model=cls(path=str(args.output/'model'),name='fixed',problem_type='binary' if args.classes==2 else 'multiclass',
        eval_metric='f1_macro',hyperparameters=parameters)
    cpu=time.process_time();start=time.monotonic()
    model.fit(X=X,y=y,num_cpus=1,num_gpus=0,time_limit=90,verbosity=0)
    fit_seconds=time.monotonic()-start;fit_cpu=time.process_time()-cpu
    start=time.monotonic();p=model.predict_proba(X.iloc[:1000],num_cpus=1);prediction_seconds=time.monotonic()-start
    start=time.monotonic();blob=cloudpickle.dumps(model);serialization_seconds=time.monotonic()-start
    if args.family=='GBM':
        rounds=model.model.current_iteration();units=model.model.num_trees()/rounds;unit_kind='trees_per_round'
    elif args.family=='XGB':
        booster=model.model.get_booster();rounds=booster.num_boosted_rounds();units=len(booster.get_dump())/rounds;unit_kind='trees_per_round'
    else:
        rounds=model.model.tree_count_;units=len(model.model.get_leaf_values())/sum(model.model.get_tree_leaf_counts());unit_kind='leaf_value_components'
    assert rounds==32 and len(p)==1000
    save(args.output/'result.json',dict(family=args.family,classes=args.classes,rows=len(X),features=len(X.columns),
        fit_seconds=fit_seconds,fit_cpu_seconds=fit_cpu,prediction_seconds=prediction_seconds,
        serialization_seconds=serialization_seconds,model_bytes=len(blob),rounds=rounds,output_work_units=units,unit_kind=unit_kind,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        X_sha256=hashlib.sha256(X.to_numpy().tobytes()).hexdigest(),y_sha256=hashlib.sha256(y.to_numpy().tobytes()).hexdigest()))


def run(args):
    import importlib.metadata as md
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    cases=[dict(family=f,classes=k,repeat=r) for f in ['GBM','CAT','XGB'] for k in [2,3,10] for r in range(2)]
    random.Random(42).shuffle(cases)
    watched=[ROOT/'automl_agent/cost_model.py',ROOT/'automl_agent/search.py',Path(__file__)]
    source_sha={str(p.relative_to(ROOT)):sha(p) for p in watched}
    freeze=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=source_sha,cases=cases,rows=10000,features=12,rounds=32,cpu=1,seed=42,
        api_calls=0,quality_benchmark=False,scope='fixed native work; fresh process per case; repetitions do not constitute independent task confirmation',
        versions={n:md.version(n) for n in ['autogluon.tabular','autogluon.core','numpy','pandas','lightgbm','catboost','xgboost']})
    save(args.output/'freeze.json',freeze)
    results=[]
    for i,case in enumerate(cases):
        directory=args.output/f"{i:02}_{case['family']}_{case['classes']}_{case['repeat']}";directory.mkdir()
        for name in ['home','tmp','mpl']:(directory/name).mkdir()
        environment={'PATH':os.defpath,'HOME':str(directory/'home'),'TMPDIR':str(directory/'tmp'),
            'MPLCONFIGDIR':str(directory/'mpl'),'PYTHONPATH':str(args.packages.resolve())+os.pathsep+str(ROOT),
            **{k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}}
        start=time.monotonic()
        with (directory/'run.log').open('wb') as stream:
            completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child','--family',case['family'],
                '--classes',str(case['classes']),'--output',str(directory)],cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        if completed.returncode:raise RuntimeError('Native probe failed: '+str(directory))
        result=json.loads((directory/'result.json').read_text());result.update(repeat=case['repeat'],process_wall_seconds=time.monotonic()-start)
        results.append(result);save(args.output/'results.json',results)
        print(json.dumps(dict(completed=i+1,total=len(cases),family=case['family'],classes=case['classes'],fit_seconds=round(result['fit_seconds'],3))),flush=True)
    assert all(sha(ROOT/name)==value for name,value in source_sha.items()),'Sources changed during frozen probe'
    save(args.output/'terminal.json',dict(status='completed',cases=len(results),sources_unchanged=True,api_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--child',action='store_true')
    p.add_argument('--family',choices=['GBM','CAT','XGB']);p.add_argument('--classes',type=int)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--packages',type=Path)
    args=p.parse_args();child(args) if args.child else run(args)
