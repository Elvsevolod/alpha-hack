"""Development-only diagnostic of fit-inner iteration selection across origins."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def early_choice(curve, patience):
    best, index=float('inf'),0
    for i, score in enumerate(curve):
        if score<best:
            best,index=float(score),i
        if i-index>=patience:
            break
    return index+1


def worker(out, profile):
    import cloudpickle
    import lightgbm as lgb
    import numpy as np
    from automl_agent.contracts import PipelineSpec
    from automl_agent.pipelines import make_estimator
    from automl_agent.preprocessing import FeatureMatrix
    from automl_agent.util import atomic_json,file_hash,digest
    from tools.pytorch_agent_study import load_dataset

    started=time.monotonic();freeze=json.loads((out/'freeze.json').read_text())
    assert all(file_hash(ROOT/p)==h for p,h in freeze['sources'].items())
    d=load_dataset('regression');assert d.source_hash==freeze['data_hash']
    directory=out/profile;directory.mkdir()
    task=d.task;target=task.targets[0];time_column=task.time_column
    split_path=ROOT/freeze['split_path'];assert file_hash(split_path)==freeze['split_hash']
    with np.load(split_path) as split:
        fit,evaluate=split['fit'],split['validation']
    assert not set(fit)&set(evaluate)
    periods=d.train.iloc[fit][time_column];unique=sorted(periods.unique().tolist())
    assert len(unique)==3 and periods.max()<d.train.iloc[evaluate][time_column].min()
    parameters=freeze['profiles'][profile]['hyperparameters'];cap=parameters['n_estimators']
    spec=PipelineSpec(model='lightgbm',hyperparameters=parameters,seed=42)

    def train(rows, query, record=False):
        X=d.features(rows.tolist());y=d.train.iloc[rows][[target]];V=d.features(query.tolist())
        processor=FeatureMatrix(spec,task,1,3072,'temporal')
        matrix=processor.fit_transform(X,y);valid=processor.transform(V)
        model=make_estimator(spec,False,dict(cpu=1));model.set_params(metric='None')
        history={}
        metric_dtype=[]
        if record:
            actual=d.train.iloc[query][target].to_numpy()
            def metric(y,p):
                if not metric_dtype:metric_dtype.append(str(y.dtype))
                return 'primary',float(np.sqrt(np.mean((y-p)**2))),False
            model.fit(matrix,y[target].to_numpy(),eval_set=[(valid,actual)],eval_names=['fit_inner'],eval_metric=metric,
                callbacks=[lgb.record_evaluation(history)])
        else:
            model.fit(matrix,y[target].to_numpy())
        assert model.n_estimators_==cap
        recorded=dict(curve=history['fit_inner']['primary'],truth_dtype=metric_dtype[0]) if record else None
        return model,processor,valid,recorded

    curves,origins=[],[]
    for end in [1,2]:
        rows=fit[np.flatnonzero(periods.isin(unique[:end]).to_numpy())]
        query=fit[np.flatnonzero(periods.eq(unique[end]).to_numpy())]
        model,processor,valid,recorded=train(rows,query,record=True)
        curve=recorded['curve']
        assert len(curve)==cap and np.isfinite(curve).all()
        chosen=early_choice(curve,freeze['patience'])
        p=model.predict(valid,num_iteration=chosen)
        actual=d.train.iloc[query][target].to_numpy()
        score=float(np.sqrt(np.mean((actual.astype(recorded['truth_dtype'])-p)**2)))
        np.testing.assert_allclose(score,curve[chosen-1],rtol=0,atol=1e-10)
        curves.append(np.asarray(curve));origins.append(dict(fit_rows=len(rows),evaluation_rows=len(query),
            fit_periods=unique[:end],evaluation_period=unique[end],early_iterations=chosen,
            fit_positions_hash=digest(rows.tolist()),callback_truth_dtype=recorded['truth_dtype'],
            callback_vs_original_rmse_delta=score-float(np.sqrt(np.mean((actual-p)**2)))))
        del model,processor,valid;gc.collect()
    pooled=np.sqrt(sum(row['evaluation_rows']*curve**2 for row,curve in zip(origins,curves))/sum(row['evaluation_rows'] for row in origins))
    choices=dict(latest_origin_early=origins[-1]['early_iterations'],
        pooled_forward_mse=int(pooled.argmin())+1,declared_cap=cap)
    # Freeze all choices before fitting/scoring the outer development period.
    atomic_json(directory/'selection.json',dict(choices=choices,origins=origins,
        selection_scope='fit periods only; no validation/control metric used',future_scored=False))
    np.savez_compressed(directory/'inner_curves.npz',**{str(i):curve for i,curve in enumerate(curves)})
    model,processor,valid,_=train(fit,evaluate)
    actual=d.train.iloc[evaluate][target].to_numpy();scores={};predictions={}
    for name,iterations in choices.items():
        p=model.predict(valid,num_iteration=iterations);predictions[name]=p
        scores[name]=float(np.sqrt(np.mean((actual-p)**2)))
    restored=cloudpickle.loads(cloudpickle.dumps((model,processor)))
    replay=restored[0].predict(restored[1].transform(d.features(evaluate.tolist())),num_iteration=choices['latest_origin_early'])
    np.testing.assert_allclose(replay,predictions['latest_origin_early'],rtol=0,atol=0)
    reference_path=ROOT/freeze['profiles'][profile]['reference_predictions']
    assert file_hash(reference_path)==freeze['profiles'][profile]['reference_predictions_sha256']
    reference=json.loads(reference_path.read_text());assert reference['row_keys']==d.features(evaluate.tolist()).index.tolist()
    np.testing.assert_allclose(predictions['latest_origin_early'],reference['values'][target],rtol=0,atol=1e-10)
    np.savez_compressed(directory/'validation_predictions.npz',**predictions)
    usage=resource.getrusage(resource.RUSAGE_SELF)
    atomic_json(directory/'result.json',dict(profile=profile,choices=choices,origins=origins,validation_rmse=scores,
        seconds=time.monotonic()-started,cpu_seconds=usage.ru_utime+usage.ru_stime,
        ru_maxrss_mib=usage.ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        base_fits=3,all_fits_at_full_cap=True,serialization_prediction_identity=True,
        previous_agent_prediction_identity=True,control_scored=False,api_calls=0))


def run(out):
    from importlib.metadata import version
    import psutil
    import sqlite3
    from automl_agent.util import atomic_json,file_hash
    from automl_agent.executor import terminate_tree
    from tools.pytorch_agent_study import load_dataset

    out.mkdir(parents=True,exist_ok=False)
    prior=ROOT/'outputs/agent_autonomy_regression_expanded_20261026/task/.agent_work/bounded_autonomy'
    db=sqlite3.connect((prior/'journal.sqlite').as_uri()+'?mode=ro',uri=True)
    profiles={}
    for name,identifier in [('baseline','baseline'),('regularized','exp_c193cc2d9e4c9c46')]:
        p=json.loads(db.execute('select spec from experiments where id=?',(identifier,)).fetchone()[0])['pipeline']
        path=prior/'attempts'/identifier/'validation_predictions.json'
        profiles[name]=dict(hyperparameters=p['hyperparameters'],reference_predictions=str(path.relative_to(ROOT)),
            reference_predictions_sha256=file_hash(path))
    db.close()
    split='research/pytorch_agent_20261026/reference_templates/regression_split.npz'
    sources=[Path(__file__),*sorted((ROOT/'automl_agent').glob('*.py'))]
    atomic_json(out/'freeze.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={str(p.relative_to(ROOT)):file_hash(p) for p in sources},profiles=profiles,
        lightgbm_version=version('lightgbm'),requirements_sha256=file_hash(ROOT/'requirements.txt'),
        data_hash=load_dataset('regression').source_hash,split_path=split,split_hash=file_hash(ROOT/split),
        patience=100,cpu=1,memory_mb=3072,profile_wall_seconds=180,
        pooled_selection='minimum row-weighted squared RMSE over the two permitted forward origins; first tie',
        control_scored=False,independent_confirmation=False,validation_previously_viewed=True,
        cost_scope='shared full-cap diagnostic fits; not separately timed complete selection algorithms',api_calls=0))
    results=[]
    for profile in profiles:
        begin=time.monotonic();peak=0
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MPLCONFIGDIR=str(out/'mpl'))
        with (out/(profile+'.log')).open('w') as log:
            process=subprocess.Popen([sys.executable,__file__,str(out),'--profile',profile],env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    try:
                        parent=psutil.Process(process.pid)
                        peak=max(peak,sum(p.memory_info().rss for p in [parent,*parent.children(recursive=True)] if p.is_running()))
                    except psutil.NoSuchProcess:pass
                    if time.monotonic()-begin>180 or peak>3072*1024**2:terminate_tree(process.pid)
                    time.sleep(.1)
                process.wait()
            finally:
                if process.poll() is None:
                    try:terminate_tree(process.pid)
                    finally:
                        if process.poll() is None:process.kill()
                        process.wait()
        result=dict(profile=profile,returncode=process.returncode,cold_wall_seconds=time.monotonic()-begin,sampled_tree_peak_mib=peak/1024**2)
        if process.returncode==0:result.update(json.loads((out/profile/'result.json').read_text()))
        results.append(result);atomic_json(out/'results.json',dict(records=results,control_scored=False,api_calls=0))
        print(json.dumps(result),flush=True)
    if any(row['returncode']!=0 for row in results):
        raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--profile',choices=['baseline','regularized']);args=parser.parse_args()
    worker(args.output.resolve(),args.profile) if args.profile else run(args.output.resolve())
