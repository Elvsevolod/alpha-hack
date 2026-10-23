"""Small preregistered sampler comparison; external, offline, never packaged."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def prepare(out):
    import numpy as np
    import pandas as pd
    out.mkdir(parents=True, exist_ok=False)
    registration = {'candidate':'TPE startup4 instead of10 for8 trials; no other search change',
        'search_seeds':[11,23,47], 'model_seed':73, 'sample_seed':20261023,
        'outer_split_seed':113, 'inner_split_seed':97, 'rows_each':6000,
        'trials_each':8, 'n_estimators':60, 'cpu':1, 'memory_mb':768,
        'maximum_wall_seconds':240, 'organizer_requests':0,
        'gates':{'minimum_class_support':3, 'all_studies_complete8_trials':True,
            'outer_relative_RMSE_median_gain_at_least':.005,
            'outer_MacroF1_median_gain_at_least':.005,
            'wins_at_least_each_task':2, 'maximum_median_search_time_ratio':1.25},
        'decision':'Any failed gate means DO_NOT_PROMOTE; no alternate startup or seeds/retry',
        'scope':'Uniform small samples of already viewed public train only; no task test labels, no private-score claim. Positive result is provisional and needs full runtime verification before promotion',
        'validation':'All studies choose using fixed inner split only; all winners frozen before any outer control is scored',
        'datasets':{}}
    for kind, folder, source in [('regression','contract_v048_regression_full3h/regression','Регрессия'),
            ('classification','stream_v045_full3h/classification','Многоклассовая классификация')]:
        dbpath=ROOT/'outputs/quality_goal_20261023'/folder/'task/.agent_work/diagnostic/journal.sqlite'
        assert not Path(str(dbpath)+'-wal').exists() or Path(str(dbpath)+'-wal').stat().st_size==0
        with sqlite3.connect('file:'+str(dbpath)+'?mode=ro&immutable=1',uri=True) as db:
            task=json.loads(db.execute("SELECT value FROM metadata WHERE key='task'").fetchone()[0])
        path=ROOT/'data_for_participants'/source/'train.csv'
        rng=np.random.default_rng(registration['sample_seed'])
        sample=None
        for frame in pd.read_csv(path,chunksize=4096):
            frame['_pilot_priority']=rng.random(len(frame))
            frame['_pilot_row']=frame.index
            sample=pd.concat([sample,frame]) if sample is not None else frame
            sample=sample.nsmallest(registration['rows_each'],'_pilot_priority')
        sample=sample.sort_values('_pilot_row').drop(columns='_pilot_priority')
        dest=out/(kind+'.csv');sample.to_csv(dest,index=False)
        write(out/(kind+'_task.json'),task)
        registration['datasets'][kind]={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'sample_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(), 'rows':len(sample),
            'task_contract_sha256':hashlib.sha256((out/(kind+'_task.json')).read_bytes()).hexdigest()}
    write(out/'REGISTRATION.json',registration)


def execute(out):
    import numpy as np
    import pandas as pd
    import optuna
    from sklearn.model_selection import train_test_split
    from automl_agent.contracts import TaskSpec,PipelineSpec
    from automl_agent.pipelines import DirectPipeline
    from automl_agent.evaluation import align_predictions,score_predictions
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    reg=json.loads((out/'REGISTRATION.json').read_text())
    assert optuna.__version__=='4.2.1'
    started=time.monotonic();rows=[];frozen=[];data={}
    result={'status':'incomplete','organizer_requests':0,'runtime_changed':False,'rows':rows,
        'registration_sha256':hashlib.sha256((out/'REGISTRATION.json').read_bytes()).hexdigest()}
    def timeout(*args):raise TimeoutError('Preregistered pilot wall budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(reg['maximum_wall_seconds'])
    try:
        for kind in ('regression','classification'):
            path=out/(kind+'.csv')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==reg['datasets'][kind]['sample_sha256']
            task=TaskSpec.model_validate_json((out/(kind+'_task.json')).read_text())
            frame=pd.read_csv(path);frame.index=['sample:'+str(x) for x in frame.pop('_pilot_row')]
            X,y=frame[task.feature_columns],frame[task.targets]
            labels=y.iloc[:,0] if task.classes else None
            if labels is not None and (set(labels)!=set(task.classes[task.targets[0]]) or labels.value_counts().min()<3):
                raise ValueError('Sample lacks preregistered class support; do not resample')
            fit,control=train_test_split(np.arange(len(X)),test_size=.2,random_state=113,stratify=labels)
            a,b=train_test_split(fit,test_size=.2,random_state=97,stratify=y.iloc[fit,0] if task.classes else None)
            ctx={'cpu':1,'memory_mb':768,'strategy':'random','workdir':str(out),
                'budget_seconds':240,'protocol_hash':'startup_pilot','split_id':'inner','task':task.model_dump(mode='json')}
            data[kind]=(task,X,y,fit,control,ctx)
            def evaluate(parameters,train,evaluation,split):
                spec=PipelineSpec(model='lightgbm',representation='native',seed=73,
                    hyperparameters={**parameters,'n_estimators':60,**({'class_weight':'balanced'} if task.classes else {})})
                model=DirectPipeline(spec,task);model.fit(X.iloc[train],y.iloc[train],ctx)
                bundle=model.predict(X.iloc[evaluation],{**ctx,'split_id':split})
                prediction=align_predictions(bundle,X.iloc[evaluation].index.tolist(),task,'startup_pilot',split)
                return score_predictions(task,y.iloc[evaluation],prediction,bundle.prediction_type)
            for seed in reg['search_seeds']:
                for variant in (['A','B'] if seed!=23 else ['B','A']):
                    study=optuna.create_study(direction='maximize' if task.metric.maximize else 'minimize',
                        sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=10 if variant=='A' else 4))
                    tick=time.monotonic()
                    def objective(trial):
                        params={'learning_rate':trial.suggest_float('learning_rate',.01,.3,log=True),
                            'num_leaves':trial.suggest_int('num_leaves',7,63),
                            'min_child_samples':trial.suggest_int('min_child_samples',5,80),
                            'reg_lambda':trial.suggest_float('reg_lambda',1e-4,10.,log=True)}
                        return evaluate(params,a,b,'inner')
                    study.optimize(objective,n_trials=8)
                    row={'kind':kind,'seed':seed,'variant':variant,'startup':10 if variant=='A' else 4,
                        'search_seconds':time.monotonic()-tick,'completed_trials':len(study.trials),
                        'inner_best':study.best_value,'selected_parameters':study.best_params,
                        'trials':[{'number':t.number,'value':t.value,'parameters':t.params} for t in study.trials]}
                    rows.append(row);frozen.append(row)
                    write(out/'PROGRESS.json',result)
        # Freeze every selection before opening any held-out scores.
        write(out/'FROZEN_WINNERS.json',frozen)
        for row in rows:
            task,X,y,fit,control,ctx=data[row['kind']]
            params={**row['selected_parameters'],'n_estimators':60,**({'class_weight':'balanced'} if task.classes else {})}
            model=DirectPipeline(PipelineSpec(model='lightgbm',representation='native',seed=73,hyperparameters=params),task)
            model.fit(X.iloc[fit],y.iloc[fit],ctx)
            bundle=model.predict(X.iloc[control],{**ctx,'split_id':'control'})
            pred=align_predictions(bundle,X.iloc[control].index.tolist(),task,'startup_pilot','control')
            row['outer_control']=score_predictions(task,y.iloc[control],pred,bundle.prediction_type)
        comparisons={}
        for kind in ('regression','classification'):
            gains=[];ratios=[]
            for seed in reg['search_seeds']:
                a=next(r for r in rows if (r['kind'],r['seed'],r['variant'])==(kind,seed,'A'))
                b=next(r for r in rows if (r['kind'],r['seed'],r['variant'])==(kind,seed,'B'))
                gains.append((a['outer_control']-b['outer_control'])/a['outer_control'] if kind=='regression'
                    else b['outer_control']-a['outer_control'])
                ratios.append(b['search_seconds']/a['search_seconds'])
            comparisons[kind]={'paired_gains':gains,'median_gain':float(np.median(gains)),
                'wins':sum(g>0 for g in gains),'median_search_time_ratio':float(np.median(ratios))}
        result.update(status='completed',comparisons=comparisons)
        passed=all(c['median_gain']>=.005 and c['wins']>=2 and c['median_search_time_ratio']<=1.25 for c in comparisons.values())
        result['decision']='PROVISIONAL_PASS_NEEDS_RUNTIME_VERIFICATION' if passed else 'DO_NOT_PROMOTE'
    except Exception as error:
        result.update(status='inconclusive',decision='DO_NOT_PROMOTE',error_type=type(error).__name__,error=str(error))
    finally:
        signal.alarm(0);result['elapsed_seconds']=time.monotonic()-started
        write(out/'REPORT.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','decision','comparisons','error','elapsed_seconds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','execute']);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();(prepare if args.phase=='prepare' else execute)(args.output.resolve())
