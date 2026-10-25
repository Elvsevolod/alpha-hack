"""Offline native-adapter acceptance in the exact organizer image, no pytest."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile


def run(archive, output):
    output.mkdir(parents=True, exist_ok=False)
    flat = output / 'delivery'
    flat.mkdir()
    if archive.is_dir():
        for path in archive.iterdir():
            assert path.is_file() and (path.suffix == '.py' or path.name == 'requirements.txt')
            shutil.copyfile(path, flat/path.name)
    else:
        with zipfile.ZipFile(archive) as package:
            assert all('/' not in name and (name.endswith('.py') or name == 'requirements.txt') for name in package.namelist())
            package.extractall(flat)
    sys.path.insert(0, str(flat))
    from agent_bundle import materialize
    runtime = materialize(flat)
    project = Path(__file__).resolve().parent.parent
    sys.path = [str(runtime), str(flat), *[p for p in sys.path if p and Path(p).resolve() not in {project, project/'tools'}]]
    import cloudpickle
    import numpy as np
    import pandas as pd
    from automl_agent.budget import Limits
    from automl_agent.contracts import Hypothesis, MetricSpec, OutputSpec, PipelineSpec, TaskSpec, ToolCommand, TrainArgs
    from automl_agent.controller import Controller
    from automl_agent.decision import frozen_decision_valid
    from automl_agent.pipelines import build_pipeline
    from automl_agent.qwen import QwenUnavailable

    report = {'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest() if archive.is_file() else None, 'python':sys.version,
        'runtime':str(runtime), 'runtime_from_archive':archive.is_file(), 'runtime_from_flat_staging':archive.is_dir(),
        'real_qwen_requests':0, 'cases':[]}
    versions = {}
    for line in (flat/'requirements.txt').read_text().splitlines():
        matched = re.fullmatch(r'([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s#]+)', line.strip())
        if matched:
            name, expected = matched.groups()
            actual = importlib.metadata.version(name)
            versions[name] = {'expected':expected, 'actual':actual}
    report['versions'] = versions
    report['exact_requirements'] = all(row['actual'] == row['expected'] for row in versions.values())
    report['network_interfaces'] = [name for _,name in __import__('socket').if_nameindex()]
    # Some kernels expose dormant tunnel devices in an isolated namespace.
    # Route availability is the relevant check; UDP connect sends no payload.
    import errno
    import socket
    routes=Path('/proc/net/route').read_text().splitlines()[1:]
    probe=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:
        probe.connect(('192.0.2.1',9)); route_error=None
    except OSError as error:route_error=error.errno
    finally:probe.close()
    report['network_route_errno']=route_error
    report['network_none']=not any(line.strip() for line in routes) and route_error in {errno.ENETUNREACH,errno.EHOSTUNREACH}
    (output/'REPORT.json').write_text(json.dumps(report,indent=2))
    assert sys.version_info[:2] == (3,12) and report['exact_requirements'] and report['network_none']
    assert not os.environ.get('OPENAI_API_KEY') and not os.environ.get('OPENAI_BASE_URL')

    rng = np.random.default_rng(32)
    X = pd.DataFrame({'x':rng.normal(size=180), 'z':rng.normal(size=180)}, index=[f'train:{i}' for i in range(180)])
    y = pd.DataFrame({'answer':2*X.x + .1*rng.normal(size=180)}, index=X.index)
    regression = TaskSpec(kind='regression', targets=['answer'], id_columns=['id'], feature_columns=list(X),
        metric=MetricSpec(name='rmse'), output=OutputSpec(columns=['id','answer'], prediction_type='values'), source_hash='fixture')
    classification = regression.model_copy(update={'kind':'classification', 'classes':{'answer':['low','middle','high']},
        'metric':MetricSpec(name='f1', average='macro'),
        'output':OutputSpec(columns=['id','answer'], prediction_type='labels')})
    labels = pd.DataFrame({'answer':np.where(X.x < -.5,'low',np.where(X.x > .5,'high','middle'))}, index=X.index)

    def context(task, directory, strategy='random'):
        directory.mkdir(parents=True, exist_ok=True)
        return {'cpu':1, 'memory_mb':4096, 'budget_seconds':100, 'strategy':strategy, 'protocol_hash':'fixed',
            'data_hash':'fixture', 'split_id':'test', 'workdir':str(directory), 'task':task.model_dump(mode='json'),
            'phase_estimates':{'selected_fit_seconds':.2, 'prediction_seconds':.1, 'publication_seconds':.1}}

    def case(name, function):
        started = time.monotonic()
        try:
            details = function()
            report['cases'].append({'case':name, 'passed':True, 'seconds':time.monotonic()-started, **(details or {})})
        except Exception as error:
            report['cases'].append({'case':name, 'passed':False, 'seconds':time.monotonic()-started,
                'error':f'{type(error).__name__}: {error}'})
        (output/'REPORT.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report['cases'][-1]), flush=True)

    def offsets():
        task = classification
        ctx = context(task,output/'offsets')
        spec = PipelineSpec(engine='autogluon', model='library_search', learners=['GBM'],
            autogluon_configurations={'GBM':[{'n_estimators':5,'min_child_samples':1}]},
            prediction_policy='class_offsets', postprocess_fit_policy='pooled_oof', decision_method='regularized')
        model = build_pipeline(spec,task,ctx)
        model.fit(X.iloc[:140],labels.iloc[:140],ctx)
        restored = cloudpickle.loads(cloudpickle.dumps(model))
        assert frozen_decision_valid(restored.frozen_spec(),task)
        restored.refit(X,labels,ctx)
        assert restored.state == model.state and restored.summary.rows == len(X)
        assert set(restored.predict(X.iloc[:5],ctx).values['answer']) <= set(task.classes['answer'])
        return {'full_refit_rows':restored.summary.rows, 'decision_preserved':True}
    case('A01_automl_oof_offsets_refit',offsets)

    for profile in ('bagging','stacking'):
        def graph(profile=profile):
            ctx = context(regression,output/profile,'exact_group')
            spec = PipelineSpec(engine='autogluon', model='library_search', learners=['GBM'], autogluon_profile=profile,
                bag_folds=2, autogluon_configurations={'GBM':[{'n_estimators':5,'num_leaves':7},{'n_estimators':7,'num_leaves':15}]})
            model = build_pipeline(spec,regression,ctx)
            frame = X.copy(); frame['x'] = np.arange(len(X))//4; frame['z'] = 0.
            model.fit(frame.iloc[:140],y.iloc[:140],ctx)
            restored = cloudpickle.loads(cloudpickle.dumps(model)); restored.refit(frame,y,ctx)
            assert restored.selected['answer']['full_refit_rows'] == len(frame)
            assert all('explicit' in name for name in restored.profile_configs['answer']['weights'])
            grouped = restored._bounded_group_codes(restored._autogluon_frame(frame,y,'answer',ctx))
            assert grouped.groupby('x')[restored.autogluon_group_column].nunique().max() == 1
            assert np.isfinite(restored.predict(frame.iloc[:5],ctx).values['answer']).all()
            return {'full_refit_rows':len(frame), 'graph':restored.profile_configs['answer']['graph'], 'groups_preserved':True}
        case('A02_A16_'+profile,graph)

    def conditional():
        ctx = context(regression,output/'conditional')
        spec = PipelineSpec(engine='optuna', model='catboost', n_trials=2,
            hyperparameters={'iterations':4,'depth':2,'bootstrap_type':'Bayesian'},
            search_space={'bootstrap_type':{'type':'categorical','choices':['Bayesian']},
                'subsample':{'type':'float','low':.6,'high':1.,'when':{'parameter':'bootstrap_type','in':['Bernoulli']}}})
        model = build_pipeline(spec,regression,ctx); model.fit(X,y,ctx)
        assert 'subsample' not in model.frozen_spec().hyperparameters
        return {'parameters':model.frozen_spec().hyperparameters}
    case('A12_conditional_catboost',conditional)

    def lightautoml():
        ctx = context(regression,output/'lightautoml')
        model = build_pipeline(PipelineSpec(engine='lightautoml',model='library_search',learners=['lgb']),regression,ctx)
        model.fit(X.iloc[:140],y.iloc[:140],ctx)
        restored = cloudpickle.loads(cloudpickle.dumps(model)); restored.refit(X,y,ctx)
        assert restored.summary.rows == len(X)
        assert np.isfinite(restored.predict(X.iloc[:5],ctx).values['answer']).all()
        return {'full_refit_rows':restored.summary.rows}
    case('lightautoml_fixed_adapter',lightautoml)

    def task_directory(name, multiclass=False):
        root=output/name; root.mkdir()
        data=X.copy(); data.index=np.arange(len(data)); data.insert(0,'id',np.arange(len(data)))
        data['answer']=(labels if multiclass else y).answer.to_numpy()
        data.iloc[:150].to_csv(root/'train.csv',index=False)
        data.iloc[150:].drop(columns='answer').to_csv(root/'test.csv',index=False)
        pd.DataFrame({'id':np.arange(150,180),'answer':0}).to_csv(root/'sample_submission.csv',index=False)
        task=classification if multiclass else regression
        (root/'task_spec.json').write_text(task.model_dump_json(exclude={'source_hash'}))
        (root/'task.md').write_text('Classification. Target: answer. Primary metric: Macro F1. Files: train.csv and test.csv. Output columns id,answer.' if multiclass else
            'Regression. Target: answer. Primary metric: RMSE. Files: train.csv and test.csv. Output columns id,answer.')
        return root

    def finalist_failure():
        root=task_directory('finalist_recovery')
        c=Controller(root/'task.md',Limits(seconds=300,cpu=1,memory_mb=4096))
        c.no_llm=True; c.qwen.disabled=True
        try:
            c.prepare()
            args=TrainArgs(pipeline=PipelineSpec(model='ridge'), budget_seconds=30, hypothesis=Hypothesis(
                observation='Offline alternative',expected_effect='Compare same metric',changed_component='family',acceptance='Same fixed development rows'))
            c.dispatch(ToolCommand(operation_id='ridge_candidate',tool='train_model',arguments=args.model_dump()))
            c.freeze('Offline failed challenger before opening control')
            baseline=c.journal.get('baseline_id')
            other=next(name for name in c.journal.get('finalization_plan')['finalists'] if name != baseline)
            original=c._prepare_finalist
            def prepare(identifier, rows, part, seconds):
                if identifier == other and part == 'control': raise RuntimeError('Synthetic challenger failure')
                return original(identifier,rows,part,seconds)
            c._prepare_finalist=prepare; c.finalize()
            assert c.journal.get('control_opened')['finalists'] == [baseline]
            assert c.journal.get('refit_status') == 'completed'
            assert c.journal.get('publication')['scope'] == 'full_train_refit'
            output_csv=pd.read_csv(root/'submission.csv')
            assert output_csv.id.tolist() == list(range(150,180)) and np.isfinite(output_csv.answer).all()
            return {'control':c.journal.get('control_status'),'refit':c.journal.get('refit_status'),'csv_valid':True}
        finally:c.close()
    case('A03_failed_challenger_full_refit',finalist_failure)

    def optional_failure():
        root=task_directory('no_llm_preparation',True)
        c=Controller(root/'task.md',Limits(seconds=300,cpu=1,memory_mb=4096))
        original=c.execute_experiment
        def execute(spec,*args,**kwargs):
            if spec.experiment_id.startswith('baseline_decision'): raise QwenUnavailable('Synthetic optional preparation failure')
            return original(spec,*args,**kwargs)
        c.execute_experiment=execute
        try:
            result=c.run(no_llm=True)
            assert result['state'] == result['refit_status'] == 'completed'
            assert c.journal.get('bootstrap_decision_id') and c.journal.get('publication')['scope'] == 'full_train_refit'
            assert c.budget.counter('llm_calls') == 0
            return {'state':result['state'],'refit':result['refit_status'],'model_requests':0}
        finally:c.close()
    case('A17_optional_preparation_no_llm',optional_failure)
    report['passed']=all(row['passed'] for row in report['cases'])
    (output/'REPORT.json').write_text(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(run(args.archive.resolve(),args.output.resolve()))
