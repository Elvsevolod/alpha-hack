import json,os,subprocess,sys,time
from pathlib import Path
import cloudpickle
import numpy as np
import pandas as pd
import pytest
from automl_agent.contracts import PipelineSpec,MetricSpec,OutputSpec,Transformation
from automl_agent.pipelines import build_pipeline,validate_parameters
from automl_agent.torch_pipeline import TorchPipeline
from automl_agent.research_evaluation import oof_predictions
from automl_agent.executor import stage_frame
from automl_agent.util import atomic_json


def context(task,tmp_path):
    return dict(cpu=1,memory_mb=2000,budget_seconds=30,strategy='random',task=task.model_dump(mode='json'),protocol_hash='fixture',split_id='validation',workdir=str(tmp_path),deadline=time.time()+60)


@pytest.mark.parametrize('params',[{'encoding':'quantile'},{'encoding':'learned_piecewise'},{'encoding':'periodic'},{'heads':4},{'residual':True},{'attention':True,'encoding':'quantile'}])
def test_builtin_bundle_frozen_refit_and_state(tabular,tmp_path,monkeypatch,params):
    X,y,task=tabular;c=context(task,tmp_path);spec=PipelineSpec(model='torch_tabular',hyperparameters={'epochs':3,**params})
    p=build_pipeline(spec,task,c);summary=p.fit(X.iloc[:150],y.iloc[:150],c)
    assert summary.auxiliary_fits==1 and summary.budget['complete_epoch_refit']
    frozen=p.frozen_spec();assert frozen.iteration_schedule['answer']>=1
    a=p.predict(X.iloc[-8:],c);restored=cloudpickle.loads(cloudpickle.dumps(p));b=restored.predict(X.iloc[-8:],c)
    np.testing.assert_array_equal(a.values['answer'],b.values['answer'])
    assert restored.model.state and restored.model.network_config
    monkeypatch.setattr('automl_agent.torch_pipeline.inner_split',lambda *args: (_ for _ in ()).throw(AssertionError('Frozen refit must not select epochs')))
    full=build_pipeline(frozen,task,c);s=full.fit(X,y,{**c,'execution_mode':'refit'})
    assert s.auxiliary_fits==0 and full.model.epochs_run==frozen.iteration_schedule['answer']
    assert full.model.mean==pytest.approx(y.answer.mean())
    assert full.predict(X[:0],c).values['answer']==[]


def test_nested_oof_selects_epochs_only_inside_own_fold(tabular,tmp_path,monkeypatch):
    X,y,task=tabular;c=context(task,tmp_path);spec=PipelineSpec(model='torch_tabular',hyperparameters={'epochs':2})
    original=TorchPipeline.fit;calls=[]
    def checked(model,a,b,ctx):
        assert not set(a.index)&set(ctx.get('forbidden',[]))
        result=original(model,a,b,ctx)
        calls.append((set(a.index),model.summary.auxiliary_fits));return result
    monkeypatch.setattr(TorchPipeline,'fit',checked)
    report=oof_predictions(spec,task,X,y,c)
    assert len(calls)==3 and all(n==1 for _,n in calls) and report['base_fits']==6
    for part,(fit,_) in zip(report['folds'],calls):
        assert fit==set(part['fit_keys']) and not fit&set(part['evaluation_keys'])


def test_early_temporal_oof_is_skipped_without_own_inner_history(tabular,tmp_path):
    X,y,t=tabular;X=X.copy();X['period']=np.repeat(range(4),len(X)//4)
    task=t.model_copy(update={'time_column':'period'})
    c={**context(task,tmp_path),'strategy':'temporal'}
    r=oof_predictions(PipelineSpec(model='torch_tabular',hyperparameters={'epochs':2}),task,X,y,c)
    assert r['skipped_folds'] and len(r['folds'])==2 and r['base_fits']==4
    for part in r['folds']:
        assert X.loc[part['fit_keys']].period.max()<X.loc[part['evaluation_keys']].period.min()


@pytest.mark.parametrize('change',[{'engine':'optuna'},{'target_transform':'log1p'},{'prediction_policy':'class_offsets','postprocess_fit_policy':'pooled_oof'}, {'imputation':{'numeric':'knn'}},{'hyperparameters':{'width':999999}},{'hyperparameters':{'attention':True,'heads':4}},{'hyperparameters':{'attention':True,'width':128}}])
def test_unsupported_modes_rejected_before_fit(tabular,change):
    _,_,task=tabular
    spec=PipelineSpec.model_validate({'model':'torch_tabular',**change})
    with pytest.raises(ValueError):build_pipeline(spec,task,{})


def test_metric_and_unfrozen_refit_rejected(tabular,tmp_path):
    X,y,task=tabular;spec=PipelineSpec(model='torch_tabular')
    with pytest.raises(ValueError,match='RMSE or Macro F1'):
        build_pipeline(spec,task.model_copy(update={'metric':MetricSpec(name='mae')}),{})
    p=build_pipeline(spec,task,{})
    with pytest.raises(ValueError,match='frozen epochs'):p.fit(X,y,{**context(task,tmp_path),'execution_mode':'refit'})


def test_stateless_transform_and_selected_roles(tabular,tmp_path):
    X,y,t=tabular
    spec=PipelineSpec(model='torch_tabular',hyperparameters={'epochs':2},exclude_features=['empty'],transformations=[Transformation(name='log1p',parameters={'columns':['number']})])
    # Fixture number has negatives. Use nonnegative values for this explicit transform.
    X=X.copy();X['number']=X['number'].abs()
    p=build_pipeline(spec,t,{});p.fit(X,y,context(t,tmp_path))
    assert 'empty' not in p.model.columns and p.summary.preprocessing['stateless_frame_transformations']==['log1p']
    assert np.isfinite(p.predict(X[:3],context(t,tmp_path)).values['answer']).all()


def test_torch_worker_uses_existing_sandbox_and_roundtrip(tabular,tmp_path):
    X,y,task=tabular;directory=tmp_path/'worker';directory.mkdir()
    for name,frame in [('fit_X',X[:150]),('fit_y',y[:150]),('validation_X',X[150:])]:stage_frame(directory/(name+'.json'),frame)
    c=context(task,directory);spec=PipelineSpec(model='torch_tabular',hyperparameters={'epochs':2})
    request={'mode':'train','task':task.model_dump(mode='json'),'pipeline':spec.model_dump(mode='json'),'context':c,'predict_splits':['validation']}
    path=directory/'request.json';atomic_json(path,request)
    env={k:v for k,v in os.environ.items() if k in {'PATH','HOME'}}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(directory),MPLCONFIGDIR=str(directory/'mpl'))
    worker=Path(__file__).resolve().parents[1]/'automl_agent/worker_entry.py'
    result=subprocess.run([sys.executable,'-I',str(worker),str(path)],env=env,capture_output=True,text=True,timeout=45)
    report=json.loads((directory/'worker_result.json').read_text())
    assert result.returncode==0 and report['status']=='completed',report
    assert (directory/'model.pkl').is_file() and (directory/'validation_predictions.json').is_file()


def test_classification_empty_bundle_and_explicit_order(tabular,tmp_path):
    from automl_agent.evaluation import align_predictions
    X,y,t=tabular;y=pd.DataFrame({'answer':np.where(y.answer>0,'b','a')},index=y.index)
    task=t.model_copy(update={'kind':'classification','classes':{'answer':['b','a']},'metric':MetricSpec(name='f1'),'output':OutputSpec(columns=['identifier','answer'],prediction_type='labels')})
    c=context(task,tmp_path);p=build_pipeline(PipelineSpec(model='torch_tabular',hyperparameters={'epochs':2}),task,c);p.fit(X,y,c)
    bundle=p.predict(X[:0],c);assert bundle.classes==task.classes
    assert align_predictions(bundle,[],task,'fixture','validation')['answer'].shape==(0,2)


def test_production_and_frozen_research_core_predict_identically(tabular):
    from tools.torch_research_core import TorchMLP as Research
    from automl_agent.torch_tabular import TorchMLP as Production
    X,y,t=tabular;params=dict(columns=list(X),categorical=t.categorical_columns,epochs=3,seed=17,encoding='learned_piecewise',learning_rate=.002)
    a=Research(**params).fit(X,y.answer);b=Production(**params).fit(X,y.answer)
    np.testing.assert_array_equal(a.predict(X),b.predict(X))


def test_undeclared_stopping_labels_are_not_silently_ignored(tabular):
    from automl_agent.torch_tabular import TorchMLP
    X,y,t=tabular;labels=np.array(['a','b']*(len(X)//2))
    with pytest.raises(ValueError,match='Undeclared evaluation label'):
        TorchMLP(list(X),t.categorical_columns,classes=['a','b'],epochs=1).fit(X,labels,(X[:2],['unknown','a']))


@pytest.mark.parametrize('transport',['native','json'])
def test_http_mock_negative_neural_command_keeps_backup(tmp_path,monkeypatch,transport):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    from threading import Thread
    from automl_agent.controller import Controller
    from automl_agent.budget import Limits
    from automl_agent.util import file_hash
    from automl_agent.tool_results import payload
    from conftest import write_task
    args={'pipeline':{'model':'torch_tabular','hyperparameters':{'width':999999}},'budget_seconds':20,
        'hypothesis':{'observation':'offline negative fixture','expected_effect':'validate bounded route','changed_component':'model','acceptance':'preserve backup on rejection'}}
    command={'operation_id':'invalid_neural','tool':'train_model','arguments':args}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            message={'role':'assistant','content':json.dumps(command)}
            if 'tools' in request:
                message={'role':'assistant','content':None,'tool_calls':[{'id':'fixture','type':'function','function':{'name':'train_model','arguments':json.dumps({'operation_id':'invalid_neural',**args})}}]}
            body=json.dumps({'choices':[{'message':message,'finish_reason':'tool_calls' if 'tools' in request else 'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':4}}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    monkeypatch.setenv('OPENAI_API_KEY','offline-fixture');monkeypatch.setenv('OPENAI_MODEL','offline-fixture')
    monkeypatch.setenv('OPENAI_BASE_URL',f'http://127.0.0.1:{server.server_port}');monkeypatch.delenv('OPENAI_EXTRA_BODY',raising=False)
    controller=Controller(write_task(tmp_path/'task'),Limits(seconds=90,cpu=1),run_id='negative',transport=transport)
    try:
        controller.prepare();before=file_hash(controller.publisher.output)
        received,_=controller.qwen.next_command('Offline fixture',{'state':'ready'})
        result=controller.dispatch(received)
        assert result['status'] in {'incompatible','technical_failure'},result
        assert file_hash(controller.publisher.output)==before
        assert 'width' in json.dumps(result).lower() or 'parameter' in json.dumps(result).lower()
    finally:
        controller.close();server.shutdown();server.server_close();thread.join()


@pytest.mark.parametrize('mode',['deadline','memory'])
def test_executor_resource_stop_does_not_publish_partial_neural_model(tabular,tmp_path,mode):
    from automl_agent.executor import Executor
    from automl_agent.journal import Journal
    from automl_agent.util import file_hash
    X,y,task=tabular;directory=tmp_path/'attempt';directory.mkdir();backup=tmp_path/'submission.csv';backup.write_text('id,y\n0,1\n');before=file_hash(backup)
    for name,frame in [('fit_X',X),('fit_y',y),('validation_X',X[:5])]:stage_frame(directory/(name+'.json'),frame)
    ctx=context(task,directory);spec=PipelineSpec(model='torch_tabular',hyperparameters={'width':512,'depth':4,'heads':8,'epochs':200})
    request={'mode':'train','task':task.model_dump(mode='json'),'pipeline':spec.model_dump(mode='json'),'context':ctx,'predict_splits':['validation']}
    journal=Journal(tmp_path/'journal')
    try:
        executor=Executor(1,128 if mode=='memory' else 2000,time.time()+40,journal)
        result=executor.run(directory,request,30 if mode=='memory' else .01)
        assert result['status']==('memory_limit' if mode=='memory' else 'timeout'),result
        assert not (directory/'model.pkl').exists() and file_hash(backup)==before
    finally:journal.close()


def test_pipeline_inference_in_a_fresh_python_process(tabular,tmp_path):
    X,y,task=tabular;c=context(task,tmp_path);spec=PipelineSpec(model='torch_tabular',hyperparameters={'epochs':3,'attention':True})
    p=build_pipeline(spec,task,c);p.fit(X,y,c)
    with (tmp_path/'model.pkl').open('wb') as stream:cloudpickle.dump(p,stream)
    query=X[:5].copy();query['category']='unseen';stage_frame(tmp_path/'input.json',query)
    np.save(tmp_path/'expected.npy',np.asarray(p.predict(query,c).values['answer']))
    atomic_json(tmp_path/'context.json',c)
    code="""import sys,json,cloudpickle,numpy as np
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from automl_agent.worker import read_frame
p=Path(sys.argv[2])
with (p/'model.pkl').open('rb') as f:model=cloudpickle.load(f)
actual=model.predict(read_frame(p/'input.json'),json.loads((p/'context.json').read_text())).values['answer']
np.testing.assert_allclose(actual,np.load(p/'expected.npy'),rtol=1e-6)
assert model.frozen_spec().iteration_schedule['answer']>=1
"""
    env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','TMPDIR'}};env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable,'-I','-c',code,str(Path(__file__).resolve().parents[1]),str(tmp_path)],env=env,check=True,timeout=30)


@pytest.mark.parametrize('transport',['native','json'])
def test_mock_agent_cycle_trains_neural_then_publishes_real_csv(tmp_path,monkeypatch,transport):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    from threading import Thread
    from automl_agent.controller import Controller
    from automl_agent.budget import Limits
    from conftest import write_task
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])));requests.append(request)
            if len(requests)==1:
                command={'operation_id':'mock_neural','tool':'train_model','arguments':{'pipeline':{'model':'torch_tabular','hyperparameters':{'epochs':15,'width':32,'learning_rate':.01}},'budget_seconds':60,
                    'hypothesis':{'observation':'Synthetic fit has numeric and categorical features','expected_effect':'Exercise nested epochs and a different model family','changed_component':'model','acceptance':'A completed bounded fit with valid probabilities or values'}}}
            else:
                command={'operation_id':'mock_finish_'+str(len(requests)),'tool':'finish','arguments':{'result_ids':[controller.journal.get('leader_id') or 'baseline'],'reason':'Fixed offline lifecycle fixture is complete; retain the measured winner even if neural is worse','imputation_skip_reason':'Fixed technical fixture; no additional imputation search', 'skipped_directions':{'all':'This controlled fixture verifies transport, candidate execution, frozen finalization and a real published CSV; it makes no scientific completeness claim.'}}}
            message={'role':'assistant','content':json.dumps(command)}
            if 'tools' in request:message={'role':'assistant','content':None,'tool_calls':[{'id':'fixture_'+str(len(requests)),'type':'function','function':{'name':command['tool'],'arguments':json.dumps({'operation_id':command['operation_id'],**command['arguments']})}}]}
            body=json.dumps({'choices':[{'message':message,'finish_reason':'tool_calls' if 'tools' in request else 'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':5}}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    monkeypatch.setenv('OPENAI_API_KEY','offline-fixture');monkeypatch.setenv('OPENAI_MODEL','offline-fixture');monkeypatch.setenv('OPENAI_BASE_URL',f'http://127.0.0.1:{server.server_port}');monkeypatch.delenv('OPENAI_EXTRA_BODY',raising=False)
    controller=Controller(write_task(tmp_path/'task'),Limits(seconds=120,cpu=1),run_id='cycle',transport=transport)
    try:
        report=controller.run()
        neural=[(s,r) for s,r in controller.records() if s.pipeline.model=='torch_tabular']
        assert neural and any(r is not None and r.status=='completed' for s,r in neural),[(s.model_dump(),r.model_dump() if r else None) for s,r in neural]
        assert report['state']=='completed' and report['refit_status']=='completed',report
        csv=pd.read_csv(controller.publisher.output);assert list(csv)==['row','answer'] and len(csv)==25 and np.isfinite(csv.answer).all()
        assert len(requests)>=2 and report['publication']['scope']=='full_train_refit'
    finally:controller.close();server.shutdown();server.server_close();thread.join()


def test_completion_forecast_counts_worker_initialization_as_fixed_cost():
    from automl_agent.cost_model import phase_forecast
    summary={'rows':100,'seconds':5.,'budget':{'selected_fit_actual_seconds':1.,'selected_fit_rows':100,'worker_initialization_seconds':3.},
        'execution_phases':{'fit_seconds':5.,'load_and_construct_seconds':.1,'save_load_seconds':.1,'predict_and_verify_seconds':.1,'prediction_seconds':.05,'prediction_rows':10,'forecast_prediction_rows':10}}
    without={**summary,'budget':{**summary['budget'],'worker_initialization_seconds':0.}}
    for rows in [100,1000]:
        delta=phase_forecast(summary,6.,rows,100)-phase_forecast(without,6.,rows,100)
        assert delta==pytest.approx(3.*1.25)


def test_learned_ensemble_can_freeze_neural_member_without_control_labels(tmp_path,monkeypatch):
    from automl_agent.controller import Controller
    from automl_agent.budget import Limits
    from automl_agent.contracts import ToolCommand
    from automl_agent.tool_results import payload
    from conftest import write_task
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    c=Controller(write_task(tmp_path/'task'),Limits(seconds=240,cpu=1),run_id='neural_blend')
    hypothesis=dict(observation='Offline numeric fixture',expected_effect='Verify honest neural OOF composition',
        changed_component='model',acceptance='Frozen weights and completed fit-only folds')
    try:
        c.prepare();ids=[]
        for i,pipeline in enumerate([{'model':'ridge'},
                {'model':'torch_tabular','hyperparameters':{'epochs':3,'width':16}}]):
            r=payload(c.dispatch(ToolCommand(operation_id='member_'+str(i),tool='train_model',
                arguments=dict(pipeline=pipeline,budget_seconds=60,hypothesis=hypothesis))))
            assert r['result']['status']=='completed',r
            ids.append(r['experiment']['experiment_id'])
        r=payload(c.dispatch(ToolCommand(operation_id='neural_oof_blend',tool='build_ensemble',
            arguments=dict(members=ids,strategy='weighted_oof',budget_seconds=80,hypothesis=hypothesis))))
        assert r['result']['status']=='completed',r
        _,saved=c.lookup(r['experiment']['experiment_id'])
        state=json.loads((Path(saved.artifacts['directory'])/'worker_result.json').read_text())
        assert np.isclose(sum(state['frozen_pipeline']['ensemble_weights']),1.)
        assert not c.journal.get('control_opened')
        assert len(state['frozen_pipeline']['members'])==2
    finally:c.close()
