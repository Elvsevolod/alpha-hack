"""Class weights refer to labels and survive JSON, inner fits and refit."""
import json

import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import PipelineSpec,TaskSpec
from automl_agent.pipelines import build_pipeline,validate_parameters


def task_for(labels,targets=None):
    targets=targets or ['answer']
    return TaskSpec(kind='multioutput_classification' if len(targets)>1 else 'classification',
        targets=targets,classes={t:labels for t in targets},feature_columns=['x'],metric={'name':'f1'},
        output={'columns':targets,'prediction_type':'labels'},source_hash='fixture')


def context(task,tmp_path):
    return {'task':task.model_dump(mode='json'),'strategy':'random','cpu':1,'memory_mb':1024,
        'budget_seconds':20,'protocol_hash':'fixed','split_id':'validation','workdir':str(tmp_path)}


@pytest.mark.parametrize('weights',[{'1':2},{'no':-1},{'yes':float('inf')},{'yes':True},{'no':0,'yes':0},{'yes':10**1000},{'yes':'3'},{'yes':None}])
def test_invalid_weights_fail_before_preprocessing(monkeypatch,weights):
    task=task_for(['no','yes']);spec=PipelineSpec(model='lightgbm',hyperparameters={'class_weight':weights})
    monkeypatch.setattr('automl_agent.preprocessing.FeatureMatrix.fit',lambda *a:pytest.fail('No preprocessing may start'))
    with pytest.raises(ValueError,match='class_weight'):build_pipeline(spec,task,{})


@pytest.mark.parametrize('model',['lightgbm','logistic','sgd','random_forest','extra_trees'])
@pytest.mark.parametrize('labels',[[0,1],[20,10],['no','yes'],['1','0']])
def test_numeric_label_keys_survive_json_and_native_fit(model,labels,tmp_path):
    task=task_for(labels);X=pd.DataFrame({'x':np.linspace(-2,2,80)})
    y=pd.DataFrame({'answer':np.where(X.x>0,labels[1],labels[0])})
    extra={'n_estimators':8} if model in {'lightgbm','random_forest','extra_trees'} else {'max_iter':1000}
    weights={labels[0]:1,labels[1]:3}
    spec=PipelineSpec.model_validate_json(PipelineSpec(model=model,hyperparameters={'class_weight':weights,**extra}).model_dump_json())
    assert set(spec.hyperparameters['class_weight'])==set(map(str,labels))
    pipeline=build_pipeline(spec,task,{});pipeline.fit(X,y,context(task,tmp_path))
    native=pipeline.models['answer']
    assert native.get_params()['class_weight']==weights
    prediction=pipeline.predict(X[:5],context(task,tmp_path))
    assert np.asarray(prediction.values['answer']).shape==(5,2)
    raw=native.predict_proba(pipeline.preprocessor.transform(X[:5]))
    np.testing.assert_allclose(prediction.values['answer'],raw[:,[list(native.classes_).index(c) for c in labels]])


def test_declared_absent_class_is_filtered_only_for_current_fit(tmp_path):
    task=task_for([0,1,2]);X=pd.DataFrame({'x':np.linspace(-2,2,90)})
    y=pd.DataFrame({'answer':np.repeat([0,1,2],30)})
    spec=PipelineSpec(model='lightgbm',hyperparameters={'n_estimators':8,'class_weight':{'0':1,'1':2,'2':9}})
    first=build_pipeline(spec,task,{});first.fit(X[:60],y[:60],context(task,tmp_path))
    assert first.models['answer'].get_params()['class_weight']=={0:1,1:2}
    assert first.frozen_spec().hyperparameters['class_weight']['2']==9
    p=np.asarray(first.predict(X[-5:],context(task,tmp_path)).values['answer'])
    assert p.shape==(5,3) and np.all(p[:,2]==0)
    full=build_pipeline(first.frozen_spec(),task,{});full.fit(X,y,context(task,tmp_path))
    assert full.models['answer'].get_params()['class_weight']=={0:1,1:2,2:9}


def test_forest_multioutput_maps_each_dictionary_to_its_target(tmp_path):
    task=task_for([0,1],['left','right']);X=pd.DataFrame({'x':np.arange(60)})
    y=pd.DataFrame({'left':np.arange(60)%2,'right':np.arange(60)%3==0}).astype(int)
    spec=PipelineSpec(model='random_forest',hyperparameters={'n_estimators':5,'class_weight':[{'0':1,'1':3},{'0':4,'1':1}]})
    p=build_pipeline(spec,task,{});p.fit(X,y,context(task,tmp_path))
    assert p.models['left'].class_weight=={0:1,1:3} and p.models['right'].class_weight=={0:4,1:1}


def test_string_labels_are_not_reinterpreted_as_numeric_codes(tmp_path):
    from automl_agent.class_weight import resolve_class_weight
    task=task_for(['0','1']);spec=PipelineSpec(model='lightgbm',hyperparameters={'class_weight':{'1':4}})
    assert resolve_class_weight(spec,task,'answer')=={'1':4}
    ambiguous=task_for([1,'1'])
    with pytest.raises(ValueError,match='ambiguous'):validate_parameters(spec,ambiguous)
    collision=PipelineSpec(model='lightgbm',hyperparameters={'class_weight':{'1':2,'1.0':3}})
    with pytest.raises(ValueError,match='ambiguous'):validate_parameters(collision,task_for([0,1]))
    large=9007199254740993
    spec=PipelineSpec(model='lightgbm',hyperparameters={'class_weight':{str(large):4}})
    assert resolve_class_weight(spec,task_for([large-1,large]),'answer')=={large:4}


def test_controller_json_weights_reach_worker_and_unknown_labels_start_no_candidate(tmp_path):
    import cloudpickle
    from pathlib import Path
    from automl_agent.budget import Limits
    from automl_agent.controller import Controller
    from automl_agent.contracts import ToolCommand
    from automl_agent.tool_results import payload
    from conftest import write_task
    path=write_task(tmp_path,kind='classification')
    train=pd.read_csv(tmp_path/'train.csv',dtype=str,keep_default_na=False)
    train['answer']=train.answer.map({'negative':'0','positive':'1'});train.to_csv(tmp_path/'train.csv',index=False)
    spec=json.loads((tmp_path/'task_spec.json').read_text());spec['classes']={'answer':[0,1]}
    (tmp_path/'task_spec.json').write_text(json.dumps(spec))
    c=Controller(path,Limits(seconds=180,cpu=1),run_id='numeric_weight')
    hypothesis={'observation':'Numeric output labels are confirmed','expected_effect':'Verify label weighting transport',
        'changed_component':'class_weight','acceptance':'Valid fit and label-aligned probabilities'}
    try:
        c.prepare();count=len(c.records())
        bad=payload(c.dispatch(ToolCommand(operation_id='bad_weight',tool='train_model',arguments={
            'pipeline':{'model':'lightgbm','hyperparameters':{'class_weight':{'UNKNOWN':2}}},
            'hypothesis':hypothesis,'budget_seconds':30})))
        assert len(c.records())==count and bad.get('status')=='rejected',bad
        assert bad['error_code']=='command:ValueError'
        good=payload(c.dispatch(ToolCommand(operation_id='good_weight',tool='train_model',arguments={
            'pipeline':{'model':'lightgbm','hyperparameters':{'class_weight':{'0':1,'1':3},'n_estimators':10}},
            'hypothesis':hypothesis,'budget_seconds':30})))
        assert good['result']['status']=='completed',good
        _,result=c.lookup(good['experiment']['experiment_id'])
        with (Path(result.artifacts['directory'])/'model.pkl').open('rb') as stream:model=cloudpickle.load(stream)
        assert model.models['answer'].class_weight=={0:1,1:3}
        assert c.budget.counter('llm_calls')==0
    finally:c.close()


def test_optuna_keeps_numeric_class_weights_in_trials_and_frozen_replay(tmp_path):
    task=task_for([0,1]);X=pd.DataFrame({'x':np.linspace(-2,2,100)},index=['r'+str(i) for i in range(100)])
    y=pd.DataFrame({'answer':(X.x>0).astype(int)},index=X.index)
    spec=PipelineSpec(engine='optuna',model='lightgbm',hyperparameters={'n_estimators':8,'class_weight':{'0':1,'1':3}},
        n_trials=2,search_space={'num_leaves':{'type':'categorical','choices':[3,5]}})
    search=build_pipeline(spec,task,{});summary=search.fit(X,y,context(task,tmp_path))
    assert any(row['state']=='completed' for row in summary.inner_scores['trials'])
    assert search.pipeline.models['answer'].class_weight=={0:1,1:3}
    frozen=search.frozen_spec();assert frozen.hyperparameters['class_weight']=={'0':1,'1':3}
    replay=build_pipeline(frozen,task,{})
    replay.fit(X,y,{**context(task,tmp_path),'execution_mode':'refit'})
    assert replay.models['answer'].class_weight=={0:1,1:3}
