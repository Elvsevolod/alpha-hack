import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
from automl_agent.budget import Limits
from automl_agent.contracts import FeedbackPolicy,MetricSpec,OutputSpec,SelectionPolicy,TaskResolutionProposal,TaskSpec,ToolCommand,ValidationPlan
from automl_agent.controller import Controller
from automl_agent.data import ContractUnresolved,Dataset
from automl_agent.evaluation import feedback_reports
from automl_agent.finalists import choose_control,select_finalists
from automl_agent.tool_results import payload
from conftest import write_task


def proposal(dataset,roles,**updates):
    spec=dict(contract_version=2,file_roles=roles,kind='regression',targets=['answer'],id_columns=['row'],
        feature_columns=['value','empty','code'],categorical_columns=['code'],metric={'name':'rmse'},
        output={'columns':['row','answer'],'prediction_type':'values'},source_hash=dataset.source_hash)
    spec.update(updates)
    return TaskResolutionProposal(request_id='r',source_hash=dataset.source_hash,status='proposed',task_spec=spec,
        evidence={field:['task','inventory'] for field in spec})


def test_inventory_and_semantic_roles_before_loading(tmp_path):
    path=write_task(tmp_path);(tmp_path/'task_spec.json').unlink()
    nested=tmp_path/'input';nested.mkdir()
    roles={}
    for original,new in [('train.csv','observations.csv'),('test.csv','future.csv'),('sample_submission.csv','example.csv')]:
        (tmp_path/original).rename(nested/new);roles[original]='input/'+new
    train=pd.read_csv(nested/'observations.csv',dtype={'row':str});train['internal_note']='ignore';train.to_csv(nested/'observations.csv',index=False)
    test=pd.read_csv(nested/'future.csv',dtype={'row':str});test['answer']=0;test['internal_note']='ignore';test.to_csv(nested/'future.csv',index=False)
    (tmp_path/'submission.csv').write_text('invalid output')
    (tmp_path/'outputs').mkdir();(tmp_path/'outputs/leak.csv').write_text('private,label\na,b')
    path.write_text('Regression answer from value, empty, code. row is ID; internal_note is service data. Auxiliary MAE; primary evaluation uses RMSE. Output row,answer. future answer is a placeholder.')
    dataset=Dataset(path,deferred=True)
    assert not hasattr(dataset,'train') and len(dataset.catalog)==3
    task=dataset.confirm_proposal(proposal(dataset,roles,service_columns=['internal_note']), 'r')
    dataset.activate(task)
    assert task.targets==['answer'] and len(dataset.test)==25 and 'internal_note' not in dataset.features()
    assert not any('leak.csv' in value for value in dataset.catalog)


def test_no_llm_requires_explicit_semantics_and_sources_are_validated(tmp_path):
    path=write_task(tmp_path);(tmp_path/'task_spec.json').unlink();dataset=Dataset(path,deferred=True)
    with pytest.raises(ContractUnresolved):dataset.explicit()
    roles={'train.csv':'../train.csv','test.csv':'test.csv'}
    with pytest.raises(ValueError,match='outside'):dataset.confirm_proposal(proposal(dataset,roles),'r')
    unresolved=TaskResolutionProposal(request_id='r',source_hash=dataset.source_hash,status='unresolved',unresolved_fields=['file_roles'])
    with pytest.raises(ContractUnresolved):dataset.confirm_proposal(unresolved,'r')


def test_semantic_llm_is_used_even_with_successful_heuristic(tmp_path,monkeypatch):
    path=write_task(tmp_path);(tmp_path/'task_spec.json').unlink()
    path.write_text('Regression, Primary metric RMSE. Features value, empty and code; code categorical. row ID. Predict answer. Output sample_submission.csv.')
    c=Controller(path,Limits(seconds=120,cpu=1));calls=[]
    monkeypatch.setattr(c.qwen,'key','offline');monkeypatch.setattr(c.qwen,'base_url','http://offline')
    def resolve(system,request):
        calls.append(request)
        assert not hasattr(c.dataset,'train') and c.journal and c.budget
        schema=request['schema']['$defs']['TaskSpec']
        assert schema['properties']['contract_version']=={'type':'integer','const':2}
        assert 'contract_version' in schema['required']
        return proposal(c.dataset,{n:n for n in ('train.csv','test.csv','sample_submission.csv')}).model_copy(update={'request_id':request['request_id']}).model_dump(mode='json')
    monkeypatch.setattr(c.qwen,'resolve',resolve)
    try:assert c._task().contract_version==2 and len(calls)==1
    finally:c.close()


def test_finish_with_one_result_early_and_irreversible(tmp_path):
    c=Controller(write_task(tmp_path),Limits(seconds=3600,cpu=1))
    try:
        c.prepare()
        result=payload(c.dispatch(ToolCommand(operation_id='enough',tool='finish',arguments={
            'reason':'Baseline is sufficient for this bounded offline fixture; no justified next hypothesis', 'result_ids':['baseline']})))
        assert result['status']=='frozen' and c.budget.remaining>3000
        assert c.journal.get('imputation_skip_reason')
        with pytest.raises(ValueError,match='state searching'):
            c.dispatch(ToolCommand(operation_id='later',tool='get_dataset_report',arguments={'report':'missing'}))
        assert c.state=='frozen'
    finally:c.close()


def class_case(counts,metric='f1'):
    labels=[str(i) for i in range(len(counts))];y=pd.DataFrame({'target':np.repeat(labels,counts)})
    X=pd.DataFrame({'x':np.ones(len(y))});task=TaskSpec(kind='classification',targets=['target'],feature_columns=['x'],
        metric=MetricSpec(name=metric),output=OutputSpec(columns=['target'],prediction_type='labels'),classes={'target':labels},source_hash='s')
    plan=ValidationPlan(strategy='random',reason='fixture',fit=[0],validation=[1],control=[2],data_version='s',
        feedback_policy=FeedbackPolicy(report_ids=['classes','missingness'],class_order=['target:'+label for label in labels]),selection_policy=SelectionPolicy(maximize=True))
    return task,plan,X,y


def test_class_metrics_use_full_confusion_and_no_counts():
    task,plan,X,y=class_case([40,40,40]);p=y.target.to_numpy().copy();p[:10]='1';p[40:45]='2'
    report=feedback_reports(task,plan,X,y,{'target':p},'labels')['classes'];rows=report['classes']
    assert rows[0]['precision']==1 and rows[0]['recall']==.75 and rows[0]['f1']==pytest.approx(60/70)
    assert rows[1]['precision']==pytest.approx(35/45) and rows[1]['recall']==.875
    assert all(set(row)=={'id','status','reason','support_category','precision','recall','f1'} for row in rows)


def test_ten_class_macro_f1_never_exposes_nine_to_reconstruct_small_tenth():
    task,plan,X,y=class_case([2]+[40]*9)
    for predictions in [y.target.to_numpy(), np.repeat('0',len(y)),y.target.to_numpy()[::-1]]:
        reports=feedback_reports(task,plan,X,y,{'target':predictions},'labels')
        rows=reports['classes']['classes']
        assert all(not {'precision','recall','f1'} & set(row) for row in rows)
        assert rows[1]['reason']=='complementary_disclosure'
        assert reports['missingness']['status']=='suppressed'
    assert plan.feedback_policy.class_order[1]=='target:1'


@pytest.mark.parametrize('maximize,scores',[(True,{'baseline':.8000,'leader':.8009,'third':.8018}),
    (False,{'baseline':.8018,'leader':.8009,'third':.8000})])
def test_control_uses_exact_best_and_is_order_independent(maximize,scores):
    policy=SelectionPolicy(maximize=maximize,min_delta_abs=.001)
    for order in itertools.permutations(scores):
        winner,audit=choose_control({i:scores[i] for i in order},'baseline',policy,{'baseline':0,'leader':2,'third':3},{i:0 for i in scores})
        assert winner=='leader' and 'baseline' not in audit['tie_set']
    exact={i:.8 for i in scores}
    assert choose_control(exact,'baseline',policy,{i:1 for i in scores},{i:0 for i in scores})[0]=='baseline'
    relative=SelectionPolicy(maximize=maximize,min_delta_abs=0,min_delta_rel=.001)
    assert 'baseline' not in choose_control(scores,'baseline',relative,{i:1 for i in scores},{i:0 for i in scores})[1]['tie_set']


def test_third_selection_dominance_ranks_and_recipe_deduplication():
    policy=SelectionPolicy(maximize=True)
    rows=[{'id':i,'identity':i,'score':score,'windows':dict(zip(['a','b'],windows)),'cost':cost,'created_index':index}
        for index,(i,score,windows,cost) in enumerate([
            ('baseline',.7,[.7,.7],1),('leader',.8,[.8,.6],2),('dominated',.69,[.6,.6],1),
            ('stable',.79,[.75,.75],2),('risky',.78,[.9,.65],1)])]
    selected,audit=select_finalists(rows,'baseline','leader',policy,['a','b'])
    assert selected==['baseline','leader','stable'] and 'dominated' not in [r['id'] for r in audit['pool']]
    rows[1]['identity']='baseline'
    selected,_=select_finalists(rows,'baseline','leader',policy,['a','b'])
    assert 'leader' not in selected and len(selected)<=3


def test_temporal_windows_retrain_pipeline_on_past_only(tmp_path,monkeypatch):
    path=write_task(tmp_path,temporal=True)
    frame=pd.read_csv(tmp_path/'train.csv',dtype={'row':str});frame['when']=np.repeat(np.arange(1,6),36);frame.to_csv(tmp_path/'train.csv',index=False)
    frame=pd.read_csv(tmp_path/'test.csv',dtype={'row':str});frame['when']=6;frame.to_csv(tmp_path/'test.csv',index=False)
    c=Controller(path,Limits(seconds=180,cpu=1));seen=[];original=c.executor.run
    def run(directory,request,seconds):
        if directory.parent.parent.name=='stability':
            train=pd.read_json(directory/'fit_X.json',orient='table') if False else json.loads((directory/'fit_X.json').read_text())
            seen.append((request,train))
        return original(directory,request,seconds)
    monkeypatch.setattr(c.executor,'run',run)
    try:
        c.prepare();windows=c.stability_windows();assert len(windows)==2
        for window in windows:
            assert max(c.dataset.train.iloc[window['fit']]['when'])<min(c.dataset.train.iloc[window['validation']]['when'])
            assert set(window['fit']+window['validation'])<=set(c.plan.fit)
        result=c.lookup('baseline')[1]
        assert result.training_summary['temporal_stability']['status']=='complete'
        assert len(seen)==2 and all(request['mode']=='train' for request,_ in seen)
    finally:c.close()


def test_probability_contract_with_placeholder_and_group_time_roles(tmp_path):
    path=write_task(tmp_path,kind='classification',probability=True,temporal=True,groups=True)
    dataset=Dataset(path,deferred=True)
    spec=json.loads((tmp_path/'task_spec.json').read_text())
    spec.update(contract_version=2,file_roles={n:n for n in ('train.csv','test.csv','sample_submission.csv')},source_hash=dataset.source_hash)
    path.write_text('Classification probability output, log loss, when is time and query is group; positive and negative are classes.')
    dataset=Dataset(path,deferred=True);spec['source_hash']=dataset.source_hash
    response=TaskResolutionProposal(request_id='r',source_hash=dataset.source_hash,status='proposed',task_spec=spec,
        evidence={field:['task','inventory'] for field in spec})
    task=dataset.confirm_proposal(response,'r');dataset.activate(task)
    assert task.output.prediction_type=='probabilities' and task.time_column=='when' and task.group_column=='query'


def test_multioutput_semantics_require_supported_aggregation(tmp_path):
    path=write_task(tmp_path,multilabel=True);(tmp_path/'task_spec.json').unlink()
    dataset=Dataset(path,deferred=True)
    response=proposal(dataset,{n:n for n in ('train.csv','test.csv','sample_submission.csv')},kind='multilabel',targets=['answer','other'],
        metric={'name':'f1','average':'macro'},output={'columns':['row','answer','other'],'prediction_type':'labels'})
    with pytest.raises(ValueError,match='aggregation'):dataset.confirm_proposal(response,'r')
    path.write_text('Multilabel answer and other. Macro F1 uses uniform target aggregation. row ID; value empty code features.')
    dataset=Dataset(path,deferred=True);response.source_hash=dataset.source_hash;response.task_spec.source_hash=dataset.source_hash
    assert dataset.confirm_proposal(response,'r').targets==['answer','other']


def test_budget_reduction_drops_third_then_leader_before_control(tmp_path,monkeypatch):
    c=Controller(write_task(tmp_path),Limits(seconds=180,cpu=1))
    try:
        c.prepare();c.freeze('budget fixture')
        saved=c.journal.get('finalization_plan');saved['finalists']=['baseline','leader','third'];c.journal.put('finalization_plan',saved)
        calls=[]
        def estimate(controller,identifiers,*args):
            calls.append(tuple(identifiers));return {'seconds':10000*len(identifiers),'disk_bytes':0,'node_seconds':{i:10000 for i in identifiers}}
        monkeypatch.setattr('automl_agent.completion.estimate',estimate)
        c.finalize();saved=c.journal.get('finalization_plan')
        assert saved['finalists']==['baseline'] and [r['id'] for r in saved['excluded_before_control']]==['third','leader']
        assert ('baseline','leader') in calls and ('baseline',) in calls and not c.journal.get('control_opened')
    finally:c.close()


def test_target_placeholder_cannot_become_structural_feature(tmp_path):
    path=write_task(tmp_path);dataset=Dataset(path,deferred=True)
    with pytest.raises(ValueError,match='structural'):
        proposal(dataset,{'train.csv':'train.csv','test.csv':'test.csv'},time_column='answer')


def test_ensemble_windows_retrain_all_dependencies_and_optional_failure_is_limited(tmp_path,monkeypatch):
    path=write_task(tmp_path,temporal=True)
    frame=pd.read_csv(tmp_path/'train.csv',dtype={'row':str});frame['when']=np.repeat(np.arange(1,6),36);frame.to_csv(tmp_path/'train.csv',index=False)
    frame=pd.read_csv(tmp_path/'test.csv',dtype={'row':str});frame['when']=6;frame.to_csv(tmp_path/'test.csv',index=False)
    c=Controller(path,Limits(seconds=240,cpu=1))
    try:
        c.prepare()
        c.dispatch(ToolCommand(operation_id='linear',tool='train_model',arguments={'pipeline':{'model':'ridge'},'budget_seconds':30,
            'hypothesis':{'observation':'Compare baseline','expected_effect':'Measure linear transfer','changed_component':'model','acceptance':'Fixed RMSE'}}))
        member=c.journal.operation('linear').experiment_id
        result=payload(c.dispatch(ToolCommand(operation_id='blend',tool='build_ensemble',arguments={
            'members':['baseline',member],'budget_seconds':30,'hypothesis':{'observation':'Two distinct fitted models','expected_effect':'Measure uniform blend','changed_component':'ensemble','acceptance':'Fixed RMSE'}})))
        assert result['result']['status']=='completed'
        identifier=c.journal.operation('blend').experiment_id;spec,measured=c.lookup(identifier)
        assert measured.training_summary['temporal_stability']['status']=='complete'
        paths=list((c.workdir/'stability'/identifier).rglob('fit_y.json'))
        assert len(paths)==4
        for path in paths:
            assert c.task.targets==['answer']
        monkeypatch.setattr(c,'window_pipeline',lambda *a,**k:(_ for _ in ()).throw(ValueError('synthetic undefined window metric')))
        limited=c.temporal_stability(spec,time.time()+30)
        assert limited['status']=='limited' and limited['diagnostic']['traceback_ref']
        assert c.lookup(identifier)[1].status=='completed'
    finally:c.close()
