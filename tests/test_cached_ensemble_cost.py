"""A self-contained prepared blend needs no work from its completed children."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from automl_agent.completion import estimate
from automl_agent.controller import Controller
from automl_agent.contracts import PipelineSpec
from automl_agent.util import atomic_json
from test_completion_recipe_identity import cache_fixture


def graph_fixture(tmp_path):
    c,_,rows,directory,data,saved=cache_fixture(tmp_path)
    records={}
    leaf=PipelineSpec(model='lightgbm',hyperparameters={'n_estimators':3})
    for name in ['left','right']:
        attempt=tmp_path/'attempts'/name;attempt.mkdir()
        (attempt/'model.pkl').write_bytes(b'fixed synthetic prototype')
        records[name]=(SimpleNamespace(pipeline=leaf),SimpleNamespace(status='completed',seconds=2,
            training_summary={'rows':5,'seconds':2},artifacts={'directory':str(attempt)}))
    root=PipelineSpec(engine='ensemble',model='blend',members=['left','right'],ensemble_weights=[.5,.5])
    spec,result=c.lookup('candidate');spec.pipeline=root
    records['candidate']=(spec,result)
    data['finalization_plan']={'policies':{name:{'experiment_id':name,'pipeline_hash':s.pipeline.hash,
        'frozen_pipeline':s.pipeline.model_dump(mode='json')} for name,(s,_) in records.items()}}
    saved['frozen_pipeline']=root.model_dump(mode='json');atomic_json(directory/'worker_result.json',saved)
    c.lookup=lambda name:records[name]
    return c,rows,directory,records,data,saved


@pytest.mark.parametrize('iterator',[False,True])
def test_prepared_ensemble_cost_matches_executor_without_child_preparations(tmp_path,monkeypatch,iterator):
    c,rows,directory,_,_,_=graph_fixture(tmp_path)
    monkeypatch.setattr('automl_agent.completion.measured_cost',lambda *a,**k:pytest.fail('A cached output must not forecast model training'))
    prepared,_=Controller._prepare_finalist(c,'candidate',rows,'control',.01)
    assert prepared==directory
    graph=estimate(c,iter(['candidate']) if iterator else ['candidate'],rows,'control')
    assert graph['seconds']==graph['disk_bytes']==0,graph
    assert set(graph['node_seconds'])=={'candidate'},graph


def test_explicit_child_request_is_not_hidden_by_a_ready_parent(tmp_path):
    c,rows,_,_,_,_=graph_fixture(tmp_path)
    graph=estimate(c,['candidate','left'],rows,'control')
    assert set(graph['node_seconds'])=={'candidate','left'},graph
    assert graph['node_seconds']['candidate']==0 and graph['node_seconds']['left']>0


@pytest.mark.parametrize('damage',['hash','rows','recipe'])
def test_unusable_root_cache_keeps_both_required_child_fits(tmp_path,damage):
    c,rows,directory,_,_,saved=graph_fixture(tmp_path)
    if damage=='hash':(directory/'control_predictions.json').write_text('changed')
    if damage=='rows':saved['fit_row_keys_hash']='foreign'
    if damage=='recipe':saved['frozen_pipeline']['ensemble_weights']=[.2,.8]
    atomic_json(directory/'worker_result.json',saved)
    graph=estimate(c,['candidate'],rows,'control')
    assert set(graph['node_seconds'])=={'left','right','candidate'}
    assert all(value>0 for value in graph['node_seconds'].values())


def test_shared_child_of_an_unprepared_root_is_still_counted_once(tmp_path):
    c,rows,_,records,data,_=graph_fixture(tmp_path)
    root=records['candidate'][0].pipeline.model_copy(update={'members':['left','candidate']})
    spec=SimpleNamespace(pipeline=root)
    result=deepcopy(records['candidate'][1]);records['other']=(spec,result)
    data['finalization_plan']['policies']['other']={'experiment_id':'other','pipeline_hash':root.hash,
        'frozen_pipeline':root.model_dump(mode='json')}
    graph=estimate(c,['candidate','other'],rows,'control')
    assert set(graph['node_seconds'])=={'candidate','left','other'},graph
    assert graph['node_seconds']['left']>0 and graph['node_seconds']['candidate']==0


def test_prepared_boundary_does_not_unprotect_dependency_prototypes_or_hide_cycles(tmp_path):
    from automl_agent.cleanup import protected_ids
    c,rows,_,records,data,_=graph_fixture(tmp_path)
    data['finalization_plan']['finalists']=['candidate']
    c.journal.incomplete_operations=lambda:[]
    assert set(estimate(c,['candidate'],rows,'control')['node_seconds'])=={'candidate'}
    assert protected_ids(c)=={'candidate','left','right'}
    assert set(estimate(c,['candidate'],rows,'refit')['node_seconds'])=={'candidate','left','right'}
    records['left'][0].pipeline=PipelineSpec(engine='ensemble',model='blend',members=['candidate','right'])
    with pytest.raises(ValueError,match='Cyclic'):
        estimate(c,['candidate'],rows,'control')


def test_real_controller_reuses_blend_then_refits_original_members(tmp_path,monkeypatch,record_property):
    import json
    import shutil
    import pandas as pd
    from automl_agent.budget import Limits
    from automl_agent.contracts import ToolCommand
    from automl_agent.tool_results import payload
    from automl_agent.util import read_json
    from conftest import write_task
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    c=Controller(write_task(tmp_path),Limits(seconds=240,cpu=1),run_id='cached_blend')
    h={'observation':'Offline cache integration fixture','expected_effect':'Avoid repeated member preparation',
        'changed_component':'Prepared blend reuse','acceptance':'One control opening, full refit and ordered CSV; no quality claim'}
    try:
        c.prepare();members=[]
        for i,alpha in enumerate([1,5]):
            result=payload(c.dispatch(ToolCommand(operation_id='ridge_'+str(i),tool='train_model',arguments={
                'pipeline':{'model':'ridge','hyperparameters':{'alpha':alpha}},'budget_seconds':35,'hypothesis':h})))
            assert result['result']['status']=='completed',result
            members.append(result['experiment']['experiment_id'])
        blend=payload(c.dispatch(ToolCommand(operation_id='blend',tool='build_ensemble',arguments={
            'members':members,'strategy':'uniform','hypothesis':h,'budget_seconds':30})))
        assert blend['result']['status']=='completed',blend
        identifier=blend['experiment']['experiment_id']
        # Pin a technical finalist; this is not a claim of autonomous selection.
        monkeypatch.setattr(c,'prospective_finalists',lambda:([identifier],{'fixture':'prepared blend reuse'}))
        c.freeze('Frozen integration fixture before opening control')
        rows=sorted(c.plan.fit+c.plan.validation)
        directory,_=c._prepare_finalist(identifier,rows,'control',60)
        retired=tmp_path/'retired_preparations';retired.mkdir()
        for member in members:
            shutil.move(str(c.workdir/'finalization'/'control'/member),str(retired/member))
        from automl_agent.cleanup import protected_ids
        assert set(members)<protected_ids(c)
        graph=estimate(c,[identifier],rows,'control')
        assert graph['seconds']==graph['disk_bytes']==0
        calls=[];original=c.executor.run
        def only_full_refits(path,request,seconds):
            assert request['mode']=='refit' and request['predict_splits']==['test']
            assert len(read_json(path/'fit_X.json')['index'])==len(c.dataset.train)
            calls.append(path.parent.name)
            return original(path,request,seconds)
        monkeypatch.setattr(c.executor,'run',only_full_refits)
        c.finalize()
        assert calls==['refit','refit']
        assert c.journal.get('control_opened')['finalists']==[identifier]
        assert c.journal.get('control_status')==c.journal.get('refit_status')=='completed'
        assert c.journal.get('publication')['scope']=='full_train_refit'
        output=pd.read_csv(c.publisher.output,dtype=str)
        assert len(output)==len(c.dataset.test)
        for column in c.task.id_columns:assert output[column].tolist()==c.dataset.test[column].astype(str).tolist()
        assert c.budget.counter('llm_calls')==0
        record_property('completion',json.dumps({'cached_node_seconds':graph['node_seconds'],
            'worker_phases':calls,'scope':c.journal.get('publication')['scope'],'rows':len(output)}))
    finally:c.close()
