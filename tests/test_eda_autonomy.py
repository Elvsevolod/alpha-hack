"""Development provider, autonomous feedback loop and resource boundaries; no live API."""
import json
import time

import httpx
import numpy as np
import pandas as pd
import pytest

from automl_agent.budget import Budget, Limits
from automl_agent.contracts import DatasetReportArgs, ToolCommand
from automl_agent.eda.agent import EDAAgent
from automl_agent.eda.deepseek import CostLedger, DeepSeekClient
from automl_agent.eda.runner import run
from automl_agent.journal import Journal
from automl_agent.profiling import correlations
from automl_agent.qwen import QwenOutcomeUnknown, QwenUnavailable
from conftest import write_task


def test_numeric_categories_never_receive_numeric_summaries(tmp_path):
    from automl_agent.data import Dataset
    task=write_task(tmp_path/'task')
    dataset=Dataset(task,deferred=True);dataset.activate(dataset.explicit())
    dataset.task.categorical_columns.append('value')
    p=dataset.profile(list(range(len(dataset.train))))
    item=p.columns['value']
    assert item['semantic_type']=='categorical' and item['most_common_fit']
    assert not {'mean','median','std','q99'} & set(item)


def test_nonlinear_relation_and_support_are_not_overclaimed():
    x=np.linspace(-3,3,1000)
    report=correlations(pd.DataFrame({'x':x,'square':x*x,'cube':x*x*x}))
    assert abs(report['correlations']['x']['square'])<1e-12
    assert report['spearman']['x']['cube']==pytest.approx(1)
    assert report['pair_counts']['x']['cube']==1000


def test_expired_original_budget_returns_partial_without_worker(tmp_path,monkeypatch):
    task=write_task(tmp_path/'task')
    monkeypatch.setattr('automl_agent.executor.Executor.run',lambda *a: pytest.fail('expired work started'))
    result=run(task,tmp_path/'result',seconds=.000001)
    assert result['state']=='partial'
    assert result['profile']['status']=='budget_too_small'


def test_usd_reservations_are_durable_and_unknown_stops_recovery(tmp_path):
    path=tmp_path/'ledger.json';ledger=CostLedger(path,'test',cap=.01)
    index=ledger.reserve({'messages':[{'content':'small'}]})
    with pytest.raises(QwenOutcomeUnknown):CostLedger(path,'other').reserve({})
    ledger.settle(index,{'prompt_tokens':10,'completion_tokens':20})
    assert CostLedger(path,'test').summary()['usd']==pytest.approx(.000027)
    with pytest.raises(QwenUnavailable):ledger.reserve({'content':'x'*100000})


def test_provider_sends_no_qwen_fields_and_retains_failed_reservation(tmp_path,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','private-fixture')
    journal=Journal(tmp_path/'run');budget=Budget(journal,Limits(seconds=120,max_calls=7),time.time())
    ledger=CostLedger(tmp_path/'ledger.json','probe',cap=.05)
    def response(request):
        body=json.loads(request.content)
        assert body['thinking']=={'type':'disabled'} and body['max_tokens']==2048
        assert not {'top_k','min_p','reasoning','repetition_penalty'} & set(body)
        return httpx.Response(500,json={'error':'unavailable'})
    client=DeepSeekClient(journal,budget,ledger,client=httpx.Client(transport=httpx.MockTransport(response)))
    with pytest.raises(QwenOutcomeUnknown):client._request([{'role':'user','content':'hello'}])
    assert ledger.summary()['outcome_unknown']
    assert 'private-fixture' not in (tmp_path/'ledger.json').read_text()
    client.close();journal.close()


@pytest.mark.parametrize('transport',['native','json'])
def test_http_autonomous_profile_feedback_fit_finish_and_recovery(tmp_path,monkeypatch,transport):
    monkeypatch.setenv('DEEPSEEK_API_KEY','private-fixture')
    task=write_task(tmp_path/'task');seen=[]
    commands=[
        {'operation_id':'corr','tool':'get_dataset_report','arguments':{'report':'correlations','columns':['value','answer'],'include_target':True}},
        {'operation_id':'diagnosis','tool':'get_dataset_report','arguments':{'report':'errors','include_target':True}},
        {'operation_id':'experiment','tool':'train_model','arguments':{'pipeline':{'model':'ridge'},
            'hypothesis':{'observation':'diagnosis baseline measured errors','expected_effect':'compare ridge capacity',
                'changed_component':'model','acceptance':'lower inner RMSE'},'budget_seconds':30}},
        {'operation_id':'finish','tool':'finish','arguments':{'reason':'corr and diagnosis confirm an association; causality unknown. Next test log1p with expected tail improvement on same inner split.'}},
    ]
    def response(request):
        body=json.loads(request.content);seen.append(body)
        content=json.loads(body['messages'][1]['content'])
        if 'request_id' in content:
            spec=json.loads((task.parent/'task_spec.json').read_text())
            spec.update(contract_version=2,source_hash=content['source_hash'],file_roles={r:r for r in ['train.csv','test.csv','sample_submission.csv']})
            proposal={'request_id':content['request_id'],'source_hash':content['source_hash'],'status':'proposed','task_spec':spec,
                'evidence':{k:['task','inventory'] for k in spec}}
            message={'content':json.dumps(proposal)}
        else:
            command=commands.pop(0)
            if command['operation_id']=='diagnosis':
                assert 'pair_counts' in json.dumps(content['recent_results'])
            if command['operation_id']=='finish':
                assert 'outer_validation_used' in json.dumps(content['recent_results'])
            if transport=='native':
                assert set(t['function']['name'] for t in body['tools'])=={'get_dataset_report','search_knowledge','list_knowledge','read_knowledge','search_run_records','list_run_records','read_run_records','train_model','finish'}
                message={'tool_calls':[{'id':command['operation_id'],'function':{'name':command['tool'],
                    'arguments':json.dumps({'operation_id':command['operation_id'],**command['arguments']})}}]}
            else:
                assert set(content['tool_argument_schemas'])=={'get_dataset_report','search_knowledge','list_knowledge','read_knowledge','search_run_records','list_run_records','read_run_records','train_model','finish'}
                message={'content':json.dumps(command)}
        return httpx.Response(200,json={'choices':[{'message':message}],'usage':{'prompt_tokens':100,'completion_tokens':100}})
    output=tmp_path/'result';ledger=tmp_path/'ledger.json'
    agent=EDAAgent(task,output,ledger,transport=transport,client=httpx.Client(transport=httpx.MockTransport(response)))
    # Override conftest's Qwen resolver stub: exercise the actual development HTTP channel.
    monkeypatch.setattr(agent.qwen,'resolve',DeepSeekClient.resolve.__get__(agent.qwen))
    result=agent.autonomous()
    assert result['state']=='completed',result
    assert result['model_fits']==2 and result['api_calls']==5
    assert not list(output.rglob('submission.csv'))
    assert 'private-fixture' not in (output/'trajectory.json').read_text()
    events=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
    assert {'base_profile','decision','tool_result','provider_request','provider_response','stop'} <= {item['phase'] for item in events}
    assert len([item for item in events if item['phase']=='decision'])==4
    for snapshot in (output/'provider').glob('*.json'):
        assert 'private-fixture' not in snapshot.read_text() and 'Authorization' not in snapshot.read_text()
    for tool in ['run_automl','tune_model','run_custom_experiment','build_ensemble','revise_validation']:
        with pytest.raises(ValueError,match='disabled'):agent.dispatch(ToolCommand(operation_id='blocked',tool=tool,arguments={}))
    agent.journal.put('eda_fits',2)
    assert agent.inner_experiment()['status']=='budget_too_small'
    # Diagnostic replay never consumes another fit, including paginated delivery.
    agent.run_tools.dataset(DatasetReportArgs(report='errors',include_target=True))
    assert agent.budget.counter('eda_fits')==2
    agent.close()
    recovered=EDAAgent(task,output,ledger,transport=transport,client=httpx.Client(transport=httpx.MockTransport(lambda req:pytest.fail('Finished run resent request'))))
    assert recovered.autonomous()['state']=='completed'
    recovered.close()


@pytest.mark.parametrize("change", [{"engine":"custom"}, {"fit_diagnostics":"inner_holdout"},
    {"prediction_policy":"calibrated","postprocess_fit_policy":"inner_holdout"}, {"hyperparameters":{"num_threads":100}},
    {"imputation":{"numeric":"iterative_trees"}}])
def test_dispatch_and_worker_share_single_fit_policy(change):
    from automl_agent.contracts import PipelineSpec
    from automl_agent.eda.policy import validate_pipeline
    spec=PipelineSpec(**change)
    with pytest.raises(ValueError):validate_pipeline(spec)
    with pytest.raises(ValueError):EDAAgent.validate_pipeline(spec)


def test_failed_report_pages_preserve_analysis_status_and_failed_operation(tmp_path,monkeypatch):
    from automl_agent.validation import choose_validation
    task=write_task(tmp_path/'task')
    agent=EDAAgent(task,tmp_path/'result',tmp_path/'ledger.json')
    try:
        agent.task=agent.dataset.activate(agent.dataset.explicit())
        agent.plan=choose_validation(agent.dataset)
        agent.journal.transition('searching','offline fixture')
        calls=[]
        def failed(*args, **kwargs):
            calls.append(1)
            return {'status':'technical_failure','diagnostic_status':'timeout','error':'unknown '*1000}
        monkeypatch.setattr(agent.eda,'correlation_report',failed)
        args={'report':'correlations','columns':['value'],'max_chars':2600}
        first=agent.dispatch(ToolCommand(operation_id='failed_report',tool='get_dataset_report',arguments=args))
        assert first['status']==first['data']['analysis_status']=='technical_failure', first
        assert first['data']['delivery_status']=='partial'
        assert agent.journal.operation('failed_report').status=='failed'
        second=agent.dispatch(ToolCommand(operation_id='failed_page',tool='get_dataset_report',arguments=first['continuation']['arguments']))
        assert second['status']==second['data']['analysis_status']=='technical_failure'
        assert second['data']['range']['start']==first['data']['range']['end']
        assert len(calls)==1
        assert agent.journal.operation('failed_page').status=='failed'
    finally:agent.close()


def test_series_connection_failure_writes_all_four_stops_without_task_calls(tmp_path,monkeypatch):
    from tools.run_eda_series import series
    def fail(client,*args,**kwargs):
        client.ledger.reserve({'messages':[{'content':'probe'}]})
        raise QwenOutcomeUnknown('simulated unknown connection')
    monkeypatch.setattr(DeepSeekClient,'_request',fail)
    monkeypatch.setattr('tools.run_eda_series.run_agent',lambda *a,**k:pytest.fail('Task started after unknown connection'))
    assert series(tmp_path)=={}
    assert (tmp_path/'REPORT.md').read_text().count('not run')==4
    assert json.loads((tmp_path/'cost_summary.json').read_text())['unknown_reservations']==1
    with pytest.raises(ValueError,match='already started'):series(tmp_path)
