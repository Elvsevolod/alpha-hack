"""Counterfactual HTTP fixtures verify the decision channel receives changed facts.

The decision rule belongs only to this test model, never the production agent.
This tests feedback plumbing, not the quality of a live model's reasoning.
"""
import json

import httpx
import pandas as pd
import pytest

from automl_agent.eda.agent import EDAAgent
from conftest import write_task


@pytest.mark.parametrize('policy',['free','evidence','alternatives'])
@pytest.mark.parametrize('has_missing',[False,True])
def test_changed_profile_can_select_a_different_next_analysis(tmp_path,monkeypatch,has_missing,policy):
    monkeypatch.setenv('DEEPSEEK_API_KEY','offline-fixture')
    task=write_task(tmp_path/'input')
    spec=json.loads((task.parent/'task_spec.json').read_text())
    spec['feature_columns'].remove('empty')
    for filename in ['train.csv','test.csv']:
        frame=pd.read_csv(task.parent/filename,keep_default_na=False,dtype=str).drop(columns=['empty'])
        if not has_missing:frame['value']=frame['value'].replace('',0)
        frame.to_csv(task.parent/filename,index=False)
    (task.parent/'task_spec.json').write_text(json.dumps(spec))
    observations=[]
    def response(request):
        body=json.loads(request.content);context=json.loads(body['messages'][1]['content'])
        if 'request_id' in context:
            contract={**spec,'source_hash':context['source_hash'],'contract_version':2,
                'file_roles':{r:r for r in ['train.csv','test.csv','sample_submission.csv']}}
            proposal={'request_id':context['request_id'],'source_hash':context['source_hash'],
                'status':'proposed','task_spec':contract,'evidence':{k:['task','inventory'] for k in contract}}
            message={'content':json.dumps(proposal)}
        elif not context['recent_results']:
            assert ('decision_ledger' in context)==(policy=='evidence')
            count=context['profile']['summary']['columns_with_missing']
            assert bool(count)==has_missing
            report='missing' if count else 'correlations'
            observations.append(report)
            tool='get_dataset_report';args={'operation_id':'question','report':report,'columns':['value'],
                'knowledge_gap':'Measure the issue indicated by the current profile.'}
            message={'tool_calls':[{'id':'question','function':{'name':tool,'arguments':json.dumps(args)}}]}
        else:
            if policy=='evidence':
                entry=context['decision_ledger']['entries'][-1]
                assert entry['operation_id']=='question' and entry['result_available']
                assert entry['claim_support']=='unassessed'
            result=context['recent_results'][-1]['result']
            assert result['status']=='success'
            data=result['data']['report']
            assert ('missing_counts' in data) if has_missing else ('pair_counts' in data)
            observations.append('result_received')
            message={'tool_calls':[{'id':'finish','function':{'name':'finish','arguments':json.dumps({
                'operation_id':'finish','reason':'question confirms the measured aggregate. Model effect remains unknown; next compare a supported direct pipeline on the inner split.'})}}]}
        return httpx.Response(200,json={'choices':[{'message':message}],'usage':{'prompt_tokens':100,'completion_tokens':100}})
    agent=EDAAgent(task,tmp_path/'result',tmp_path/'MOCK_ledger.json',client=httpx.Client(transport=httpx.MockTransport(response)),policy=policy)
    agent.qwen.provider_mode='http_mock'
    try:result=agent.autonomous()
    finally:agent.close()
    assert result['state']=='completed',result
    assert observations==['missing' if has_missing else 'correlations','result_received']
    assert result['model_fits']==result['api_calls']==0
    assert result['model_generations']==3

    assert result['decision_policy']==policy
    assert result['decision_assessment']['status']=='unassessed'
    assert json.loads((tmp_path/'result'/'decision_ledger.json').read_text())['entries'][0]['operation_id']=='question'
