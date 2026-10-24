"""Organizer adapter accounting/transport is tested without real API or credentials."""
import json
import time

import httpx
import pytest

from automl_agent.budget import Budget,Limits
from automl_agent.journal import Journal
from automl_agent.qwen import QwenOutcomeUnknown,QwenUnavailable
from tools.qwen_eda_live import GatewayClient,GatewayLedger,limits


@pytest.mark.parametrize('failure',[False,True])
def test_organizer_adapter_settles_reported_units_and_never_retries_unknown(tmp_path,monkeypatch,failure):
    account={'spend':0.,'total_spend':0.,'max_budget':600.,'blocked':False}
    monkeypatch.setattr(limits,'fetch_info',lambda key:dict(account))
    before=limits.snapshot(account,'fake-key','before')
    ledger=GatewayLedger(tmp_path/'costs.json','fake-key',before,'run',cap=.25)
    journal=Journal(tmp_path/'run');budget=Budget(journal,Limits(seconds=120,max_calls=7),time.time())
    bodies=[]
    def respond(request):
        body=json.loads(request.content);bodies.append(body)
        assert body['reasoning']=={'enabled':True,'effort':'medium'} and body['max_tokens']==8192
        assert 'thinking' not in body and body['response_format']=={'type':'json_object'}
        if failure:return httpx.Response(500,json={'error':'controlled failure'})
        account.update(spend=.25,total_spend=.25)
        return httpx.Response(200,headers={'x-litellm-response-cost':'.25'},json={
            'choices':[{'finish_reason':'stop','message':{'content':'{"ok":true}'}}],
            'usage':{'prompt_tokens':100,'completion_tokens':10}})
    client=GatewayClient(journal,budget,ledger,{'key':'fake-key','model':'fixture-model','base_url':'https://fixture.invalid/v1'},
        client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        if failure:
            with pytest.raises(QwenOutcomeUnknown):client._request([{'role':'user','content':'check'}])
            assert ledger.summary()['outcome_unknown']
            with pytest.raises(QwenOutcomeUnknown):client._request([{'role':'user','content':'check'}])
        else:
            assert client._request([{'role':'user','content':'check'}])['content']=='{"ok":true}'
            assert ledger.summary()['gateway_units']==.25
            with pytest.raises(QwenUnavailable,match='threshold'):client._request([{'role':'user','content':'check'}])
        assert len(bodies)==client.sent_requests==1
        assert 'fake-key' not in ''.join(p.read_text() for p in (tmp_path/'run'/'provider').glob('*.json'))
    finally:client.close();journal.close()


def test_resolution_advertises_one_citation_location_with_exact_source_ids(tmp_path,monkeypatch):
    from automl_agent.contracts import TaskResolutionProposal
    from automl_agent.eda.deepseek import DeepSeekClient,CostLedger
    monkeypatch.setenv('DEEPSEEK_API_KEY','offline-only-key')
    journal=Journal(tmp_path/'run');budget=Budget(journal,Limits(seconds=60,max_calls=1),time.time())
    client=DeepSeekClient(journal,budget,CostLedger(tmp_path/'ledger.json','fixture'))
    request={'schema':TaskResolutionProposal.model_json_schema(),'sources':{
        'task':{'text':'task description'},'inventory':{'text':'{}'},'description_csv:fields.csv':{'text':'metadata'}}}
    def capture(messages,**kwargs):
        body=json.loads(messages[-1]['content'])
        assert 'evidence' not in body['schema']['$defs']['TaskSpec']['properties']
        assert 'evidence' in body['schema']['required']
        assert body['schema']['properties']['evidence']['additionalProperties']['items']['enum']==list(request['sources'])
        assert body['allowed_evidence_source_ids']==list(request['sources'])
        return {'content':'{"status":"unresolved"}'}
    monkeypatch.setattr(client,'_request',capture)
    try:
        client.resolve('Resolve from evidence.',request)
        assert 'evidence' in request['schema']['$defs']['TaskSpec']['properties']  # caller untouched
    finally:client.close();journal.close()


def test_historical_provider_cannot_generate_with_a_real_transport(tmp_path,monkeypatch):
    from automl_agent.eda.deepseek import DeepSeekClient,CostLedger
    monkeypatch.setenv('DEEPSEEK_API_KEY','offline-fixture')
    journal=Journal(tmp_path/'run');budget=Budget(journal,Limits(seconds=60,max_calls=1),time.time())
    client=DeepSeekClient(journal,budget,CostLedger(tmp_path/'ledger.json','fixture'))
    monkeypatch.setattr(client.client,'post',lambda *a,**kw:pytest.fail('Forbidden live request'))
    try:
        with pytest.raises(QwenUnavailable,match='disabled'):
            client._request([{'role':'user','content':'check'}])
        assert budget.counter('llm_calls')==0 and client.ledger.summary()['requests']==0
    finally:client.close();journal.close()
