import json
import time

import httpx
import pytest

from automl_agent.budget import Limits
from automl_agent.controller import Controller
from automl_agent.qwen import QwenUnavailable
from conftest import write_task


def finish():
    return {'choices':[{'finish_reason':'stop','message':{'tool_calls':[{'id':'done','type':'function',
        'function':{'name':'finish','arguments':json.dumps({'operation_id':'done','reason':'Bounded transport test completed','result_ids':['baseline'],
            'skipped_directions': {'all': 'This fixed transport fixture exercises completion after channel recovery only.'}})}}]}}]}


@pytest.mark.parametrize('failure',[httpx.ReadTimeout,httpx.ReadError,httpx.WriteError,httpx.RemoteProtocolError])
def test_sent_unknown_request_is_not_repeated_and_stays_latched_on_resume(tmp_path,monkeypatch,failure):
    monkeypatch.setenv('OPENAI_API_KEY','offline');monkeypatch.setenv('OPENAI_BASE_URL','http://offline')
    c=Controller(write_task(tmp_path),Limits(seconds=90,cpu=1,max_calls=2));calls=[]
    c.prepare();c.qwen.client.close()
    def request(wire):
        calls.append(wire)
        if len(calls)==1:raise failure('synthetic outcome lost',request=wire)
        return httpx.Response(200,json=finish())
    c.qwen.client=httpx.Client(transport=httpx.MockTransport(request))
    monkeypatch.setattr(c,'recover_search',lambda error:False)
    try:
        result=c.run()
        assert len(calls)==1
        saved=c.journal.get('qwen_unknown_outcome');assert saved['call_id']==1 and saved['error_type']==failure.__name__
        assert result['control_status']==result['refit_status']=='completed'
        c.qwen.close()
        from automl_agent.qwen import QwenClient
        resumed=QwenClient(c.journal,c.budget);resumed.client.close();resumed.client=httpx.Client(transport=httpx.MockTransport(request))
        try:
            with pytest.raises(QwenUnavailable):resumed._request([{'role':'user','content':'No blind retry'}])
            assert len(calls)==1
        finally:resumed.close()
    finally:c.close()


@pytest.mark.parametrize('failure',[httpx.ConnectError,httpx.ConnectTimeout,httpx.PoolTimeout])
def test_confirmed_unsent_request_can_retry(tmp_path,monkeypatch,failure):
    monkeypatch.setenv('OPENAI_API_KEY','offline');monkeypatch.setenv('OPENAI_BASE_URL','http://offline')
    c=Controller(write_task(tmp_path),Limits(seconds=90,cpu=1,max_calls=3));calls=[];c.prepare();c.qwen.client.close()
    def request(wire):
        calls.append(wire)
        if len(calls)==1:raise failure('synthetic connection not established',request=wire)
        return httpx.Response(200,json=finish())
    c.qwen.client=httpx.Client(transport=httpx.MockTransport(request))
    try:
        result=c.run()
        assert len(calls)==2 and not c.journal.get('qwen_unknown_outcome') and result['refit_status']=='completed'
    finally:c.close()


def test_inflight_from_killed_process_blocks_new_generation(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','offline');monkeypatch.setenv('OPENAI_BASE_URL','http://offline')
    c=Controller(write_task(tmp_path),Limits(seconds=90,cpu=1));c.prepare()
    c.journal.put('llm_inflight',{'started':time.time()-1,'transport':'native','call_id':7})
    c.qwen.client.close();c.qwen.client=httpx.Client(transport=httpx.MockTransport(lambda r:pytest.fail('Unknown pending generation must not resend')))
    try:
        with pytest.raises(QwenUnavailable):c.qwen._request([{'role':'user','content':'resume'}])
        assert c.journal.get('qwen_unknown_outcome')['call_id']==7 and c.journal.get('llm_calls',0)==0
    finally:c.close()


@pytest.mark.parametrize('interruption',[KeyboardInterrupt,SystemExit])
def test_explicit_interrupt_keeps_unknown_outcome(tmp_path,monkeypatch,interruption):
    monkeypatch.setenv('OPENAI_API_KEY','offline');monkeypatch.setenv('OPENAI_BASE_URL','http://offline')
    c=Controller(write_task(tmp_path),Limits(seconds=90,cpu=1));c.prepare();c.qwen.client.close()
    def interrupted(request):raise interruption('synthetic cancellation')
    c.qwen.client=httpx.Client(transport=httpx.MockTransport(interrupted))
    try:
        with pytest.raises(interruption):c.qwen._request([{'role':'user','content':'cancel'}])
        assert c.journal.get('llm_inflight') is None
        assert c.journal.get('qwen_unknown_outcome')['reason']=='request_interrupted'
    finally:c.close()
