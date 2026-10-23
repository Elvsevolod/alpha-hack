import json

import httpx
import pytest

from automl_agent.budget import Limits
from automl_agent.controller import Controller
from automl_agent.qwen import QwenOutcomeUnknown, QwenRetryableError, QwenUnavailable
from automl_agent.qwen_stream import IncompleteModelStream, ProviderStreamError, completion_from_sse
from conftest import write_task
from test_qwen_stream import client, chunk, frames
from test_qa_qwen_unknown_outcome import finish


def failed_stream(code='502'):
    # The discarded prefix is syntactically complete, but has no terminal choice.
    return frames([chunk({'content':'{"tool":"finish"}'}),
        {'error':{'code':code, 'message':'Provider disconnected unexpectedly'}}], done=False)


@pytest.mark.parametrize('code', [502, '502', 429, '503', '401'])
def test_framed_explicit_provider_failure_is_never_an_action(code):
    with pytest.raises(ProviderStreamError) as caught:
        completion_from_sse(failed_stream(code).splitlines())
    assert caught.value.code == int(code)


@pytest.mark.parametrize('error', [None, {}, [], {'code':'502'}, {'code':'502','message':''},
    {'code':'502x','message':'bad'}, {'code':True,'message':'bad'},
    {'code':502.0,'message':'bad'}, {'code':'200','message':'bad'}])
def test_malformed_error_does_not_unlock_unknown_generation(error):
    with pytest.raises(IncompleteModelStream):
        completion_from_sse(frames([{'error':error}], done=False).splitlines())


def test_unframed_error_at_eof_is_still_unknown():
    with pytest.raises(IncompleteModelStream):
        completion_from_sse(['data: {"error":{"code":"502","message":"bad"}}'])


@pytest.mark.parametrize('prefix', [
    {'usage':{'completion_tokens_details':[1]},'choices':[]},
    chunk({}, 'bogus'), chunk({}, []), chunk({'content':42}),
    {'choices':[{'index':1,'delta':{}}]},
])
def test_malformed_prefix_cannot_be_reclassified_as_known_failure(prefix):
    raw=frames([prefix,{'error':{'code':'502','message':'provider unavailable'}}], done=False)
    with pytest.raises(IncompleteModelStream):completion_from_sse(raw.splitlines())


def test_explicit_failure_accounts_attempt_and_time_without_native_fallback(client):
    calls=[]
    def reply(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, headers={'content-type':'text/event-stream'}, content=failed_stream())
    client.client=httpx.Client(transport=httpx.MockTransport(reply))
    with pytest.raises(QwenRetryableError):
        client.next_command('One action', {})
    assert len(calls)==1 and calls[0]['tools']
    assert client.transport=='auto' and client.journal.get('transport') is None
    assert client.journal.get('qwen_unknown_outcome') is None
    assert client.journal.get('llm_inflight') is None
    assert client.journal.get('llm_calls')==1 and client.journal.get('llm_seconds')>0
    assert client.journal.get('qwen_provider_failure_streak')==1


def test_three_explicit_failures_stop_without_fourth_send_even_after_client_resume(client):
    calls=[]
    def reply(request):
        calls.append(request)
        return httpx.Response(200, headers={'content-type':'text/event-stream'}, content=failed_stream())
    client.client=httpx.Client(transport=httpx.MockTransport(reply))
    for expected in (QwenRetryableError, QwenRetryableError, QwenUnavailable, QwenUnavailable):
        with pytest.raises(expected):client._request([{'role':'user','content':'One action'}])
    assert len(calls)==3 and client.journal.get('llm_calls')==3
    from automl_agent.qwen import QwenClient
    resumed=QwenClient(client.journal, client.budget)
    resumed.client.close();resumed.client=httpx.Client(transport=httpx.MockTransport(reply))
    try:
        with pytest.raises(QwenUnavailable):resumed._request([{'role':'user','content':'Resume'}])
        assert len(calls)==3
    finally:resumed.close()


@pytest.mark.parametrize('code', ['401', '402', '403', '400', '422'])
def test_explicit_nonretryable_failure_does_not_become_native_fallback(client, code):
    calls=[]
    def reply(request):
        calls.append(request)
        return httpx.Response(200, headers={'content-type':'text/event-stream'}, content=failed_stream(code))
    client.client=httpx.Client(transport=httpx.MockTransport(reply))
    with pytest.raises(QwenUnavailable):client.next_command('One action', {})
    assert len(calls)==1 and client.journal.get('qwen_unknown_outcome') is None


def test_a_later_unterminated_stream_still_latches_unknown(client):
    calls=[]
    def reply(request):
        calls.append(request)
        raw=failed_stream() if len(calls)==1 else frames([chunk({'content':'{}'})], done=False)
        return httpx.Response(200, headers={'content-type':'text/event-stream'}, content=raw)
    client.client=httpx.Client(transport=httpx.MockTransport(reply))
    with pytest.raises(QwenRetryableError):client._request([{'role':'user','content':'One action'}])
    for _ in range(2):
        with pytest.raises(QwenOutcomeUnknown):client._request([{'role':'user','content':'One action'}])
    assert len(calls)==2


def test_controller_recovers_known_failure_then_commits_only_complete_action(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','offline')
    monkeypatch.setenv('OPENAI_BASE_URL','http://offline')
    c=Controller(write_task(tmp_path), Limits(seconds=90,cpu=1))
    c.prepare();c.qwen.client.close();calls=[]
    def reply(request):
        calls.append(request)
        if len(calls)==1:
            poison=finish()['choices'][0]['message']['tool_calls'][0]
            poison['id']='discarded';poison['index']=0
            poison['function']['arguments']=poison['function']['arguments'].replace('done','discarded')
            raw=frames([chunk({'tool_calls':[poison]}),
                {'error':{'code':'502','message':'provider unavailable'}}], done=False)
            return httpx.Response(200,headers={'content-type':'text/event-stream'},content=raw)
        return httpx.Response(200,json=finish())
    c.qwen.client=httpx.Client(transport=httpx.MockTransport(reply))
    monkeypatch.setattr(c,'recover_search',lambda error:False)
    try:
        result=c.run()
        assert len(calls)==2 and c.journal.get('llm_calls')==2
        assert c.journal.operation('discarded') is None
        assert c.journal.operation('done') is not None
        assert not c.journal.get('qwen_unknown_outcome')
        assert c.journal.get('qwen_provider_failure_streak')==0
        assert result['control_status']==result['refit_status']=='completed'
    finally:c.close()
