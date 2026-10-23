import json
import time

import httpx
import pytest

from automl_agent.budget import Budget, Limits
from automl_agent.journal import Journal
from automl_agent.qwen import QwenClient, QwenOutcomeUnknown, QwenRetryableError
from automl_agent.qwen_stream import IncompleteModelStream, completion_from_sse


def frames(chunks, done=True):
    return ''.join('data: '+json.dumps(c, ensure_ascii=False)+'\n\n' for c in chunks) + ('data: [DONE]\n\n' if done else '')


def chunk(delta, finish=None):
    return {'choices':[{'index':0, 'delta':delta, 'finish_reason':finish}]}


def test_split_native_arguments_and_usage_are_assembled_before_action_parsing():
    parts = [chunk({'reasoning_content':'Never treat this as a command'}),
        chunk({'tool_calls':[{'index':0,'id':'call_1','type':'function',
            'function':{'name':'train_','arguments':'{"model":'}}]}),
        chunk({'tool_calls':[{'index':0,'function':{'name':'model','arguments':'"дерево"}'}}]}, 'tool_calls'),
        {'choices':[], 'usage':{'prompt_tokens':20, 'completion_tokens':15,
            'completion_tokens_details':{'reasoning_tokens':5}}}]
    result = completion_from_sse(frames(parts).splitlines())
    call = result['choices'][0]['message']['tool_calls'][0]
    assert call['function']['name']=='train_model' and json.loads(call['function']['arguments'])=={'model':'дерево'}
    assert result['choices'][0]['message']['content']==''
    assert result['usage']['completion_tokens_details']['reasoning_tokens']==5


@pytest.mark.parametrize('body', [frames([chunk({'content':'{}'})]),
    frames([chunk({'content':'{}'}, 'stop')],done=False),
    'data: {bad json}\n\ndata: [DONE]\n\n',
    frames([{'error':{'message':'upstream failed'}}]),
    frames([chunk({'content':'{}'},'stop'),{'choices':[], 'usage':[]}]),
    frames([chunk({'content':'{}'},'stop'),{'choices':[], 'usage':{'completion_tokens_details':[1]}}]),
    frames([chunk({'content':'{}'},'length'),chunk({},'stop')]),
    frames([chunk({'content':'{}'},[])]),
    frames([chunk({'tool_calls':[{'index':-1}]},'tool_calls')])])
def test_partial_malformed_and_error_streams_never_become_actions(body):
    with pytest.raises(IncompleteModelStream):
        completion_from_sse(body.splitlines())


def test_sse_comments_multiline_data_and_usage_only_trailer():
    data = ': heartbeat\n\ndata: {"choices": [\ndata: {"index":0,"delta":{"content":"{}"},"finish_reason":"stop"}]}\n\ndata: {"choices":[],"usage":{"total_tokens":12}}\n\ndata: [DONE]\n\n'
    result = completion_from_sse(data.splitlines())
    assert result['choices'][0]['message']['content']=='{}' and result['usage']['total_tokens']==12


def test_observed_provider_empty_role_trailer_after_finish_is_not_new_output():
    result=completion_from_sse(frames([chunk({'content':'{}'},'stop'),
        chunk({'role':'assistant','content':''}),chunk({}),
        {'choices':[], 'usage':{'total_tokens':12}}]).splitlines())
    assert result['choices'][0]['finish_reason']=='stop' and result['usage']['total_tokens']==12


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','offline-fixture')
    monkeypatch.setenv('OPENAI_BASE_URL','http://fixture.invalid/v1')
    monkeypatch.delenv('OPENAI_EXTRA_BODY',raising=False)
    journal=Journal(tmp_path)
    qwen=QwenClient(journal,Budget(journal,Limits(seconds=60),time.time()))
    qwen.client.close()
    yield qwen
    qwen.close();journal.close()


def test_interrupted_stream_latches_unknown_and_does_not_retry_or_fallback(client):
    requests=[]
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200,headers={'content-type':'text/event-stream'},
            content=frames([chunk({'content':'{"tool":'})],done=False))
    client.client=httpx.Client(transport=httpx.MockTransport(handler))
    for _ in range(2):
        with pytest.raises(QwenOutcomeUnknown):client._request([{'role':'user','content':'JSON command'}])
    assert len(requests)==1 and requests[0]['stream'] and requests[0]['stream_options']['include_usage']
    assert client.journal.get('qwen_unknown_outcome')['reason']=='stream_incomplete'


@pytest.mark.parametrize('raw', ['{"choices":','[]','{}','{"choices":[]}',
    json.dumps({'choices':[{'message':{},'finish_reason':[]}]}),
    json.dumps({'choices':[{'message':{},'finish_reason':'stop'}],'usage':{'completion_tokens_details':[1]}})])
def test_malformed_json_fallback_is_unknown_without_a_second_request(client, raw):
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(200,headers={'content-type':'application/json'},content=raw)
    client.client=httpx.Client(transport=httpx.MockTransport(handler))
    for _ in range(2):
        with pytest.raises(QwenOutcomeUnknown):client.next_command('Use a tool',{})
    assert len(requests)==1 and client.journal.get('qwen_unknown_outcome')['reason']=='invalid_response_body'


def test_unicode_json_chunks_and_usage_reach_the_existing_contract_parser(client):
    class Fragments(httpx.SyncByteStream):
        def __iter__(self):
            raw=frames([chunk({'reasoning_content':'ignore'}),chunk({'content':'{"тест":'}),
                chunk({'content':'true}'},'stop'),{'choices':[],'usage':{'prompt_tokens':2,'completion_tokens':3}}]).encode()
            for i in range(0,len(raw),7):yield raw[i:i+7]
    client.client=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,
        headers={'content-type':'text/event-stream'},stream=Fragments())))
    assert client.resolve('Return JSON',{})=={'тест':True}
    assert client.journal.get('token_usage')['completion_tokens']==3


def test_terminal_length_grows_output_without_unknown_retry_latch(client):
    client.client=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,
        headers={'content-type':'text/event-stream'},content=frames([chunk({'content':''},'length')]))))
    initial=client.output_tokens
    with pytest.raises(QwenRetryableError):client._request([{'role':'user','content':'JSON command'}])
    assert client.output_tokens>initial and client.journal.get('qwen_unknown_outcome') is None
