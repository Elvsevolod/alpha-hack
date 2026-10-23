import copy
import json

import httpx
import pytest

from automl_agent.controller import SYSTEM
from automl_agent.contracts import ToolCommand
from context_compressor import ContextCompressor, ContextError, ContextStore
from context_compressor.tests.helpers import allowed, emit, note, response, setup, transport
from test_tool_api_v2 import agent


def multiple(first=None, second=None):
    first=first or response('native',operation_id='first',tool='get_reference',arguments={'query':'Read evidence'},note=note())
    second=second or response('native',operation_id='ignored',tool='finish',arguments={'reason':'Not executed'},
        note={'question':'INVALID_IGNORED_NOTE'})
    result=copy.deepcopy(first)
    result['choices'][0]['message']['tool_calls'] += copy.deepcopy(second['choices'][0]['message']['tool_calls'])
    return result


@pytest.fixture
def compressor(tmp_path):
    store=ContextStore(tmp_path/'journal.sqlite');setup(store)
    value=ContextCompressor(store,transport())
    yield value
    store.close()


def test_first_only_has_one_receipt_pending_command_note_and_delivery_ack(compressor):
    store=compressor.store;observation=emit(store)
    prepared=compressor.prepare(compressor.snapshot(),'native')
    raw=multiple();original=copy.deepcopy(raw)
    selected=compressor.accept_response(prepared.request_id,raw,allowed)
    again=compressor.accept_response(prepared.request_id,raw,allowed)
    assert selected==again and selected.operation_id=='first' and raw==original
    assert [x['operation_id'] for x in store.pending_commands()]==['first']
    assert store.db.execute('SELECT COUNT(*) FROM cc_receipts').fetchone()[0]==1
    assert store.db.execute('SELECT COUNT(*) FROM cc_notes').fetchone()[0]==1
    assert store.active_note()['question']==note()['question']
    assert store.observation_status(observation)['state']=='delivered'
    changed=multiple(response('native',operation_id='first',tool='get_reference',arguments={'query':'Changed'}))
    with pytest.raises(ContextError):compressor.accept_response(prepared.request_id,changed,allowed)
    assert len(store.pending_commands())==1


@pytest.mark.parametrize('failure',['unknown_tool','bad_json','bad_operation_id','missing_reason'])
def test_invalid_first_never_selects_a_valid_second_or_acknowledges_delivery(compressor,failure):
    store=compressor.store;observation=emit(store)
    prepared=compressor.prepare(compressor.snapshot(),'native')
    bad=response('native',operation_id='bad',tool='finish',arguments={})
    call=bad['choices'][0]['message']['tool_calls'][0]
    if failure=='unknown_tool':call['function']['name']='not_permitted'
    elif failure=='bad_json':call['function']['arguments']='{bad json'
    elif failure=='bad_operation_id':call['function']['arguments']=json.dumps({'operation_id':'bad.id','reason':'Done'})
    good=response('native',operation_id='good_second',tool='finish',arguments={'reason':'Done'},note=note())
    with pytest.raises((ContextError,ValueError)):
        compressor.accept_response(prepared.request_id,multiple(bad,good),allowed)
    assert store.pending_commands()==[]
    assert store.db.execute('SELECT COUNT(*) FROM cc_receipts').fetchone()[0]==0
    assert store.db.execute('SELECT COUNT(*) FROM cc_notes').fetchone()[0]==0
    assert store.observation_status(observation)['state']!='delivered'


@pytest.mark.parametrize('reason',['length','content_filter',None])
def test_incomplete_or_filtered_response_cannot_ack_even_valid_first(compressor,reason):
    prepared=compressor.prepare(compressor.snapshot(),'native');raw=multiple()
    raw['choices'][0]['finish_reason']=reason
    with pytest.raises(ContextError):compressor.accept_response(prepared.request_id,raw,allowed)
    assert compressor.store.pending_commands()==[]


def test_json_transport_still_rejects_native_calls():
    with pytest.raises(ContextError):transport().parse_response(multiple(),'json')


def test_real_controller_keeps_native_with_one_request_and_executes_only_first(agent):
    c=agent;snapshot=c.context_memory.snapshot(SYSTEM)
    core=json.loads(snapshot.core_json);op=core['current_work']['suggested_new_operation_id']
    raw=response('native',operation_id=op,tool='get_dataset_report',arguments={
        'scope':'fit','include_target':False,'knowledge_gap':'Inspect training data'},note=note())
    tail=response('native',operation_id='never_execute_finish',tool='finish',arguments={
        'reason':'Must be discarded','result_ids':['baseline']},note={'question':'INVALID_IGNORED_NOTE'})
    wire=multiple(raw,tail);requests=[]
    def reply(request):
        requests.append(json.loads(request.content));return httpx.Response(200,json=wire)
    c.qwen.client.close();c.qwen.client=httpx.Client(transport=httpx.MockTransport(reply))
    selected,_=c.qwen.next_command(SYSTEM,snapshot)
    assert len(requests)==1 and requests[0]['tools'] and c.qwen.transport=='native'
    assert selected.operation_id==op and c.context_memory.pending().operation_id==op
    c.dispatch(ToolCommand.model_validate(selected.model_dump(mode='json')))
    assert c.context_memory.pending() is None and c.state=='searching'
    assert c.journal.operation('never_execute_finish') is None
    assert c.context_memory.store.db.execute('SELECT COUNT(*) FROM cc_commands').fetchone()[0]==1
    rows=c.journal.db.execute("SELECT payload FROM events WHERE kind='extra_native_calls_deferred'").fetchall()
    assert len(rows)==1 and json.loads(rows[0][0])=={'received':2,'accepted':1,'discarded':1,'queued':0,
        'reason':'One operation per updated research state; only the first call was validated'}
