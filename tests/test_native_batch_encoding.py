import json

import httpx
import pytest

from automl_agent.contracts import BatchArgs, RunReadArgs, TOOL_SCHEMAS, ToolCommand, public_tool_schema
from automl_agent.tool_results import payload
from context_compressor import ContextError, Transport
from test_tool_api_v2 import agent


def renderer():
    return Transport(system='Offline encoding fixture', tool_schemas={
        key: public_tool_schema(TOOL_SCHEMAS[key]) for key in ['run_experiments', 'read_run_records']})


def candidates():
    return [dict(pipeline=dict(model='ridge', hyperparameters=dict(alpha=alpha)), budget_seconds=20,
        hypothesis=dict(observation='Two independently specified regularization values',
            expected_effect='Measure the declared parameter difference', changed_component='hyperparameters',
            acceptance='Complete the same fixed development evaluation')) for alpha in [17, 29]]


def response(arguments, *, mode='native', tool='run_experiments'):
    if mode == 'native':
        message = dict(role='assistant', tool_calls=[dict(id='batch_call', type='function',
            function=dict(name=tool, arguments=json.dumps(dict(operation_id='batch_encoding', **arguments))))])
    else:
        message = dict(role='assistant', content=json.dumps(dict(operation_id='batch_encoding',
            tool=tool, arguments=arguments)))
    return dict(choices=[dict(message=message, finish_reason='tool_calls' if mode=='native' else 'stop')])


def test_native_encoded_array_preserves_every_candidate_and_matches_json():
    items = candidates()
    native = renderer().parse_response(response(dict(candidates=json.dumps(items))), 'native')
    plain = renderer().parse_response(response(dict(candidates=items), mode='json'), 'json')
    assert native == plain
    assert BatchArgs.model_validate(native.as_dict()['arguments']).candidates == items


@pytest.mark.parametrize('text', ['[] trailing', '{}', 'null',
    '[{"pipeline":{},"pipeline":{}}]', '[{"budget_seconds":NaN}]',
    json.dumps(json.dumps(candidates()))])
def test_native_array_requires_one_strict_complete_json_array(text):
    with pytest.raises(ContextError):
        renderer().parse_response(response(dict(candidates=text)), 'native')


@pytest.mark.parametrize('count', [1, 5])
def test_normalized_array_retains_batch_size_bounds(count):
    command = renderer().parse_response(response(dict(candidates=json.dumps(candidates()[:1]*count))), 'native')
    with pytest.raises(ValueError):
        BatchArgs.model_validate(command.as_dict()['arguments'])


def test_json_route_and_read_identifiers_are_not_repaired():
    command = renderer().parse_response(response(dict(candidates=json.dumps(candidates())), mode='json'), 'json')
    with pytest.raises(ValueError):
        BatchArgs.model_validate(command.as_dict()['arguments'])
    for args in [dict(view='pipeline'), dict(ids='["baseline"]', view='pipeline')]:
        command = renderer().parse_response(response(args, tool='read_run_records'), 'native')
        with pytest.raises(ValueError):
            RunReadArgs.model_validate(command.as_dict()['arguments'])


def test_http_encoded_batch_reaches_existing_sequential_dispatch_once(agent, monkeypatch):
    c = agent
    c.qwen.transport = 'native'
    requests, executed = [], []
    items = candidates()
    before = c.publisher.output.read_bytes()

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response(dict(candidates=json.dumps(items), budget_seconds=45)))

    def execute(spec, **kwargs):
        executed.append(spec)
        return c.failed_result(spec, dict(status='error', error='Offline execution sentinel'))

    c.qwen.client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(c, 'execute_experiment', execute)
    command, _ = c.qwen.next_command(c.system_prompt, c.context_memory.snapshot(c.system_prompt))
    result = payload(c.dispatch(command))
    assert len(requests) == 1 and c.budget.counter('llm_calls') == 1
    assert result['status'] == 'success' and len(executed) == 2
    assert [spec.pipeline.hyperparameters['alpha'] for spec in executed] == [17, 29]
    assert c.publisher.output.read_bytes() == before and not c.journal.get('control_opened')


@pytest.mark.parametrize('invalid', ['engine', 'evidence', 'duplicate'])
def test_decoded_batch_still_validates_all_members_before_first_fit(agent, monkeypatch, invalid):
    c = agent
    items = candidates()
    if invalid == 'engine':
        items[1]['pipeline']['engine'] = 'custom'
    elif invalid == 'evidence':
        items[1]['hypothesis']['evidence_refs'] = [dict(kind='record', id='invented_record', version='1')]
    else:
        items[1] = items[0]
    before = c.publisher.output.read_bytes()
    executed = []
    monkeypatch.setattr(c, 'execute_experiment', lambda *args, **kwargs: executed.append(args))
    command = renderer().parse_response(response(dict(candidates=json.dumps(items))), 'native')
    c.context_memory.validate(command)
    result = payload(c.dispatch(ToolCommand.model_validate({
        k:v for k,v in command.as_dict().items() if k!='research_note'})))
    assert result['status'] == 'rejected' and not executed
    assert c.budget.counter('candidate_count') == 0
    assert c.publisher.output.read_bytes() == before and not c.journal.get('control_opened')
