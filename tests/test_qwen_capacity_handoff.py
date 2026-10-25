import json

import httpx
import pytest

from automl_agent.budget import Limits
from automl_agent.controller import Controller
from automl_agent.qwen import QwenUnavailable
from conftest import write_task


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('AGENT_OUTPUT_TOKENS', raising=False)
    c = Controller(write_task(tmp_path), Limits(seconds=300, cpu=1), run_id='capacity')
    c.no_llm = True
    c.prepare()
    c.qwen.disabled = False
    c.qwen.key = 'fixture-not-a-real-key'
    c.qwen.base_url = 'http://fixture.invalid/v1'
    c.journal.put('last_tool_result', {'status': 'success'})
    c.journal.put('last_research_tool', 'read_run_records')
    c.journal.put('stagnation_count', 0)
    yield c
    c.close()


@pytest.mark.parametrize('mode', ['native', 'json'])
def test_real_loop_hands_known_ordinary_truncation_to_existing_complex_budget(prepared, mode):
    c = prepared
    c.qwen.transport = mode
    c.qwen.output_by_effort['medium'] = 8192
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            assert body['reasoning']['effort'] == 'medium' and body['max_tokens'] == 8192
            arguments = {'reason': 'This complete-looking action must still be discarded on length.',
                'result_ids': [c.journal.get('baseline_id')]}
            if mode == 'native':
                truncated = {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'discarded_truncated_finish', 'type': 'function', 'function': {'name': 'finish',
                        'arguments': json.dumps({'operation_id': 'discarded_truncated_finish', **arguments})}}]}
            else:
                truncated = {'role': 'assistant', 'content': json.dumps({
                    'operation_id': 'discarded_truncated_finish', 'tool': 'finish', 'arguments': arguments})}
            return httpx.Response(200, json={'choices': [{'finish_reason': 'length',
                'message': truncated}], 'usage': {
                    'prompt_tokens': 100, 'completion_tokens': 8192, 'total_tokens': 8292,
                    'completion_tokens_details': {'reasoning_tokens': 8192}}})
        assert len(requests) == 2
        assert body['reasoning']['effort'] == 'xhigh' and body['max_tokens'] == 12288
        arguments = {'reason': 'Offline known-capacity handoff verified; remaining effects are unmeasured.',
            'result_ids': [c.journal.get('baseline_id')],
            'skipped_directions': {'all': 'This HTTP transport fixture tests safe completion, not ML search quality.'}}
        operation = 'finish_capacity'
        if mode == 'native':
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': operation,
                'type': 'function', 'function': {'name': 'finish',
                    'arguments': json.dumps({'operation_id': operation, **arguments})}}]}
        else:
            message = {'role': 'assistant', 'content': json.dumps({'operation_id': operation,
                'tool': 'finish', 'arguments': arguments})}
        return httpx.Response(200, json={'choices': [{'finish_reason': 'tool_calls' if mode == 'native' else 'stop',
            'message': message}], 'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}})
    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(handler))
    report = c.run()
    assert report['state'] == 'completed' and report['control_status'] == 'completed'
    assert report['refit_status'] == 'completed'
    assert len(requests) == 2
    assert report['token_usage']['completion_tokens'] == 8212
    assert c.journal.get('qwen_unknown_outcome') is None
    assert c.journal.operation('finish_capacity').status == 'succeeded'
    assert c.journal.operation('discarded_truncated_finish') is None
    assert c.journal.db.execute("SELECT COUNT(*) FROM events WHERE kind='qwen_ordinary_capacity_reached'").fetchone()[0] == 1
    assert c.publisher.output.is_file()


@pytest.mark.parametrize('version,effort,ceiling', [(4, 'medium', 8192), (5, 'xhigh', 24576)])
def test_legacy_and_largest_capacity_still_stop_without_accepting_partial_action(prepared, version, effort, ceiling):
    c = prepared
    c.prompt_contract = {**c.prompt_contract, 'version': version}
    if effort == 'xhigh':
        c.journal.put('last_tool_result', {'status': 'model_transport', 'error': 'fixture'})
    c.qwen.output_by_effort[effort] = ceiling
    c.qwen.transport = 'native'
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'finish_reason': 'length',
            'message': {'role': 'assistant', 'content': ''}}]})
    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(QwenUnavailable, match='Bounded output repair exhausted'):
        c.qwen.next_command(c.system_prompt, c.context_memory.snapshot(c.system_prompt))
    assert len(calls) == 1
    assert c.context_memory.pending() is None
    assert c.journal.get('finish_evidence') is None
