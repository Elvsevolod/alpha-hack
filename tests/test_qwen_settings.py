import json
import time

import httpx
import pytest

from automl_agent.budget import Budget, Limits
from automl_agent.contracts import EnsembleArgs, Hypothesis, PipelineSpec, ToolCommand, TrainArgs
from automl_agent.controller import Controller, SYSTEM
from automl_agent.journal import Journal
from automl_agent.qwen import GENERATION, QwenClient, QwenRetryableError
from automl_agent.tool_results import payload as tool_payload
from conftest import write_task
from test_context_integration import response


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://fixture.invalid/v1')
    monkeypatch.setenv('OPENAI_EXTRA_BODY', json.dumps({'temperature': .2, 'reasoning': {'enabled': False},
        'reasoning_effort': 'low', 'chat_template_kwargs': {'enable_thinking': False}}))
    monkeypatch.delenv('AGENT_OUTPUT_TOKENS', raising=False)
    journal = Journal(tmp_path)
    qwen = QwenClient(journal, Budget(journal, Limits(), time.time()))
    yield qwen
    qwen.close()
    journal.close()


def test_wire_settings_override_stale_env_and_request_has_no_300_second_cap(client):
    captured = []
    def handler(request):
        captured.append((json.loads(request.content), request.extensions['timeout']))
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}],
            'usage': {'completion_tokens_details': {'reasoning_tokens': 12}}})
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    for effort in ('medium', 'xhigh', 'medium'):
        client.select_reasoning(effort, 'test')
        client._request([{'role': 'user', 'content': 'fixture'}])
    for (body, timeout), effort, output in zip(captured, ('medium', 'xhigh', 'medium'), (16384, 32768, 16384)):
        assert all(body[k] == v for k, v in GENERATION.items())
        assert body['reasoning'] == {'enabled': True, 'effort': effort}
        assert 'reasoning_effort' not in body and 'chat_template_kwargs' not in body
        assert body['max_tokens'] == output
        assert 300 < timeout['read'] <= client.budget.limits.seconds - 300
        assert timeout['connect'] <= 15
    assert client.journal.get('token_usage')['reasoning_tokens'] == 36


def test_task_resolution_requests_json_and_never_uses_reasoning_as_the_final_contract(client):
    captured = []
    def handler(request):
        body = json.loads(request.content); captured.append(body)
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': None, 'reasoning_content': '{"status":"proposed"}'}}]})
    client.client.close(); client.client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match='single JSON object'):
        client.resolve('Return one JSON task contract.', {'sources': {}})
    assert captured[0]['response_format'] == {'type': 'json_object'}
    assert captured[0]['reasoning'] == {'enabled': True, 'effort': 'xhigh'}


@pytest.mark.parametrize('mode', ['native', 'json'])
def test_compressed_response_format_belongs_to_the_rendered_wire(mode):
    from context_compressor.transport import Transport
    from automl_agent.contracts import TOOL_SCHEMAS
    transport = Transport(system='Return a JSON command.', tool_schemas={name: schema.model_json_schema()
        for name, schema in TOOL_SCHEMAS.items()}, response_reserve=1024)
    payload = json.loads(transport.render({}, mode))
    if mode == 'json':
        assert payload['response_format'] == {'type': 'json_object'}
        assert 'tools' not in payload
    else:
        assert 'response_format' not in payload
        assert payload['tools'] and payload['tool_choice'] == 'required'


def test_output_growth_is_per_effort_and_survives_resume_beyond_32768(client):
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200,
        json={'choices': [{'finish_reason': 'length', 'message': {'content': ''}}]})))
    client.select_reasoning('xhigh', 'test')
    with pytest.raises(QwenRetryableError):
        client._request([{'role': 'user', 'content': 'fixture'}])
    assert client.output_tokens == 65536
    client.select_reasoning('medium', 'test')
    assert client.output_tokens == 16384
    resumed = QwenClient(client.journal, client.budget)
    try:
        resumed.select_reasoning('xhigh', 'test')
        assert resumed.output_tokens == 65536
    finally:
        resumed.close()


@pytest.mark.parametrize('transport', ['native', 'json'])
def test_compressed_wire_switches_medium_xhigh_and_back(tmp_path, monkeypatch, transport):
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://fixture.invalid/v1')
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1), transport=transport)
    try:
        c.prepare()
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, json=response(body, operation='step_' + str(len(requests))))
        c.qwen.client.close()
        c.qwen.client = httpx.Client(transport=httpx.MockTransport(handler))
        for tool in ('get_dataset_report', 'train_model', 'read_knowledge', 'run_custom_experiment'):
            c.journal.put('last_research_tool', tool)
            snapshot = c.context_memory.snapshot(SYSTEM)
            command, _ = c.qwen.next_command(SYSTEM, snapshot)
            c.dispatch(command)
        assert [r['reasoning']['effort'] for r in requests] == ['medium', 'xhigh', 'medium', 'xhigh']
        assert all(r['temperature'] == 1 and r['reasoning']['enabled'] for r in requests)
        assert [r['max_tokens'] for r in requests] == [16384, 32768, 16384, 32768]
        c.journal.put('last_tool_result', {'status': 'rejected', 'error': 'A failed candidate needs repair'})
        c.journal.put('last_research_tool', 'read_knowledge')
        c.context_memory.snapshot(SYSTEM)
        assert c.qwen.effort == 'xhigh'
        c.journal.put('last_tool_result', {})
        from context_compressor.contracts import ResearchNote
        c.context_memory.store._put_meta('active_note', ResearchNote(question='Implement a custom hypothesis',
            reading_goal='API checked', expected_evidence='Compare metrics', status='ready_to_compare').persisted('complex_note'))
        c.context_memory.snapshot(SYSTEM)
        assert c.qwen.effort == 'xhigh'
    finally:
        c.close()


def test_platform_budget_is_three_hours_without_global_counts_or_training_cap(tmp_path):
    journal = Journal(tmp_path)
    try:
        limits = Limits()
        budget = Budget(journal, limits, time.time())
        assert limits.seconds == 10800 and limits.max_debug is None
        journal.put('llm_calls', 10000)
        journal.put('candidate_count', 10000)
        journal.put('llm_seconds', 5000)
        assert budget.claim_call() == 10001 and budget.claim_candidate() == 10001
        assert budget.snapshot()['qwen_calls'] is None and budget.snapshot()['candidates'] is None
        assert budget.grant_training(9000) == 9000
        hypothesis = Hypothesis(observation='Compare', expected_effect='Improve', changed_component='ensemble', acceptance='Metric')
        assert EnsembleArgs(members=['a', 'b'], hypothesis=hypothesis, budget_seconds=3600).budget_seconds == 3600
        budget.deadline = time.time() + 299
        with pytest.raises(TimeoutError):
            budget.claim_call()
    finally:
        journal.close()


def test_stalled_search_recovers_then_returns_to_qwen_instead_of_finishing(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://fixture.invalid/v1')
    c = Controller(write_task(tmp_path), Limits(seconds=3600, cpu=1))
    c.prepare()
    calls = []
    monkeypatch.setattr('automl_agent.controller.progress.exhausted', lambda c: True)
    monkeypatch.setattr(c, 'recover_search', lambda error: False)
    def next_command(*args):
        calls.append(1)
        c.budget.deadline = time.time() + c.budget.final_reserve(c.finalist_costs()) + 30
        return ToolCommand(operation_id='finish_at_reserve', tool='finish', arguments={
            'reason': 'Finalization reserve approaching', 'result_ids':['baseline'], 'imputation_skip_reason': 'fixture'}), ''
    monkeypatch.setattr(c.qwen, 'next_command', next_command)
    try:
        early = tool_payload(c.dispatch(ToolCommand(operation_id='early', tool='finish', arguments={'reason': 'Stalled'})))
        assert early['status'] == 'rejected' and c.state == 'searching'
        report = c.run()
        assert len(calls) == 1 and report['state'] == 'completed'
        assert report['research_stop_reason'] == 'Finalization reserve approaching'
    finally:
        c.close()


def test_more_than_two_debug_repairs_are_admissible(tmp_path, monkeypatch):
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1))
    try:
        c.prepare()
        parent, result = c.lookup('baseline')
        parent = parent.model_copy(update={'experiment_id': 'failed_parent', 'debug_root': 'root'})
        result = result.model_copy(update={'status': 'error'})
        monkeypatch.setattr(c, 'lookup', lambda _: (parent, result))
        monkeypatch.setattr(c, 'records', lambda: [(parent.model_copy(update={'search_action': 'debug'}), result)] * 20)
        args = TrainArgs(pipeline=PipelineSpec(model='ridge'), parent_id='failed_parent', search_action='debug',
            hypothesis=Hypothesis(observation='Failure', expected_effect='Repair', changed_component='fix', acceptance='Run'))
        assert c.make_spec(args, 'repair_21').debug_root == 'root'
    finally:
        c.close()


def test_task_resolution_can_recover_after_more_than_three_invalid_responses(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://fixture.invalid/v1')
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1))
    try:
        task = c.dataset.explicit()
        monkeypatch.setattr(c.dataset, 'draft', lambda: (None, ['time_column']))
        monkeypatch.setattr(c.dataset, 'confirm_proposal', lambda proposal, request_id: task)
        monkeypatch.setattr('automl_agent.controller.time.sleep', lambda _: None)
        calls = []
        def resolve(prompt, request):
            calls.append(1)
            if len(calls) <= 4:
                raise ValueError('Invalid synthetic response')
            return {'request_id': request['request_id'], 'source_hash': request['source_hash'],
                'status': 'proposed', 'task_spec': task.model_dump(mode='json')}
        monkeypatch.setattr(c.qwen, 'resolve', resolve)
        confirmed = c._task()
        assert len(calls) == 5 and confirmed.targets == task.targets
    finally:
        c.close()
