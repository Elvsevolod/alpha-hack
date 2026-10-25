"""Real controller delivery, pinned prompts and bounded context in both transports."""
import json

import pytest

from automl_agent.controller import Controller, SYSTEM, LEGACY_SYSTEM
from automl_agent.prompting import bind_prompt, build_system_prompt
from automl_agent.references import INTERFACE
from automl_agent.budget import Limits
from automl_agent.contracts import TOOL_SCHEMAS
from automl_agent.journal import Journal
from automl_agent.decision_context import decision_frame
from conftest import write_task


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1), run_id='layered')
    c.no_llm = True
    c.qwen.disabled = True
    c.prepare()
    yield c
    c.close()


def read_all(c, route):
    text = ''
    while route:
        args = TOOL_SCHEMAS['read_run_records'].model_validate(route['arguments'])
        result = c.run_tools.read(args)
        assert result['status'] == 'success'
        text += result['content']
        route = result['continuation']
    return text


@pytest.mark.parametrize('mode', ['native', 'json'])
def test_guides_and_last_operation_survive_every_compaction_stage(controller, mode):
    c = controller
    snap = c.context_memory.snapshot("Caller override must not replace a run-pinned prompt")
    cold = json.loads(json.loads(c.context_memory.compressor.fit(snap, mode, budget=262144).wire_json)['messages'][1]['content'])
    assert cold['reference_evidence']['status'] == 'deferred'
    assert cold['reference_evidence']['materials'] == []
    for stage in range(4):
        prepared = c.context_memory.compressor.fit(snap, mode, budget=262144, minimum_stage=stage)
        wire = json.loads(prepared.wire_json)
        context = json.loads(wire['messages'][1]['content'])
        work = context['core']['current_work']
        assert work['decision_frame']['version'] == 1
        catalog = json.loads(read_all(c, work['local_guides']))
        interface = next(entry for entry in catalog['guides'] if entry['id'] == 'candidate_interface')
        assert read_all(c, interface['read']) == INTERFACE
        policy = next(entry for entry in catalog['guides'] if entry['id'] == 'research_policy')
        assert read_all(c, policy['read']) == c.prompt_contract['guides']['research_policy']
        assert '<authority_and_scope>' in wire['messages'][0]['content']
        assert 'provided Qwen API' in wire['messages'][0]['content']
        assert (len(wire['tools']) if mode == 'native' else len(context['tool_argument_schemas'])) == 17


def test_existing_run_keeps_legacy_prompt_and_new_run_is_pinned(tmp_path):
    j = Journal(tmp_path)
    try:
        old = bind_prompt(j, LEGACY_SYSTEM, existing_run=True)
        assert old['version'] == 1 and old['system'] == LEGACY_SYSTEM
        assert bind_prompt(j, 'changed', existing_run=False) == old
    finally:
        j.close()
    j = Journal(tmp_path / 'new')
    try:
        new = bind_prompt(j, LEGACY_SYSTEM, existing_run=False)
        assert new['version'] == 2 and new['system'] == SYSTEM == build_system_prompt()
        assert new['guides']['candidate_interface'] == INTERFACE
        assert bind_prompt(j, 'different runtime', existing_run=True) == new
        j.put('llm_prompt_contract', {**new, 'system': 'tampered'})
        with pytest.raises(ValueError, match='changed'):
            bind_prompt(j, LEGACY_SYSTEM, existing_run=True)
    finally:
        j.close()


def test_frame_never_copies_predictions_or_treats_rejected_generation_as_fit(controller):
    c = controller
    c.journal.put('last_tool_result', {'status': 'command_validation', 'error': 'PRIVATE_SENTINEL'})
    frame = decision_frame(c)
    assert frame['last_request_status'] == 'command_validation'
    assert 'PRIVATE_SENTINEL' not in json.dumps(frame)
    assert not any(key in frame for key in ['values', 'labels', 'predictions', 'score'])


def test_controller_resume_uses_pinned_legacy_when_no_prompt_metadata(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    path = write_task(tmp_path)
    first = Controller(path, Limits(seconds=180, cpu=1), run_id='old')
    try:
        first.no_llm = True
        first.qwen.disabled = True
        first.prepare()
        first.journal.db.execute("DELETE FROM metadata WHERE key='llm_prompt_contract'")
        first.journal.db.commit()
    finally:
        first.close()
    second = Controller(path, Limits(seconds=180, cpu=1), run_id='old')
    try:
        assert second.prompt_contract['version'] == 1
        assert second.system_prompt == LEGACY_SYSTEM
        second.prepare()
        snap = second.context_memory.snapshot(SYSTEM)
        prepared = second.context_memory.compressor.fit(snap, 'native', budget=262144)
        wire = json.loads(prepared.wire_json)
        assert wire['messages'][0]['content'].startswith(LEGACY_SYSTEM)
        work = json.loads(snap.core_json)['current_work']
        assert 'local_guides' not in work and 'decision_frame' not in work
    finally:
        second.close()


@pytest.mark.parametrize('mode', ['native', 'json'])
def test_http_mock_reads_guide_then_sees_its_operation_and_continuation(controller, mode):
    import httpx
    from automl_agent.tool_results import payload
    c = controller
    c.qwen.disabled = False
    c.qwen.key = 'fixture-not-a-real-key'
    c.qwen.base_url = 'http://fixture.invalid/v1'
    c.qwen.transport = mode
    c.qwen.client.close()
    snap = c.context_memory.snapshot(c.system_prompt)
    route = json.loads(snap.core_json)['current_work']['local_guides']
    calls = []
    next_route = [route]

    def handler(request):
        wire = json.loads(request.content)
        context = json.loads(wire['messages'][1]['content'])
        assert context['reference_evidence']['status'] == 'deferred'
        if calls:
            assert context['core']['current_work']['decision_frame']['last_operation']['operation_id'] == 'guide_1'
            assert context['observations']
        calls.append(wire)
        arguments = next_route[0]['arguments']
        op = 'guide_' + str(len(calls))
        if mode == 'native':
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': op,
                'type': 'function', 'function': {'name': 'read_run_records',
                    'arguments': json.dumps({'operation_id': op, **arguments})}}]}
        else:
            message = {'role': 'assistant', 'content': json.dumps({'operation_id': op,
                'tool': 'read_run_records', 'arguments': arguments})}
        return httpx.Response(200, json={'choices': [{'message': message,
            'finish_reason': 'tool_calls' if mode == 'native' else 'stop'}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}})

    c.qwen.client = httpx.Client(transport=httpx.MockTransport(handler))
    for i in range(2):
        command, _ = c.qwen.next_command(c.system_prompt, c.context_memory.snapshot(c.system_prompt))
        result = payload(c.dispatch(command))
        assert result['status'] == 'success'
        next_route[0] = result['continuation']
        if i == 0:
            assert next_route[0]  # first catalog fragment is incomplete
    assert len(calls) == 2
    metadata = c.report()['prompt_contract']
    assert metadata['version'] == 2 and 'guides' not in metadata and 'system' not in metadata
    assert c.journal.get('control_status', 'not_opened') == 'not_opened'
