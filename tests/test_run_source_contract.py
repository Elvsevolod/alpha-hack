"""The public source-reader schema must describe the executor's byte contract."""
import json

import pytest
from pydantic import ValidationError

from automl_agent.contracts import RunReadArgs, RunSourceRef, public_tool_schema
from context_compressor.contracts import Source


@pytest.mark.parametrize('field,value', [('source_id',''),('version',''),
    ('source_id','x'*97),('version','x'*129),('section','x'*129),
    ('source_id','я'*49),('version','я'*65),('section','я'*65)])
def test_source_contract_rejects_values_the_backend_cannot_read(field, value):
    data = dict(source_id='known_source', version='known_version', scope='controller', section='')
    data[field] = value
    with pytest.raises(ValidationError):
        RunSourceRef.model_validate(data)


def test_actual_live_empty_version_is_rejected_by_public_schema():
    observed = {'source':{'limit':768,'offset':0,'scope':'reference','section':'',
        'source_id':'research_options:e2625ca1b58059aa','version':''},'view':'full'}
    with pytest.raises(ValidationError):
        RunReadArgs.model_validate(observed)
    schema = public_tool_schema(RunReadArgs)
    # The native/JSON provider sees the nonempty-version requirement too.
    assert schema['$defs']['RunSourceRef']['properties']['version']['minLength'] == 1


def test_valid_empty_section_and_unicode_byte_boundary_remain_supported():
    ref = RunSourceRef(source_id='я'*48, version='я'*64, section='', scope='controller')
    assert Source(ref.source_id,ref.version,'text',ref.scope,section=ref.section).section == ''
    assert RunReadArgs(source=ref).source == ref


@pytest.mark.parametrize('mode', ['native', 'json'])
def test_invalid_source_is_not_accepted_and_published_id_route_still_reads(tmp_path, monkeypatch, mode):
    from automl_agent.budget import Limits
    from automl_agent.contracts import TOOL_SCHEMAS
    from automl_agent.controller import Controller
    from conftest import write_task
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1), run_id='source_contract')
    try:
        c.no_llm = True
        c.prepare()
        snap = c.context_memory.snapshot(c.system_prompt)
        prepared = c.context_memory.compressor.prepare(snap, mode, budget=262144)
        command = dict(operation_id='bad_source', tool='read_run_records', arguments={
            'source':{'source_id':'research_options:e2625ca1b58059aa','version':'','scope':'reference'}})
        message = {'content':json.dumps(command)}
        if mode == 'native':
            message = {'tool_calls':[{'id':'bad', 'type':'function', 'function':{'name':command['tool'],
                'arguments':json.dumps({'operation_id':command['operation_id'], **command['arguments']})}}]}
        with pytest.raises(ValidationError):
            c.context_memory.compressor.accept_response(prepared.request_id,
                {'choices':[{'message':message, 'finish_reason':'tool_calls' if mode == 'native' else 'stop'}]},
                c.context_memory.validate)
        assert c.context_memory.store.db.execute('SELECT COUNT(*) FROM cc_receipts').fetchone()[0] == 0
        assert c.journal.operation('bad_source') is None
        route = json.loads(snap.core_json)['current_work']['research_options']
        assert 'ids' in route['arguments'] and 'source' not in route['arguments']
        result = c.run_tools.read(TOOL_SCHEMAS['read_run_records'].model_validate(route['arguments']))
        assert result['status'] == 'success'
        assert c.budget.counter('llm_calls') == 0
    finally:
        c.close()
