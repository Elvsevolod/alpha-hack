"""Audit saved requests without exporting sensitive context or altering journals."""
import hashlib
import json
import sqlite3

import pytest

from tools.context_structure_audit import audit


@pytest.mark.parametrize('mode', ['native', 'json'])
def test_audit_preserves_journal_and_exports_only_structure(tmp_path, mode):
    path = tmp_path / 'journal.sqlite'
    context = {'core': {'task': {'description': 'PRIVATE_TASK_SENTINEL'}},
        'observations': [{'content': 'PRIVATE_RESULT_SENTINEL'}], 'history': [],
        'compaction': {'stage': 0}}
    wire = {'messages': [{'role': 'system', 'content': 'PRIVATE_PROMPT_SENTINEL'},
        {'role': 'user', 'content': ''}]}
    if mode == 'native':
        wire['tools'] = [{'type': 'function', 'function': {'name': 'example', 'parameters': {}}}]
    else:
        context['tool_argument_schemas'] = {'example': {'type': 'object'}}
    wire['messages'][1]['content'] = json.dumps(context)
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE cc_requests(wire_json TEXT)')
        connection.execute('INSERT INTO cc_requests VALUES (?)', (json.dumps(wire),))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = audit(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    serialized = json.dumps(result)
    assert 'PRIVATE_' not in serialized
    assert result['only_snapshot_messages']
    assert result['requests'][0]['logical_tool_count'] == 1
    assert result['requests'][0]['transport'] == mode
    assert result['requests'][0]['observation_count'] == 1


def test_empty_journal_does_not_claim_a_transport_observation(tmp_path):
    path = tmp_path / 'empty.sqlite'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE cc_requests(wire_json TEXT)')
    assert audit(path)['only_snapshot_messages'] is None
