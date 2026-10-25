"""Bounded controller facts linking decisions to durable operation outcomes."""
from __future__ import annotations
import json
from context_compressor import Source, Record
from .util import canonical, digest, utf8_short
from .tool_results import payload


def publish_guides(c, store):
    entries = []
    for name, content in sorted(c.prompt_contract['guides'].items()):
        source = Source('guide:' + name, digest(content), content, 'reference',
            'external_data', origin='project_runtime_guide')
        route = store.put_source(source)
        entries.append({'id': name, 'when': {
            'candidate_interface': 'Before implementing an unfamiliar custom component',
            'runtime_guidance': 'Exact report, model, transport and resource adapter rules',
            'research_policy': 'Recipe inheritance, composition and evidence interpretation',
        }[name], 'read': route})
    content = canonical({'version': 1, 'guides': entries,
        'use': 'Read a relevant interface before unfamiliar operations. No mandatory reading sequence.',
        'authority': 'Local reference; execution contracts and controller permissions remain authoritative.'})
    source = Source('startup_guides', digest(content), content, 'reference',
        'external_data', origin='project_guide_catalog')
    route = store.put_source(source)
    store.publish_records([Record('startup_guides:' + source.version[:16], source.source_id, source.version, source.scope,
        kind='reference', component='startup_guides', engine='controller', status='available')])
    return route


def decision_frame(c):
    frame = {'version': 1, 'facts_source': 'controller_journal'}
    row = c.journal.db.execute("""SELECT record FROM operations ORDER BY
        COALESCE(json_extract(record,'$.ended_at'),json_extract(record,'$.started_at')) DESC,rowid DESC LIMIT 1""").fetchone()
    if row:
        operation = json.loads(row[0])
        frame['last_operation'] = {key: operation.get(key) for key in
            ('operation_id', 'tool', 'status', 'outcome', 'experiment_id')}
        frame['last_operation']['current_protocol'] = (operation.get('protocol_hash') == c.plan.hash
            if operation.get('protocol_hash') is not None else None)
        result = operation.get('result') or {}
        frame['analysis_status'] = result.get('status')
        body = payload(result)
        if isinstance(body, dict) and isinstance(body.get('status'), str):
            frame['analysis_status'] = utf8_short(body['status'], 64)
    else:
        frame['last_operation'] = None
    # A rejected proposal or failed generation is not an executed operation.
    last = c.journal.get('last_tool_result') or {}
    if last.get('status') in {'command_validation', 'model_outcome_unknown', 'model_unavailable', 'model_transport'}:
        frame['last_request_status'] = last['status']
    frame['limitation'] = 'Operation outcome is not proof of ML gain; question/hypothesis remain agent assumptions.'
    return frame
