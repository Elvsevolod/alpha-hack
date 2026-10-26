"""Bounded controller facts linking decisions to durable operation outcomes."""
from __future__ import annotations
import json
from context_compressor import Source, Record
from .util import canonical, digest, utf8_short
from .tool_results import payload


def resource_budget(c):
    """Paid-call facts at snapshot time, before the pending generation is charged."""
    from .critic import EXPENSIVE_SECONDS
    remaining = c.budget.snapshot()
    calls = remaining["qwen_calls"]
    candidates = remaining["candidates"]
    return {"local_qwen_calls_limit": c.budget.limits.max_calls,
        "qwen_calls_remaining_at_snapshot": None if calls is None else max(0, calls),
        "qwen_calls_remaining_after_one_generation": None if calls is None else max(0, calls - 1),
        "local_candidate_limit": c.budget.limits.max_candidates,
        "candidates_remaining": None if candidates is None else max(0, candidates),
        "local_qwen_seconds_limit": c.budget.limits.max_llm_seconds,
        "qwen_seconds_used": remaining["llm_seconds_used"],
        "critic_call_cost_if_required_and_uncached": 1,
        "critic_disabled": bool(getattr(c, "no_llm", False) or getattr(c.qwen, "disabled", False)),
        "critic_measured_threshold_seconds": EXPENSIVE_SECONDS,
        "critic_condition": "Custom or measured expensive proposals; cheap native fits and cached reviews may need no additional call.",
        "scope": "Immutable decision snapshot before its first generation; transport retries consume additional calls. Null means no local cap; provider limits are not inferred."}


def bounded_read_route(c, store, source, component):
    """Use existing record pagination for v4; retain old byte routes for old runs."""
    route = store.put_source(source)
    if c.prompt_contract['version'] < 4:
        return route
    identifier = source.source_id + ':' + source.version[:16]
    store.publish_records([Record(identifier, source.source_id, source.version, source.scope,
        kind='reference', component=component, engine='controller', status='available')])
    return {'tool': 'read_run_records', 'arguments': {'ids': [identifier], 'max_chars': 12000}}


def guide_route(c, store, name):
    content = c.prompt_contract['guides'][name]
    source = Source('guide:' + name, digest(content), content, 'reference',
        'external_data', origin='project_runtime_guide')
    return bounded_read_route(c, store, source, name)


def parameter_repair(c, error):
    from .pipelines import UnsupportedParameters
    if c.prompt_contract['version'] < 6 or not isinstance(error, UnsupportedParameters):
        return None
    import json
    # Use the run-pinned public adapter contract, not the full library API.
    contract = json.loads(c.prompt_contract['guides']['runtime_parameters'])
    unsupported = [utf8_short(name, 96) for name in error.unsupported[:8]]
    return {'model': error.model, 'unsupported': unsupported,
        'unsupported_total': len(error.unsupported),
        'unsupported_names_truncated': unsupported != error.unsupported,
        'accepted_parameter_names': contract['models'][error.model],
        'read': guide_route(c, c.context_memory.store, 'runtime_parameters'),
        'next_action': 'Correct the declared recipe with a new operation ID. Names do not establish value compatibility; no settings were substituted.'}


def publish_guides(c, store):
    entries = []
    for name, content in sorted(c.prompt_contract['guides'].items()):
        route = guide_route(c, store, name)
        entries.append({'id': name, 'when': {
            'candidate_interface': 'Before implementing an unfamiliar custom component',
            'runtime_guidance': 'Exact report, model, transport and resource adapter rules',
            'research_policy': 'Recipe inheritance, composition and evidence interpretation',
            'runtime_parameters': 'Direct-model parameter names and declared transformations; verify before claiming an adapter capability is absent',
        }[name], 'read': route})
    content = canonical({'version': 1, 'guides': entries,
        'use': 'Read a relevant interface before unfamiliar operations. No mandatory reading sequence.',
        'authority': 'Local reference; execution contracts and controller permissions remain authoritative.'})
    source = Source('startup_guides', digest(content), content, 'reference',
        'external_data', origin='project_guide_catalog')
    route = bounded_read_route(c, store, source, 'startup_guides')
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
