"""One registered provenance-interface experiment; never executes training."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import diagnostic_3h as d
from tools.action_schema_probe import validate_action
from automl_agent.budget import Budget, Limits
from automl_agent.capabilities import CapabilityRegistry
from automl_agent.context_memory import ContextMemory
from automl_agent.contracts import TaskSpec, ValidationPlan, TOOL_SCHEMAS, TOOL_DESCRIPTIONS
from automl_agent.controller import Controller, SYSTEM
from automl_agent.journal import Journal
from automl_agent.knowledge import AgentKnowledge, AgentReferences
from automl_agent.qwen_stream import completion_from_sse, IncompleteModelStream
from automl_agent.util import canonical
from context_compressor.transport import Transport


def sources():
    return {name: d.digest(ROOT / 'automl_agent' / name) for name in ('contracts.py', 'controller.py', 'knowledge.py')}


def controller(output, body):
    clone = output / 'journal'
    c = Controller.__new__(Controller)
    c.workdir = clone
    c.journal = Journal(clone)
    c.context_memory = ContextMemory(c)
    c.capabilities = CapabilityRegistry(clone / 'capabilities.json')
    c.references = AgentReferences(ROOT, c.capabilities)
    c.task = TaskSpec.model_validate(c.journal.get('task'))
    c.plan = ValidationPlan.model_validate(c.journal.get('validation'))
    c.dataset = SimpleNamespace(source_hash=c.task.source_hash)
    resources = json.loads(body['messages'][-1]['content'])['core']['policies_resources']
    c.budget = Budget(c.journal, Limits(), time.time())
    c.budget.deadline = time.time() + resources['remaining_seconds']
    c.budget_mode = c.journal.get('budget_mode', 'research')
    c.qwen = SimpleNamespace(reserve_seconds=resources['reserve_seconds'])
    return c


def close(c):
    c.references.close()
    c.context_memory.store.close()
    c.journal.close()


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    store = d.Store(args.source)
    artifact = next(a for a in store.index()['artifacts'].values()
        if a.get('name') == 'qwen_request.json' and a.get('identity') == args.identity)
    original = json.loads(gzip.decompress((store.root / artifact['path']).read_bytes()))
    assert original['tools'] and original['stream']
    assert original['reasoning']['effort'] in {'medium', 'xhigh'}
    context = json.loads(original['messages'][-1]['content'])
    assert context['core']['current_work']['latest_id'] == 'baseline'
    clone = args.output / 'journal'; clone.mkdir()
    shutil.copyfile(args.backup, clone / 'journal.sqlite')
    c = controller(args.output, original)
    try:
        resources = context['core']['policies_resources']
        epoch = resources['deadline_epoch'] - resources['remaining_seconds']
        future = [identifier for identifier, raw in c.journal.db.execute('SELECT id,record FROM operations')
            if json.loads(raw)['started_at'] > epoch]
        with c.journal.db:
            c.journal.db.executemany('DELETE FROM operations WHERE id=?', [(i,) for i in future])
            c.journal.db.execute("DELETE FROM experiments WHERE id!='baseline'")
        c.journal.put_many({'state': 'searching', 'leader_id': 'baseline', 'baseline_id': 'baseline',
            'protocol_revision': None})
        knowledge = AgentKnowledge.__new__(AgentKnowledge); knowledge.c = c
        packet = knowledge.with_evidence_refs(context['reference_evidence'])
        assert packet['materials'] == context['reference_evidence']['materials'] and packet['evidence_refs']
        c.validate_evidence(packet['evidence_refs'])
        t = Transport(system=SYSTEM, tool_schemas={k: {**v.model_json_schema(), 'description': TOOL_DESCRIPTIONS[k]}
            for k, v in TOOL_SCHEMAS.items()}, model=original['model'], response_reserve=original['max_tokens'],
            extra={k: v for k, v in original.items() if k not in {'model', 'max_tokens', 'messages', 'tools',
                'parallel_tool_calls', 'tool_choice', 'stream', 'stream_options'}}, reference_packet=packet)
        body = json.loads(t.render({k: v for k, v in context.items() if k != 'reference_evidence'}, 'native'))
        assert {k: v for k, v in body.items() if k not in {'messages', 'tools'}} == {
            k: v for k, v in original.items() if k not in {'messages', 'tools'}}
        assert json.loads(body['messages'][-1]['content']) == {**context, 'reference_evidence': packet}
        overhead = {m: t.overhead_bytes(m) for m in ('native', 'json')}
        assert max(overhead.values()) <= 49152
        d.write(args.output / 'request.json', body)
        d.write(args.output / 'REGISTRATION.json', {'source_identity': args.identity,
            'hypothesis': 'Canonical copyable packet refs plus explicit required kind/id/version schema prevent the observed provenance rejection',
            'A': 'One actual train_model rejected for missing kind and wrong evidence version; no new A call',
            'B': 'Same decision state/settings, coherent provenance interface only',
            'criterion': 'Exactly one distinct direct train_model, valid canonical supplied evidence, existing full controller admission passes; no retries on invalid or inconclusive result',
            'training_executed': False, 'quality_gain_claim': False, 'runtime_hashes': sources(),
            'source_request_sha256': hashlib.sha256(canonical(original).encode()).hexdigest(),
            'request_sha256': d.digest(args.output / 'request.json'), 'packet_chars': len(canonical(packet)),
            'ready_refs': packet['evidence_refs'], 'overhead_bytes': overhead,
            'snapshot': 'Linux sqlite backup; clone restored to source baseline-only searching state',
            'new_paid_calls': 0})
        print(json.dumps({'prepared': True, 'packet_chars': len(canonical(packet)), 'overhead_bytes': overhead}))
    finally:
        close(c)


def validate(output, response):
    body = d.read(output / 'request.json')
    c = controller(output, body)
    try:
        t = Transport(system=SYSTEM, tool_schemas={k: v.model_json_schema() for k, v in TOOL_SCHEMAS.items()})
        command = t.parse_response(response, 'native')
        baseline, _ = c.lookup('baseline')
        subset = validate_action(body, response, c.task, baseline, c.journal.db)
        refs = command.as_dict()['arguments'].get('hypothesis', {}).get('evidence_refs', [])
        assert refs, 'This probe requires actual canonical evidence use'
        supplied = json.loads(body['messages'][-1]['content'])['reference_evidence']['evidence_refs']
        assert all(ref in supplied for ref in refs), 'A provenance hint must be copied exactly'
        before = {t: c.journal.db.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0] for t in ('operations', 'experiments')}
        c.context_memory.validate(command)
        after = {t: c.journal.db.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0] for t in before}
        assert before == after
        return {**subset, 'full_controller_admission': True, 'exact_supplied_refs_used': refs,
            'no_operations_or_experiments_created': True}
    finally:
        close(c)


def execute(args):
    registration = d.read(args.output / 'REGISTRATION.json')
    assert registration['runtime_hashes'] == sources(), 'Implementation changed after registration'
    assert registration['request_sha256'] == d.digest(args.output / 'request.json')
    assert not (args.output / 'REPORT.json').exists()
    load_dotenv(ROOT / '.env', override=False)
    key = d.limits.load_key(ROOT / '.env'); d.register_secret(key)
    before = d.limits.snapshot(d.limits.fetch_info(key), key, 'before')
    d.write(args.output / 'api_limits_before.json', before)
    anchor = d.read(args.anchor)
    print(json.dumps({'api_limits_before': before['remaining_budget'], 'limit': before['limits'].get('max_budget')}), flush=True)
    assert before['remaining_budget'] is not None and before['remaining_budget'] >= 20
    assert not before['budget_exhausted'] and not before['limits'].get('blocked')
    assert before['limits'].get('budget_reset_at') == anchor['limits'].get('budget_reset_at'), 'Period changed; reconcile400 allowance'
    result = {'admissible': False, 'upstream_outcome_unknown': False, 'new_paid_calls': 1,
        'retries': 0, 'training_executed': False, 'quality_gain_claim': False}
    started = time.monotonic()
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Probe wall limit')))
    try:
        signal.setitimer(signal.ITIMER_REAL, 900)
        body = d.read(args.output / 'request.json')
        with httpx.Client(transport=httpx.HTTPTransport(retries=0), timeout=httpx.Timeout(900, connect=15)) as client:
            with client.stream('POST', os.environ['OPENAI_BASE_URL'].rstrip('/') + '/chat/completions',
                    headers={'Authorization': 'Bearer ' + key}, json=body) as response:
                result['http_status'] = response.status_code
                if response.status_code != 200:
                    response.read(); response.raise_for_status()
                def lines():
                    with gzip.open(args.output / 'response.sse.gz', 'wb') as saved:
                        for line in response.iter_lines():
                            saved.write(d.scrub(line + '\n')); yield line
                value = completion_from_sse(lines())
        d.write(args.output / 'response.json', value)
        result.update(usage=value.get('usage'), finish_reason=value['choices'][0]['finish_reason'])
        result['validation'] = validate(args.output, value)
        result['admissible'] = True
    except (httpx.TransportError, TimeoutError, IncompleteModelStream) as error:
        result.update(error=type(error).__name__, upstream_outcome_unknown=True)
    except Exception as error:
        result.update(error=type(error).__name__, detail=d.scrub(str(error)).decode()[:2000])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
        result['elapsed_seconds'] = time.monotonic() - started
        d.write(args.output / 'REPORT.json', result)
        try:
            after = d.limits.snapshot(d.limits.fetch_info(key), key, 'after')
            after['comparison'] = d.limits.compare(before, after)
            d.write(args.output / 'api_limits_after.json', after)
            result.update(remaining_before=before['remaining_budget'], remaining_after=after['remaining_budget'],
                shared_interval_increase=after['comparison'].get('increase'))
        except Exception as error:
            d.write(args.output / 'AFTER_LIMITS_ERROR.json', {'error': type(error).__name__, 'remaining_unknown': True})
            result['remaining_after'] = None
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path); p.add_argument('--identity'); p.add_argument('--backup', type=Path)
    p.add_argument('--anchor', type=Path); p.add_argument('--execute', action='store_true')
    p.add_argument('--validate-response', action='store_true')
    args = p.parse_args()
    if args.execute: execute(args)
    elif args.validate_response: print(json.dumps(validate(args.output, d.read(args.output / 'response.json'))))
    else: prepare(args)
