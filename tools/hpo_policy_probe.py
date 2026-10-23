"""One preregistered prompt-only HPO policy experiment on a terminal decision state."""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import diagnostic_3h as d
from tools.citation_interface_probe import controller, close
from automl_agent.contracts import TOOL_SCHEMAS
from automl_agent.compatibility import require_compatible
from automl_agent.qwen_stream import completion_from_sse, IncompleteModelStream
from context_compressor.transport import Transport

ANCHOR = 'reference_ids are API IDs; search_space_ref names a space; evidence_refs uses returned kind/id/version/collection.'
INSTRUCTION = ('Once a useful model family and preprocessing are established and the next change only tunes hyperparameters, '
    'prefer one bounded tune_model (engine=optuna) with a small typed search_space and finite n_trials over separate Qwen calls for individual settings. '
    'Keep the metric, data roles and validation fixed; protect the selected fit and finalization reserve. '
    'If search is unsuitable, explain why and use one direct comparison.\n')


def validate(output, response):
    body = d.read(output / 'request.json'); c = controller(output, body)
    try:
        t = Transport(system='Validate one action.', tool_schemas={k: v.model_json_schema() for k, v in TOOL_SCHEMAS.items()})
        command = t.parse_response(response, 'json')
        assert command.tool == 'tune_model', 'The registered gate requires an immediately admissible tune_model'
        args = TOOL_SCHEMAS[command.tool].model_validate(command.as_dict()['arguments'])
        assert args.pipeline.engine == 'optuna' and args.pipeline.search_owner in {'none', 'optuna'}
        assert args.pipeline.search_space and args.pipeline.n_trials is not None and 4 <= args.pipeline.n_trials <= 12
        require_compatible(args.pipeline, c.task, c.plan.strategy, c.capabilities)
        resources = json.loads(body['messages'][-1]['content'])['core']['policies_resources']
        assert args.budget_seconds + resources['reserve_seconds'] <= resources['remaining_seconds']
        for identifier in args.method_refs: c.references.catalog_entry('methods', identifier, c.task, c.plan)
        assert all(c.references.known_document(x) for x in args.reference_ids)
        if args.recipe_ref: c.references.catalog_entry('recipes', args.recipe_ref, c.task, c.plan)
        if args.search_space_ref: c.references.catalog_entry('search_spaces', args.search_space_ref, c.task, c.plan)
        before = {t: c.journal.db.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0] for t in ('operations', 'experiments')}
        c.context_memory.validate(command)
        assert before == {t: c.journal.db.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0] for t in before}
        d.write(output / 'validated_command.json', command.as_dict())
        return {'tool': command.tool, 'pipeline': args.pipeline.model_dump(mode='json'),
            'budget_seconds': args.budget_seconds, 'full_controller_admission': True,
            'execution_count': 0, 'quality_effect_claim': False}
    finally:
        close(c)


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    s = d.Store(args.source)
    a = next(a for a in s.index()['artifacts'].values() if a.get('name') == 'qwen_request.json' and a.get('identity') == args.identity)
    original = json.loads(gzip.decompress((s.root / a['path']).read_bytes()))
    assert not original.get('tools') and original['stream'] and original['response_format'] == {'type': 'json_object'}
    assert original['reasoning']['effort'] == 'xhigh'
    body = copy.deepcopy(original); system = body['messages'][0]['content']
    assert system.count(ANCHOR) == 1
    body['messages'][0]['content'] = system.replace(ANCHOR, INSTRUCTION + ANCHOR)
    context = json.loads(body['messages'][-1]['content'])
    journal = args.source / 'classification/task/.agent_work/diagnostic/journal.sqlite'
    assert not Path(str(journal) + '-wal').exists() or Path(str(journal) + '-wal').stat().st_size == 0
    clone = args.output / 'journal'; clone.mkdir(); shutil.copyfile(journal, clone / 'journal.sqlite')
    c = controller(args.output, body)
    try:
        resources = context['core']['policies_resources']; epoch = resources['deadline_epoch'] - resources['remaining_seconds']
        future = [i for i, raw in c.journal.db.execute('SELECT id,record FROM operations') if json.loads(raw)['started_at'] > epoch]
        future_ids = []
        for i in future:
            value = json.loads(c.journal.db.execute('SELECT record FROM operations WHERE id=?', (i,)).fetchone()[0])
            if value.get('experiment_id'): future_ids.append(value['experiment_id'])
        with c.journal.db:
            c.journal.db.executemany('DELETE FROM operations WHERE id=?', [(i,) for i in future])
            c.journal.db.executemany('DELETE FROM experiments WHERE id=?', [(i,) for i in future_ids])
        work = context['core']['current_work']
        c.journal.put_many({'state': 'searching', 'leader_id': work['leader_id'], 'baseline_id': work['baseline_id'], 'protocol_revision': None})
        assert c.journal.operation(work['suggested_new_operation_id']) is None
        d.write(args.output / 'request.json', body); (args.output / 'instruction.txt').write_text(INSTRUCTION)
        d.write(args.output / 'REGISTRATION.json', {'A_identity': args.identity, 'source_run': str(args.source),
            'A': 'Recorded actual single direct LightGBM parameter change after a successful model-family experiment',
            'B': 'Same JSON state, tools, generation settings, response budget and data; add bounded-HPO preference only',
            'criterion': 'One admissible tune_model, Optuna, finite4..12trials, nonempty typed search space, existing admission and budget/provenance pass; invalid/inconclusive => do not promote, no repeated generation',
            'next_gate': 'If generation passes, execute public fixed-decision HPO offline; require actual trials on fit-only inner split, frozen winner and completed control/refit CSV',
            'claim': 'Routing/efficiency hypothesis; no ML quality uplift claimed from generation alone',
            'runtime_changed': False, 'production_archive': 'v049', 'new_paid_calls': 0,
            'request_sha256': d.digest(args.output / 'request.json'), 'later_operations_excluded': future})
        print(json.dumps({'prepared': True, 'runtime_changed': False, 'request_bytes': (args.output / 'request.json').stat().st_size}))
    finally:
        close(c)


def execute(args):
    registration = d.read(args.output / 'REGISTRATION.json')
    assert registration['request_sha256'] == d.digest(args.output / 'request.json')
    assert not (args.output / 'REPORT.json').exists()
    load_dotenv(ROOT / '.env', override=False)
    key = d.limits.load_key(ROOT / '.env'); d.register_secret(key)
    before = d.limits.snapshot(d.limits.fetch_info(key), key, 'before'); d.write(args.output / 'api_limits_before.json', before)
    print(json.dumps({'remaining_before': before['remaining_budget'], 'limit': before['limits'].get('max_budget')}), flush=True)
    assert before['remaining_budget'] is not None and before['remaining_budget'] >= 20
    assert not before['budget_exhausted'] and not before['limits'].get('blocked')
    assert before['limits']['budget_reset_at'] == d.read(args.anchor)['limits']['budget_reset_at'], 'Period changed; reconcile400 allowance'
    result = {'admissible': False, 'upstream_outcome_unknown': False, 'new_paid_calls': 1, 'retries': 0,
        'training_executed': False, 'runtime_changed': False, 'quality_gain_claim': False}
    started = time.monotonic(); previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Probe wall limit')))
    try:
        signal.setitimer(signal.ITIMER_REAL, 900)
        with httpx.Client(transport=httpx.HTTPTransport(retries=0), timeout=httpx.Timeout(900, connect=15)) as client:
            with client.stream('POST', os.environ['OPENAI_BASE_URL'].rstrip('/') + '/chat/completions',
                    headers={'Authorization': 'Bearer ' + key}, json=d.read(args.output / 'request.json')) as response:
                result['http_status'] = response.status_code
                if response.status_code != 200: response.read(); response.raise_for_status()
                def lines():
                    with gzip.open(args.output / 'response.sse.gz', 'wb') as saved:
                        for line in response.iter_lines(): saved.write(d.scrub(line + '\n')); yield line
                value = completion_from_sse(lines())
        d.write(args.output / 'response.json', value)
        result.update(usage=value.get('usage'), finish_reason=value['choices'][0]['finish_reason'])
        result['validation'] = validate(args.output, value); result['admissible'] = True
    except (httpx.TransportError, TimeoutError, IncompleteModelStream) as error:
        result.update(error=type(error).__name__, upstream_outcome_unknown=True)
    except Exception as error:
        result.update(error=type(error).__name__, detail=d.scrub(str(error)).decode()[:2000])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
        result['elapsed_seconds'] = time.monotonic() - started; d.write(args.output / 'REPORT.json', result)
        try:
            after = d.limits.snapshot(d.limits.fetch_info(key), key, 'after'); after['comparison'] = d.limits.compare(before, after)
            d.write(args.output / 'api_limits_after.json', after)
            result.update(remaining_before=before['remaining_budget'], remaining_after=after['remaining_budget'], shared_interval_increase=after['comparison'].get('increase'))
        except Exception as error:
            d.write(args.output / 'AFTER_LIMITS_ERROR.json', {'error': type(error).__name__, 'remaining_unknown': True})
            result['remaining_after'] = None
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path); p.add_argument('--identity')
    p.add_argument('--output', type=Path, required=True); p.add_argument('--anchor', type=Path)
    p.add_argument('--execute', action='store_true'); p.add_argument('--validate-response', action='store_true')
    args = p.parse_args()
    if args.execute: execute(args)
    elif args.validate_response: print(json.dumps(validate(args.output, d.read(args.output / 'response.json'))))
    else: prepare(args)
