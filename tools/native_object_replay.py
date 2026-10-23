"""Offline replay of lossless native object decoding; no model calls or fits."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.diagnostic_3h import Store, write
from tools.action_schema_probe import validate_action
from tools.comparison_protocol import archived_runtime
from automl_agent.budget import Budget, Limits
from automl_agent.capabilities import CapabilityRegistry
from automl_agent.context_memory import ContextMemory
from automl_agent.contracts import TOOL_SCHEMAS, TaskSpec, ValidationPlan
from automl_agent.controller import Controller
from automl_agent.journal import Journal
from automl_agent.knowledge import AgentReferences
from automl_agent.qwen_stream import completion_from_sse
from context_compressor.transport import Transport, _object


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    store = Store(args.source)
    artifacts = [a for a in store.index()['artifacts'].values() if a.get('identity') == args.identity]
    def read(name):
        item = next(a for a in artifacts if a['name'] == name)
        return gzip.decompress((store.root / item['path']).read_bytes())
    body = json.loads(read('qwen_request.json'))
    raw = read('qwen_response.txt')
    response = json.loads(raw) if raw.lstrip().startswith(b'{') else completion_from_sse(raw.decode().splitlines())
    transport = Transport(system='Execute one permitted action.',
        tool_schemas={n: s.model_json_schema() for n, s in TOOL_SCHEMAS.items()})
    baseline_source = archived_runtime(args.baseline_archive)['context_compressor/transport.py']
    baseline_namespace = {'__name__': 'context_compressor._baseline_transport', '__package__': 'context_compressor'}
    exec(compile(baseline_source, 'archived_transport.py', 'exec'), baseline_namespace)
    baseline_transport = baseline_namespace['Transport'](system='Execute one permitted action.',
        tool_schemas={n: s.model_json_schema() for n, s in TOOL_SCHEMAS.items()})
    old = baseline_transport.parse_response(response, 'native')
    try:
        TOOL_SCHEMAS[old.tool].model_validate(old.as_dict()['arguments'])
    except ValueError as error:
        old_error = type(error).__name__
    else:
        raise AssertionError('This replay requires a terminal typed-argument rejection')
    repaired = copy.deepcopy(response)
    calls = repaired['choices'][0]['message']['tool_calls']
    assert len(calls) == 1
    values = _object(calls[0]['function']['arguments'])
    decoded = []
    for name in ('pipeline', 'hypothesis'):
        if isinstance(values.get(name), str):
            values[name] = _object(values[name])
            decoded.append(name)
    assert decoded
    calls[0]['function']['arguments'] = json.dumps(values)
    command = transport.parse_response(response, 'native')
    assert command.as_dict() == transport.parse_response(repaired, 'native').as_dict()
    journal_path = args.source / 'regression/task/.agent_work/diagnostic/journal.sqlite'
    assert not Path(str(journal_path) + '-wal').exists() or Path(str(journal_path) + '-wal').stat().st_size == 0
    clone = args.output / 'journal'; clone.mkdir()
    shutil.copyfile(journal_path, clone / 'journal.sqlite')
    c = Controller.__new__(Controller)
    c.workdir = clone
    c.journal = Journal(clone)
    c.context_memory = ContextMemory(c)
    c.capabilities = CapabilityRegistry(clone / 'capabilities.json')
    c.references = AgentReferences(ROOT, c.capabilities)
    try:
        c.task = TaskSpec.model_validate(c.journal.get('task'))
        c.plan = ValidationPlan.model_validate(c.journal.get('validation'))
        c.dataset = SimpleNamespace(source_hash=c.task.source_hash)
        resources = json.loads(body['messages'][-1]['content'])['core']['policies_resources']
        request_epoch = resources['deadline_epoch'] - resources['remaining_seconds']
        future_operations = [identifier for identifier, record in c.journal.db.execute('SELECT id,record FROM operations')
            if json.loads(record)['started_at'] > request_epoch]
        with c.journal.db:
            c.journal.db.executemany('DELETE FROM operations WHERE id=?', [(i,) for i in future_operations])
        c.journal.put('state', 'searching')
        c.budget = Budget(c.journal, Limits(), time.time())
        c.budget.deadline = time.time() + resources['remaining_seconds']
        c.budget_mode = c.journal.get('budget_mode', 'research')
        c.qwen = SimpleNamespace(reserve_seconds=resources['reserve_seconds'])
        baseline, _ = c.lookup('baseline')
        subset = validate_action(body, repaired, c.task, baseline, c.journal.db)
        before = {t: c.journal.db.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0]
                  for t in ('operations', 'experiments')}
        c.context_memory.validate(command)
        after = {t: c.journal.db.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0]
                 for t in before}
        assert before == after
        result = {'source_identity': args.identity, 'source_response_sha256': hashlib.sha256(raw).hexdigest(),
            'old_typed_error': old_error, 'decoded_fields': decoded, 'subset': subset,
            'controller_admissibility_passed': True, 'no_operation_or_experiment_created': True,
            'request_epoch': request_epoch, 'later_operations_excluded_from_clone': future_operations,
            'execution_count': 0, 'organizer_requests': 0, 'runtime_changed': True,
            'baseline_archive': str(args.baseline_archive), 'original_response_accepted_without_rewrite': True,
            'scope': 'Frozen task/protocol and remaining time, restored searching state in a cloned journal; no dispatch, receipt or fit'}
        write(args.output / 'REPORT.json', result)
        print(json.dumps(result))
    finally:
        c.references.close(); c.context_memory.store.close(); c.journal.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--identity', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline-archive', type=Path, default=Path('dist/solution_v046.zip'))
    run(p.parse_args())
