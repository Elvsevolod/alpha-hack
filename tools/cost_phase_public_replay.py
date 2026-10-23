"""One offline public-data fit from a previously rejected recorded decision."""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_agent.budget import Limits
from automl_agent.contracts import TrainArgs, ToolCommand, ExperimentSpec, ExperimentResult, PipelineSpec
from automl_agent.controller import Controller
from automl_agent.qwen import QwenClient
from automl_agent.tool_results import payload
from tools.comparison_protocol import archived_runtime
from tools.diagnostic_3h import validate_csv, write, digest


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    original_work = ROOT / 'outputs/quality_goal_20261023/stream_v045_full3h/classification/task/.agent_work/diagnostic'
    source_db = original_work / 'journal.sqlite'
    assert not Path(str(source_db) + '-wal').exists() or Path(str(source_db) + '-wal').stat().st_size == 0
    with sqlite3.connect('file:' + str(source_db.resolve()) + '?mode=ro&immutable=1', uri=True) as db:
        task_spec = json.loads(db.execute("SELECT value FROM metadata WHERE key='task'").fetchone()[0])
        spec = json.loads(db.execute('SELECT spec FROM experiments WHERE id=?', (args.experiment,)).fetchone()[0])
        original_result = json.loads(db.execute('SELECT result FROM experiments WHERE id=?', (args.experiment,)).fetchone()[0])
        parents = []
        parent = spec.get('parent_id')
        while args.gate == 'cost' and parent and parent != 'baseline':
            row = db.execute('SELECT spec,result FROM experiments WHERE id=?', (parent,)).fetchone()
            parent_spec, parent_result = json.loads(row[0]), json.loads(row[1])
            assert parent_result['status'] != 'completed', 'This fixture imports failed history only'
            parents.append((parent_spec, parent_result)); parent = parent_spec.get('parent_id')
    if args.gate == 'cost':
        assert original_result['status'] == 'timeout' and 'Measured pilot' in original_result['error']
    else:
        assert 'Unsupported parameters' in original_result['error'] and 'feature_fraction' in original_result['error']
        spec = copy.deepcopy(spec); spec.update(parent_id=None, search_action='draft', debug_root=None)
    task = args.output / 'task'; task.mkdir()
    inputs = {}
    for name in ('task.md', 'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv'):
        source = ROOT / 'data_for_participants/Многоклассовая классификация' / name
        shutil.copyfile(source, task / name); inputs[name] = digest(source)
    shutil.copyfile(ROOT / 'data_for_participants/requirements.txt', task / 'requirements.txt')
    inputs['requirements.txt'] = digest(task / 'requirements.txt')
    os.environ.update(OPENAI_API_KEY='offline-no-network', OPENAI_BASE_URL='http://fixture.invalid/v1')
    for name in ('OPENAI_EXTRA_BODY', 'AGENT_RESEARCH_FOCUS'):
        os.environ.pop(name, None)
    def resolve(self, system, request):
        assert request['source_hash'] == task_spec['source_hash']
        return {'request_id': request['request_id'], 'source_hash': request['source_hash'], 'status': 'proposed',
            'task_spec': task_spec, 'evidence': {field: ['task', 'inventory'] for field in task_spec}}
    QwenClient.resolve = resolve
    original_source = archived_runtime(ROOT / 'dist/solution_v048.zip')['automl_agent/completion.py'].decode()
    node = next(n for n in ast.parse(original_source).body if isinstance(n, ast.FunctionDef) and n.name == 'measured_cost')
    namespace = {}; exec(ast.get_source_segment(original_source, node), namespace)
    old_alias_rejected = False
    if args.gate == 'feature_alias':
        import automl_agent.pipelines as pipelines
        old_namespace = vars(pipelines).copy()
        old_module = archived_runtime(ROOT / 'dist/solution_v049.zip')['automl_agent/pipelines.py'].decode()
        for name in ('effective_parameters', 'make_estimator'):
            old_node = next(n for n in ast.parse(old_module).body if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(ast.get_source_segment(old_module, old_node), old_namespace)
        try:
            old_namespace['make_estimator'](PipelineSpec.model_validate(spec['pipeline']), True, {'cpu': 2})
        except ValueError as error:
            assert 'Unsupported parameters' in str(error) and 'feature_fraction' in str(error)
            old_alias_rejected = True
        assert old_alias_rejected
    write(args.output / 'REGISTRATION.json', {'experiment': args.experiment, 'original_error': original_result['error'],
        'budget_seconds': spec['budget_seconds'], 'imported_failed_parent_ids': [s['experiment_id'] for s, r in parents], 'inputs': inputs, 'organizer_requests': 0,
        'gate': args.gate, 'old_alias_constructor_rejection': old_alias_rejected,
        'criterion': 'Recorded pipeline completes within unchanged budget; old adapter rejects feature_fraction (alias gate) or old cost rejects attempt (cost gate); completed control/refit/full_train_refitCSV',
        'ancestry_adjustment': 'Detached recorded pipeline to draft; no imported completed-parent artifacts' if args.gate == 'feature_alias' else None,
        'scope': 'Frozen real public task contract + recorded agent pipeline/hypothesis; offline fixed decision, no autonomous-generation or quality-effect claim',
        'runtime_hashes': {name: digest(ROOT / 'automl_agent' / name) for name in ('completion.py', 'worker.py')}})
    c = Controller(task / 'task.md', Limits(seconds=10800, cpu=2, memory_mb=4096), run_id='phase_replay')
    started = time.monotonic()
    try:
        c.prepare()
        for parent_spec, parent_result in reversed(parents):
            c.journal.save_experiment(ExperimentSpec.model_validate(parent_spec), ExperimentResult.model_validate(parent_result))
        arguments = TrainArgs.model_validate({key: value for key, value in spec.items() if key in TrainArgs.model_fields})
        command = ToolCommand(operation_id='recorded_cost_candidate', tool='train_model', arguments=arguments.model_dump(mode='json'))
        result = payload(c.dispatch(command))
        operation = c.journal.operation(command.operation_id)
        trained_spec, trained = c.lookup(operation.experiment_id)
        pilot_path = c.workdir / 'attempts' / operation.experiment_id / 'pilot/worker_result.json'
        pilot = json.loads(pilot_path.read_text())
        events = [json.loads(raw) for (raw,) in c.journal.db.execute("SELECT payload FROM events WHERE kind='automatic_pilot'")]
        measured = next(e['seconds'] for e in events if e['experiment_id'] == operation.experiment_id)
        old_fit = namespace['measured_cost'](pilot['training_summary'], measured, len(c.plan.fit), len(c.plan.fit))
        forecast = next(json.loads(raw) for (raw,) in c.journal.db.execute("SELECT payload FROM events WHERE kind='candidate_completion_forecast'")
            if json.loads(raw)['experiment_id'] == operation.experiment_id)
        c.freeze('Fixed recorded decision complete; verify protected finalization')
        c.finalize(); c.report()
        csv = validate_csv({'kind': 'classification', 'directory': str(task)})
        publication = c.journal.get('publication') or {}
        report = {'elapsed_seconds': time.monotonic() - started, 'original_experiment': args.experiment,
            'gate': args.gate, 'old_alias_constructor_rejection': old_alias_rejected, 'normalized_hyperparameters': trained_spec.pipeline.hyperparameters,
            'candidate_status': trained.status, 'candidate_seconds': trained.seconds,
            'candidate_budget_seconds': trained_spec.budget_seconds, 'old_fit_forecast_on_same_measurement': old_fit,
            'new_fit_forecast_on_same_measurement': forecast['candidate_fit_seconds'],
            'old_would_reject_before_fit': old_fit > trained_spec.budget_seconds - measured,
            'state': c.state, 'control': c.journal.get('control_status'), 'refit': c.journal.get('refit_status'),
            'csv': csv, 'publication_scope': publication.get('scope'), 'organizer_requests': 0,
            'stock_default_seconds': c.budget.limits.seconds, 'quality_effect_claim': False}
        report['passed'] = (trained.status == 'completed' and (old_alias_rejected if args.gate == 'feature_alias' else report['old_would_reject_before_fit'])
            and trained.seconds < trained_spec.budget_seconds and c.state == 'completed'
            and report['control'] == report['refit'] == 'completed' and csv['passed']
            and publication.get('scope') == 'full_train_refit')
        write(args.output / 'REPORT.json', report); print(json.dumps(report), flush=True)
        return 0 if report['passed'] else 1
    finally:
        c.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gate', choices=('cost', 'feature_alias'), default='cost')
    p.add_argument('--experiment', default='exp_c755af55c14dd98f')
    args = p.parse_args(); args.output = args.output.resolve()
    raise SystemExit(run(args))
