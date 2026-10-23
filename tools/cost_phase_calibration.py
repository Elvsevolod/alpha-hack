"""Read-only retrospective calibration on terminal public-task measurements."""
import ast
import copy
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_agent.completion import measured_cost
from tools.comparison_protocol import archived_runtime
from tools.diagnostic_3h import write

source = archived_runtime(ROOT / 'dist/solution_v048.zip')['automl_agent/completion.py'].decode()
function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'measured_cost')
namespace = {}; exec(ast.get_source_segment(source, function), namespace)
old_cost = namespace['measured_cost']
rows = []
for run in ('baseline_v043_30m', 'json_v044_30m', 'stream_v045_30m', 'stream_v045_full3h'):
    for kind in ('regression', 'classification'):
        work = ROOT / 'outputs/quality_goal_20261023' / run / kind / 'task/.agent_work/diagnostic'
        journal = work / 'journal.sqlite'
        if not journal.exists(): continue
        assert not Path(str(journal) + '-wal').exists() or Path(str(journal) + '-wal').stat().st_size == 0
        db = sqlite3.connect('file:' + str(journal.resolve()) + '?mode=ro&immutable=1', uri=True)
        metadata = {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
        if metadata.get('state') not in {'completed', 'failed', 'contract_unresolved'}:
            raise ValueError('Refuse to read a live journal')
        measurements = {json.loads(raw)['experiment_id']: json.loads(raw) for (raw,) in db.execute(
            "SELECT payload FROM events WHERE kind='automatic_pilot'")}
        for identifier, spec_raw, result_raw in db.execute('SELECT id,spec,result FROM experiments'):
            spec = json.loads(spec_raw); result = json.loads(result_raw) if result_raw else {}
            pilot = work / 'attempts' / identifier / 'pilot/worker_result.json'
            if not pilot.exists() or identifier not in measurements: continue
            raw = json.loads(pilot.read_text())
            if raw.get('status') != 'completed' or not raw.get('training_summary'): continue
            summary = copy.deepcopy(raw['training_summary'])
            protocol = metadata.get('protocols', {}).get(spec['validation_hash']) or metadata.get('validation')
            if not protocol: continue
            fit_rows = len(protocol['fit']); full_rows = fit_rows + len(protocol['validation']) + len(protocol['control'])
            summary['execution_phases']['loaded_fit_rows'] = fit_rows
            seconds = measurements[identifier]['seconds']
            before = old_cost(raw['training_summary'], seconds, fit_rows, fit_rows)
            after = measured_cost(summary, seconds, fit_rows, fit_rows)
            staging = summary['execution_phases'].get('controller_staging_seconds', 0)
            actual = result.get('seconds') if result.get('status') == 'completed' else None
            rows.append({'run': run, 'task': kind, 'id': identifier, 'model': spec['pipeline']['model'],
                'source_protocol': spec['validation_hash'], 'pilot_rows': summary['rows'], 'loaded_fit_rows': fit_rows,
                'old_fit_forecast': before, 'new_fit_forecast': after, 'requested_candidate_seconds': spec['budget_seconds'],
                'old_exceeds_attempt_after_pilot': before > spec['budget_seconds'] - seconds - staging,
                'new_exceeds_attempt_after_pilot': after > spec['budget_seconds'] - seconds - staging,
                'actual_full_attempt_seconds': actual, 'new_forecast_covers_observed_full_attempt': None if actual is None else after >= actual,
                'original_status': result.get('status'), 'original_error': result.get('error'),
                'new_full_refit_forecast': measured_cost(summary, seconds, full_rows, fit_rows)})
        db.close()
report = {'scope': 'Retrospective terminal public-task data; loaded_fit_rows reconstructed from frozen protocol because old workers did not record it. Sources unchanged; not a prospective quality comparison.',
    'organizer_requests': 0, 'training_executed': False, 'rows': rows,
    'summary': {'pilot_cases': len(rows), 'old_exceeds_attempt_after_pilot': sum(r['old_exceeds_attempt_after_pilot'] for r in rows),
        'new_exceeds_attempt_after_pilot': sum(r['new_exceeds_attempt_after_pilot'] for r in rows),
        'observed_completed_cases': sum(r['actual_full_attempt_seconds'] is not None for r in rows),
        'new_forecast_covers_observed_completed': sum(r['new_forecast_covers_observed_full_attempt'] is True for r in rows)}}
write(ROOT / 'outputs/quality_goal_20261023/COST_PHASE_CALIBRATION_01.json', report)
print(json.dumps(report['summary']))
