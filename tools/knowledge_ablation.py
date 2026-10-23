"""Equal-budget reference-layer comparison on already copied provided tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from automl_agent.budget import Limits
from automl_agent.controller import Controller
from automl_agent.util import atomic_json, read_json
from tools.archive_development_runs import archive


def run(tasks, output, seconds, transport='auto'):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(tasks.glob('*/task.md')):
        for mode in ('none', 'documentation', 'full'):
            receipt = output / f'{path.parent.name}_{mode}.json'
            if receipt.is_file():
                results.append(read_json(receipt)); continue
            controller = Controller(path, Limits(seconds=seconds, cpu=4, max_candidates=6, max_calls=12),
                run_id=f'knowledge_{mode}_42', training_seed=42, transport=transport,
                research_focus='Equal-budget development comparison. Form useful hypotheses from data, keep resource reserves and use permitted sources. Training seed is fixed at 42. No external data.')
            original = controller.references.query
            requests = []

            def query(args, *a, **kw):
                started = time.monotonic()
                owned = args.reference_id == 'candidate_interface' or args.mode == 'api' and args.query in {'', 'candidate_interface', 'build_pipeline'}
                if not owned and (mode == 'none' or mode == 'documentation' and args.mode in {'methods', 'recipes', 'search_space'}):
                    value = {'status': 'disabled_reference_layer', 'ablation_mode': mode,
                        'reason': 'Declared development comparison; use available sources or a justified finish'}
                else:
                    value = original(args, *a, **kw)
                requests.append({'mode': args.mode, 'status': value.get('status'), 'seconds': time.monotonic() - started})
                return value

            controller.references.query = query
            try:
                report = controller.run()
                records = controller.records()
                attempted = [(s, r) for s, r in records if s.status != 'baseline']
                successful = [r for s, r in attempted if r and r.status == 'completed' and r.score is not None]
                row = {'reference_layer': mode, 'kind': controller.task.kind if controller.task else None,
                    'budget_seconds': seconds, 'cpu': 4, 'training_seed': 42,
                    'data_version': controller.dataset.source_hash, 'validation_hash': controller.plan.hash if controller.plan else None,
                    'valid_csv': report['state'] == 'completed' and report['publication'] is not None,
                    'trained_result': any(r and r.status == 'completed' and r.score is not None for s, r in records),
                    'attempted_candidates': len(attempted), 'successful_candidates': len(successful),
                    'successful_candidate_fraction': len(successful) / len(attempted) if attempted else None,
                    'experiment_api_errors': [r.error for s, r in attempted if r and r.status in {'error', 'import_error', 'unsupported_mode'}],
                    'qwen_failures': controller.journal.db.execute("SELECT count(*) FROM events WHERE kind='qwen_failure'").fetchone()[0],
                    'peak_worker_memory_mb': max((r.peak_memory_mb for s, r in records if r), default=0),
                    'research_score': report['raw_best'], 'control_scores': report['control_scores'],
                    'control': report['control_status'], 'refit': report['refit_status'],
                    'first_csv_seconds': report['first_csv_seconds'], 'elapsed_seconds': report['elapsed_seconds'],
                    'token_usage': report['token_usage'], 'reference_requests': requests,
                    'run_report': str(controller.workdir / 'run_report.json')}
                atomic_json(receipt, row); results.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                for folder in (controller.workdir / 'attempts', controller.workdir / 'finalization'):
                    for file in folder.rglob('*.json'):
                        if file.name.endswith('_X.json') or file.name == 'fit_y.json': file.unlink()
            finally:
                controller.close()
                if receipt.is_file():
                    packed = archive(controller.workdir, min_age=0)
                    if packed:
                        row['archived_models'] = str(controller.workdir / 'execution_artifacts.zip')
                        atomic_json(receipt, row)
            atomic_json(output / 'REPORT.json', {'results': results, 'same_budget_seconds': seconds,
                'training_seed': 42, 'hidden_test_validated': False,
                'interpretation': 'One live development run per mode and task; stochastic decisions do not establish a causal quality improvement'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--transport', choices=['auto','native','json'], default='auto')
    args = parser.parse_args(); load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
    run(args.tasks, args.output, args.seconds, args.transport)
