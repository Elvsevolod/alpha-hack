"""Audit actual short A/B receipts before interpreting their control scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def directed_gap(score_a, score_b, policy):
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (score_a, score_b)):
        raise ValueError('Missing or nonfinite control score')
    gap = (score_b - score_a) * (1 if policy['maximize'] else -1)
    tolerance = max(policy['min_delta_abs'], policy['min_delta_rel'] * abs(score_a))
    return {'score_A': score_a, 'score_B': score_b, 'directed_gain_B_minus_A': gap,
            'tolerance': tolerance, 'regression_requires_rollback': gap < -tolerance}


def compare(directory_a, directory_b, protocol_path):
    protocol = read(protocol_path)
    if protocol.get('status') != 'authorized_preregistered_before_live_results':
        raise ValueError('Missing authorized short protocol')
    manifests = [read(directory / 'MANIFEST.json') for directory in (directory_a, directory_b)]
    reports = [read(directory / 'REPORT.json') for directory in (directory_a, directory_b)]
    checks = {
        'exact_archives': all(manifests[i]['archive_sha256'] == protocol['archive_sha256'][arm]
                              for i, arm in enumerate(('A', 'B'))),
        'duration': all(m['duration_seconds'] == protocol['effective_time_limit_seconds_per_task'] for m in manifests),
        'cpu': all(m['cpu_per_task'] == protocol['cpu_per_task'] for m in manifests),
        'memory': all(m['memory_gib_per_task'] == protocol['memory_gib_per_task'] for m in manifests),
        'budget_caps': all(m['api_budget_cap'] == protocol['api_budget_cap_per_series'] for m in manifests),
        'same_supervisor': manifests[0]['supervisor_sources'] == manifests[1]['supervisor_sources'],
        'technical_and_diagnostic_acceptance': all(r['full_run_passed'] for r in reports),
    }
    tasks = {}
    for kind in ('regression', 'classification'):
        meta = [read(directory / kind / 'analysis/metadata.json') for directory in (directory_a, directory_b)]
        experiments = [read(directory / kind / 'analysis/experiments.json') for directory in (directory_a, directory_b)]
        results = [next((row for row in report['results'] if row['kind'] == kind), {'report': {}}) for report in reports]
        plans = [item.get('validation') or {} for item in meta]
        specs = [item.get('task') or {} for item in meta]
        split_keys = ('strategy', 'fit', 'validation', 'control', 'seed')
        task_checks = {
            'inputs': all(m['jobs'][kind]['input_hashes'] == protocol['basis']['inputs'][kind] for m in manifests),
            'split_rows_and_seed': all(key in plan for plan in plans for key in split_keys) and
                                   all(plans[0][key] == plans[1][key] for key in split_keys),
            'training_seed': all(rows and all(row['spec']['pipeline']['seed'] == protocol['basis']['training_seed']
                                              for row in rows) for rows in experiments),
            'split_seed': all(plan.get('seed') == protocol['basis']['training_seed'] for plan in plans),
            'primary_metric': all(spec.get('metric', {}).get('name') == protocol['basis']['metrics'][kind]['name'] for spec in specs),
            'metric_options': all(spec.get('metric', {}).get('average') == protocol['basis']['metrics'][kind]['average']
                                  for spec in specs) if kind == 'classification' else True,
            'targets_and_output': all(spec.get('targets') and 'output' in spec and 'id_columns' in spec for spec in specs) and
                                  all(specs[0][key] == specs[1][key] for key in ('targets', 'output', 'id_columns')),
            'model': all((item.get('qwen_configuration') or {}).get('model') == protocol['basis']['model'] for item in meta),
        }
        for arm, directory in zip(('A', 'B'), (directory_a, directory_b)):
            # The committed event ledger exposes requested settings without reading
            # model payloads or opening any live SQLite database on macOS.
            from tools.diagnostic_3h import events
            requests = [event for event in events(directory) if event['kind'] == 'request_queued' and event.get('task_id') == kind]
            task_checks['generation_settings_' + arm] = bool(requests) and all(
                event.get('temperature') == protocol['basis']['temperature'] and
                event.get('effort') in protocol['basis']['reasoning_efforts'] for event in requests)
        winner_ids = [item['report'].get('control_winner') for item in results]
        scores = [(item['report'].get('control_scores') or {}).get(winner) for item, winner in zip(results, winner_ids)]
        observation = None
        if all(isinstance(score, (int, float)) and math.isfinite(score) for score in scores):
            observation = directed_gap(*scores, protocol['basis']['metrics'][kind])
        comparable = all(checks.values()) and all(task_checks.values()) and observation is not None
        tasks[kind] = {'checks': task_checks, 'comparable': comparable, 'control_winners': winner_ids,
                       'observed_scores': observation,
                       'rollback_required': observation['regression_requires_rollback'] if comparable else None}
    spent = [report['reconciliation'].get('interval_increase') for report in reports]
    return {'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            'checks': checks, 'tasks': tasks, 'comparison_valid': all(row['comparable'] for row in tasks.values()),
            'api_spend_by_series': spent,
            'measured_api_spend_total': sum(spent) if all(isinstance(x, (int, float)) for x in spent) else None,
            'scope': 'Previously viewed public controls; not independent hidden quality or official normalized score',
            'full_plan_complete': False}


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', type=Path, required=True)
    parser.add_argument('--b', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.a.resolve(), args.b.resolve(), args.protocol.resolve())
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'comparison_valid': result['comparison_valid'], 'api_spend_total': result['measured_api_spend_total']}))
