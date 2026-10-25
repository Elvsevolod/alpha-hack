"""Offline audit first; opening sealed labels freezes all compared revisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_agent.util import atomic_json, file_hash


def audit_run(directory):
    journal = directory / 'task/.agent_work/research_memory_ab/journal.sqlite'
    if not journal.exists():
        return {'status': 'journal_missing'}
    with sqlite3.connect(journal.resolve().as_uri() + '?mode=ro', uri=True) as db:
        allowed = {'state', 'llm_calls', 'candidate_count', 'first_csv_seconds',
            'first_trained_csv_seconds', 'control_status', 'refit_status', 'failure',
            'token_usage', 'llm_prompt_contract', 'finalization_plan', 'publication'}
        metadata = {key: json.loads(value) for key, value in db.execute('SELECT key,value FROM metadata')
            if key in allowed}
        prompt = metadata.get('llm_prompt_contract') or {}
        metadata['llm_prompt_contract'] = {key: prompt.get(key) for key in
            ('version', 'system_hash', 'guides_hash', 'sections')}
        decisions = []
        for seq, kind, raw in db.execute('SELECT seq,kind,payload FROM events ORDER BY seq'):
            if kind == 'qwen_decision':
                event = json.loads(raw)
                command = event.get('command') or {}
                decisions.append({'sequence': seq, 'action': command.get('action'),
                    'tool': command.get('tool'), 'operation_id': command.get('operation_id'),
                    'arguments': {key: value for key, value in command.get('arguments', {}).items()
                        if key not in {'code', 'source_code'}},
                    'interpretation': event.get('interpretation', '')[:4000]})
        experiments = []
        for identifier, raw_spec, raw_result in db.execute('SELECT id,spec,result FROM experiments'):
            spec = json.loads(raw_spec)
            result = json.loads(raw_result) if raw_result else {}
            experiments.append({'id': identifier, 'parent_id': spec.get('parent_id'),
                'hypothesis': spec.get('hypothesis'), 'pipeline': spec.get('pipeline'),
                'protocol': spec.get('validation_hash'), 'status': result.get('status', 'pending'),
                'score': result.get('score'), 'scope': result.get('scope'), 'error': result.get('error'),
                'seconds': result.get('seconds')})
        operations = []
        for raw, in db.execute('SELECT record FROM operations'):
            row = json.loads(raw)
            item = {key: row.get(key) for key in
                ('operation_id', 'tool', 'status', 'error', 'started_at', 'ended_at')}
            result = row.get('result') or {}
            item['analysis_status'] = result.get('status')
            body = result.get('data') or {}
            if isinstance(body, dict):
                item['error'] = body.get('error')
                item['error_code'] = body.get('error_code')
            operations.append(item)
    csv = directory / 'task/submission.csv'
    return {'metadata': metadata, 'decisions': decisions, 'operations': operations,
        'experiments': experiments, 'csv_sha256': file_hash(csv) if csv.exists() else None,
        'outer_labels_opened': False}


def audit(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    result = {row['id']: audit_run(output / 'runs' / row['id']) for row in protocol['runs']}
    atomic_json(output / 'technical_audit.json', result)
    print(json.dumps({key: {'experiments': len(value.get('experiments', [])),
        'decisions': len(value.get('decisions', [])), 'csv_present': bool(value.get('csv_sha256'))}
        for key, value in result.items()}))
    return result


def evaluate(output):
    import numpy as np
    import pandas as pd
    from automl_agent.contracts import TaskSpec
    from automl_agent.evaluation import score_predictions

    protocol = json.loads((output / 'protocol.json').read_text())
    if any(row['state'] == 'registered' for row in protocol['runs']):
        raise ValueError('Finish or explicitly terminate every registered variant before opening labels')
    audit_path = output / 'technical_audit.json'
    if not audit_path.exists():
        raise ValueError('Audit technical traces before opening external labels')
    protocol.setdefault('evaluation_frozen_at', time.time())
    protocol['technical_audit_sha256'] = file_hash(audit_path)
    # Durable freeze precedes the first label read. No agent run may follow it.
    atomic_json(output / 'protocol.json', protocol)
    scores = []
    sealed = Path(protocol['labels_source_outside_tasks'])
    for row in protocol['runs']:
        result = {'id': row['id'], 'task': row['task'], 'arm': row['arm'],
            'commit': protocol['commits'][row['arm']], 'external_score': None,
            'gateway_increase': row.get('gateway_increase'), 'wall_seconds': row.get('wall_seconds')}
        try:
            directory = output / 'runs' / row['id'] / 'task'
            csv = directory / 'submission.csv'
            reference = json.loads((sealed / row['task'] / 'reference_contract.json').read_text())
            task = TaskSpec.model_validate({**reference, 'source_hash': 'sealed_external_evaluator'})
            dtype = {key: str for key in task.id_columns}
            prediction = pd.read_csv(csv, dtype=dtype)
            test = pd.read_csv(directory / 'test.csv', dtype=dtype)
            actual = pd.read_csv(sealed / row['task'] / 'labels.csv', dtype=dtype)
            if list(prediction) != task.output.columns or len(prediction) != len(test):
                raise ValueError('Published CSV schema or row count differs')
            ids = task.id_columns
            if not prediction[ids].equals(test[ids]) or prediction[ids].duplicated().any():
                raise ValueError('Published identifiers or ordering differ')
            if actual[ids].duplicated().any():
                raise ValueError('Sealed labels contain duplicate identifiers')
            actual = test[ids].merge(actual, on=ids, how='left', validate='one_to_one', sort=False)
            if actual[task.targets].isna().any().any():
                raise ValueError('Missing external labels')
            values = {target: prediction[target].to_numpy() for target in task.targets}
            if task.kind == 'classification':
                if any(not set(values[t]) <= set(task.classes[t]) for t in task.targets):
                    raise ValueError('Undeclared output class')
            elif any(not np.isfinite(values[t].astype(float)).all() for t in task.targets):
                raise ValueError('Nonfinite published prediction')
            result.update(status='scored', metric=task.metric.model_dump(mode='json'),
                external_score=score_predictions(task, actual, values, task.output.prediction_type),
                csv_sha256=file_hash(csv))
        except (ValueError, OSError, KeyError) as error:
            result.update(status='not_scored', reason=type(error).__name__ + ': ' + str(error))
        scores.append(result)
    atomic_json(output / 'external_evaluation.json', scores)
    print(json.dumps(scores, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'evaluate'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (audit if args.action == 'audit' else evaluate)(args.output.resolve())
