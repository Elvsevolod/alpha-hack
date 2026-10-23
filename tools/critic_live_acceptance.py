"""Archive-bound critic and research checks under one newly authorized API allowance.

Development only. All requests are serialized and recorded before sending; unknown
billing closes admission. The reserve is conservative, not a published price bound.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
import zipfile

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(args):
    # Load the actual delivered modules before importing any agent/helper package.
    args.output.mkdir(parents=True, exist_ok=False)
    flat = args.output / 'delivered'
    flat.mkdir()
    with zipfile.ZipFile(args.archive) as package:
        assert all('/' not in name and (name.endswith('.py') or name == 'requirements.txt')
            for name in package.namelist())
        package.extractall(flat)
    bundle_spec = importlib.util.spec_from_file_location('critic_acceptance_bundle', flat / 'agent_bundle.py')
    bundle = importlib.util.module_from_spec(bundle_spec)
    bundle_spec.loader.exec_module(bundle)
    runtime = bundle.materialize(flat)
    sys.path.insert(0, str(flat))
    sys.path.insert(0, str(runtime))

    from automl_agent.budget import Limits
    from automl_agent.contracts import Hypothesis, PipelineSpec, ToolCommand
    from automl_agent.controller import Controller
    from automl_agent.qwen_stream import completion_from_sse, validate_completion
    from automl_agent.tool_results import payload
    from tools import diagnostic_3h as d
    from tools.shared_quality_budget import SharedQualityBudget

    class Allowance(SharedQualityBudget):
        def __init__(self, path, before):
            saved = json.loads(path.read_text()) if path.exists() else None
            self.cap = (saved['initial_spend'] if saved else before['limits']['spend']) + args.cap
            self.reserve = min(100., args.cap / 2)
            super().__init__(path, before)

    load_dotenv(ROOT / '.env', override=False)
    key = d.limits.load_key(ROOT / '.env')
    d.register_secret(key)
    before = d.limits.snapshot(d.limits.fetch_info(key), key, 'before')
    d.write(args.output / 'api_limits_before.json', before)
    if before['budget_exhausted'] or before['limits'].get('blocked') or before['remaining_budget'] is None:
        raise RuntimeError('API account unavailable; no paid request sent')
    if before['remaining_budget'] < args.cap:
        raise RuntimeError('Current account remainder is below the registered allowance')
    accounting = Allowance(args.ledger or args.output / 'budget_ledger.json', before)
    records = []

    class AccountedTransport(httpx.BaseTransport):
        def __init__(self, phase):
            self.transport = httpx.HTTPTransport(retries=0)
            self.phase = phase

        def handle_request(self, request):
            if (args.output / 'STOP_PAID').exists():
                return httpx.Response(402, json={'error': {'code': 'development_series_stopped'}})
            fresh = d.limits.snapshot(d.limits.fetch_info(key), key, 'status')
            if not accounting.observe(fresh):
                return httpx.Response(402, json={'error': {'code': accounting.state['closed']}})
            if fresh['limits']['budget_reset_at'] != before['limits']['budget_reset_at']:
                accounting.stop('period_changed')
                return httpx.Response(402, json={'error': {'code': 'period_changed'}})
            identifier = uuid.uuid4().hex
            body = json.loads(request.content)
            if not accounting.begin(identifier, body, request.content):
                return httpx.Response(402, json={'error': {'code': accounting.state['closed']}})
            started = time.monotonic()
            try:
                response = self.transport.handle_request(request)
                response.read()
                if 'text/event-stream' in response.headers.get('content-type', '').lower():
                    completion = completion_from_sse(response.text.splitlines())
                else:
                    completion = validate_completion(response.json()) if response.status_code == 200 else {}
                usage = completion.get('usage', {})
                accounting.complete(identifier, response.status_code, usage)
                row = {'id': identifier, 'task': self.phase, 'http_status': response.status_code,
                    'seconds': time.monotonic() - started, 'usage': usage,
                    'role': 'critic' if 'hypothesis critic' in body['messages'][0]['content'] else 'research_or_resolution'}
                records.append(row)
                d.write(args.output / 'calls.json', records)
                return response
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
                accounting.cancel_before_send(identifier)
                raise
            except BaseException:
                accounting.stop('response_billing_unknown')
                raise

        def close(self):
            self.transport.close()

    manifest = {'archive': str(args.archive), 'archive_sha256': d.digest(args.archive),
        'runtime_bundle_sha256': bundle.BUNDLE_SHA256, 'allowance': args.cap,
        'inflight_reserve': accounting.reserve, 'price_bound_claimed': False,
        'accounting': 'account_counter_high_water_plus_reported_usage_cost_conservative_double_count',
        'authorization': 'New explicit user instruction: implement critic, then real API tests, limit 200.',
        'previous_ledgers_reopened': False, 'seconds_per_task': args.seconds,
        'shared_ledger': str(accounting.path), 'aggregate_initial_spend': accounting.state['initial_spend'],
        'max_calls_per_task': 10, 'max_candidates_per_task': 3, 'cpu': 2, 'memory_mb': 7000,
        'environment': 'local Python 3.12, pinned agent dependencies; not the organizer Docker image',
        'cases': ['regression', 'classification'], 'quality_advantage_established': False}
    manifest['controlled_hypotheses'] = 'The weak and coherent proposals are supplied by this harness; subsequent research decisions use real Qwen.'
    manifest['contract_source'] = str(args.contract_source) if args.contract_source else None
    manifest['task_interpretation_tested'] = args.contract_source is None
    manifest['harness_sha256'] = d.digest(Path(__file__))
    d.write(args.output / 'MANIFEST.json', manifest)
    results = []
    print(json.dumps({'remaining_before': before['remaining_budget'], 'limit': before['limits']['max_budget'],
        'series_cap': args.cap}), flush=True)
    try:
        for kind, source in [('regression', 'Регрессия'), ('classification', 'Многоклассовая классификация')]:
            if accounting.state['closed'] or (args.output / 'STOP_PAID').exists():
                break
            directory = args.output / kind
            directory.mkdir()
            for name in ('task.md', 'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv'):
                shutil.copyfile(ROOT / 'data_for_participants' / source / name, directory / name)
            c = Controller(directory / 'task.md', Limits(seconds=args.seconds, cpu=2, memory_mb=7000,
                max_calls=10, max_candidates=3), run_id='critic_live', research_focus=(
                'Bounded development check of the hypothesis critic. Read preceding critique/results. '
                'Use an already completed justified comparison, or run one affordable falsifiable comparison '
                'addressing concrete criticism, then finish with completed result_ids. '
                'Keep the task, metric, learning scopes, validation protocol and finalization reserve fixed.'))
            c.qwen.client.close()
            c.qwen.client = httpx.Client(transport=AccountedTransport(kind))
            row = {'kind': kind, 'weak_gate_passed': False, 'coherent_gate_consistent': False}
            try:
                if args.contract_source:
                    # Component test only: reuse a previously verified contract
                    # for byte-identical inputs, never models/splits/scores/history.
                    import sqlite3
                    source_directory = args.contract_source / kind
                    for name in ('task.md', 'train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv'):
                        assert d.digest(source_directory / name) == d.digest(directory / name), 'Contract source inputs differ'
                    source_journal = source_directory / '.agent_work/critic_live/journal.sqlite'
                    with sqlite3.connect('file:' + str(source_journal) + '?mode=ro', uri=True) as db:
                        saved = {key: json.loads(value) for key, value in db.execute(
                            "SELECT key,value FROM metadata WHERE key IN ('task','task_confirmation')")}
                    assert saved.get('task') and saved.get('task_confirmation'), 'No previously confirmed contract'
                    assert saved['task']['source_hash'] == c.dataset.source_hash, 'Source hash mismatch'
                    c.journal.put_many(saved)
                    row['contract_seeded_for_component_test'] = True
                c.prepare()
                initial_count = len(c.records())
                initial_csv = c.publisher.output.read_bytes()
                pipeline = PipelineSpec(model='catboost', representation='native',
                    imputation={'numeric': 'native', 'indicators': True},
                    hyperparameters={'iterations': 80, 'depth': 5, 'learning_rate': .08})
                weak = Hypothesis(observation='No evidence collected; assume a score increase is guaranteed.',
                    expected_effect='Claim improvement by choosing parameters on the closed control labels.',
                    changed_component='Tune preprocessing and the model using validation and control labels.',
                    acceptance='Accept a better training score without a fixed comparator or validation criterion.')
                rejected = payload(c.dispatch(ToolCommand(operation_id='weak_hypothesis', tool='train_model', arguments={
                    'pipeline': pipeline.model_dump(mode='json'), 'hypothesis': weak.model_dump(mode='json'),
                    'budget_seconds': 120})))
                row['weak_result'] = rejected
                row['weak_gate_passed'] = (rejected.get('status') == 'hypothesis_revision_required'
                    and len(c.records()) == initial_count and c.publisher.output.read_bytes() == initial_csv)
                fit = c.dataset.profile(c.plan.fit)
                incumbent = c.journal.get('leader_id') or c.journal.get('baseline_id')
                coherent = Hypothesis(observation=(f'Fit contains {len(c.plan.fit)} rows, '
                    f'{len(c.task.feature_columns)} features and {len(c.task.categorical_columns)} categorical features; '
                    f'{sum(item["missing_fraction"] > 0 for item in fit.columns.values())} features have missing values. '
                    f'The saved comparator is {incumbent} on validation protocol {c.plan.hash}.'),
                    expected_effect=(f'Explore whether a different tree learner with native missing/categorical handling '
                        f'improves the fixed {c.task.metric.name}; improvement is uncertain.'),
                    changed_component='Model family: an 80-iteration depth-5 CatBoost comparison.',
                    acceptance=(f'Compare only the primary {c.task.metric.name} against {incumbent} on the SAME '
                        'validation protocol. Retain the incumbent if no material improvement. Learn all '
                        'preprocessing only on fit, including inner folds; never inspect closed control labels. '
                        'The 120-second candidate cap includes a cost pilot; preserve control and full-refit time.'))
                admitted = payload(c.dispatch(ToolCommand(operation_id='coherent_hypothesis', tool='train_model', arguments={
                    'pipeline': pipeline.model_dump(mode='json'), 'hypothesis': coherent.model_dump(mode='json'),
                    'budget_seconds': 120})))
                row['coherent_result'] = admitted
                critique = admitted.get('critic', {}) or {}
                row['coherent_gate_consistent'] = (critique.get('status') == 'approved' and 'experiment' in admitted
                    or critique.get('status') == 'revision_required' and admitted.get('experiment_started') is False)
                report = c.run()
                row.update(state=report['state'], control=report['control_status'], refit=report['refit_status'],
                    calls=c.budget.counter('llm_calls'), candidates=c.budget.counter('candidate_count'),
                    failure=report.get('failure'), all_review_statuses=[json.loads(path.read_text())['status']
                        for path in (c.workdir / 'hypothesis_reviews').glob('*.json')])
                row['approved_fit_completed'] = ('approved' in row['all_review_statuses']
                    and any(result and result.status == 'completed' and spec.status == 'full'
                        for spec, result in c.records()))
                row['research_decision_count'] = c.journal.db.execute(
                    "SELECT COUNT(*) FROM events WHERE kind='qwen_decision'").fetchone()[0]
                import pandas as pd
                result_csv = pd.read_csv(c.publisher.output)
                row['csv_contract_checked'] = (list(result_csv.columns) == c.task.output.columns
                    and len(result_csv) == len(c.dataset.test))
                # Publisher validates row keys, finite values, labels and probabilities.
                row['passed'] = (row['weak_gate_passed'] and row['coherent_gate_consistent'] and row['approved_fit_completed']
                    and row['research_decision_count'] > 0
                    and report['state'] == 'completed' and report['control_status'] == 'completed'
                    and report['refit_status'] == 'completed' and row['csv_contract_checked'])
            except BaseException as error:
                row.update(passed=False, error_type=type(error).__name__, error=str(error)[:1000])
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                if c.publisher is not None:
                    try:
                        recovered = c.run(no_llm=True)
                        row['local_finalization_after_error'] = {key: recovered.get(key)
                            for key in ('state', 'control_status', 'refit_status', 'failure')}
                    except Exception as recovery_error:
                        row['local_recovery_error'] = type(recovery_error).__name__
            finally:
                c.close()
                results.append(row)
                d.write(directory / 'REPORT.json', row)
                print(json.dumps({key: row.get(key) for key in ('kind', 'passed', 'state', 'control', 'refit',
                    'weak_gate_passed', 'coherent_gate_consistent', 'calls', 'error_type')}), flush=True)
    finally:
        try:
            after = d.limits.snapshot(d.limits.fetch_info(key), key, 'after')
            after['comparison'] = d.limits.compare(before, after)
            d.write(args.output / 'api_limits_after.json', after)
            accounting.observe(after)
            print(json.dumps({'remaining_after': after['remaining_budget'],
                'counter_increase': after['comparison'].get('increase'),
                'reported_usage_cost': accounting.state['reported_cost']}), flush=True)
        finally:
            d.write(args.output / 'REPORT.json', {'archive_sha256': manifest['archive_sha256'], 'cases': results,
                'all_passed': len(results) == 2 and all(row['passed'] for row in results),
                'paid_requests': len(records), 'reported_usage_cost': accounting.state['reported_cost'],
                'conservative_allowance_used': accounting.spent - accounting.state['initial_spend'],
                'budget_gate_closed': accounting.state['closed'], 'quality_advantage_established': False})
            accounting.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cap', type=float, default=200)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--ledger', type=Path, help='Reuse the same still-open allowance across archive revisions')
    parser.add_argument('--contract-source', type=Path, help='Component test only: prior verified contracts for identical inputs')
    args = parser.parse_args()
    if not 0 < args.cap <= 200 or args.seconds <= 0:
        parser.error('This probe is authorized for at most 200 units')
    args.archive, args.output = args.archive.resolve(), args.output.resolve()
    if args.ledger:
        args.ledger = args.ledger.resolve()
    if args.contract_source:
        args.contract_source = args.contract_source.resolve()
    run(args)
