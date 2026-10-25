"""Regressions for measured ML-trace defects; all data and providers are offline."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import PipelineSpec, TaskSpec
from automl_agent.decision import decide, fit_decision
from automl_agent.finalists import select_finalists
from automl_agent.supervised_features import SupervisedFeatures
from automl_agent.util import canonical, digest


def temporal_data(periods=3, grouped=False, horizon=1):
    X = pd.DataFrame({'period': np.repeat(np.arange(periods), 12),
        'category': ['a', 'b', None] * (periods * 4),
        'signal': np.linspace(-1, 1, periods * 12),
        'entity': np.repeat(np.arange(periods), 12)}, index=[f'r{i}' for i in range(periods * 12)])
    y = pd.DataFrame({'response': 10 + 2 * np.arange(len(X)) % 13}, index=X.index)
    task = TaskSpec(kind='forecasting' if horizon > 1 else 'regression', targets=['response'],
        feature_columns=list(X), categorical_columns=['category'], time_column='period',
        group_column='entity' if grouped else None, horizon=horizon,
        metric={'name': 'rmse'}, output={'columns': ['response'], 'prediction_type': 'values'}, source_hash='fixture')
    spec = PipelineSpec(model='ridge', transformations=[{'name': 'target_encode',
        'parameters': {'columns': ['category'], 'smoothing': 20}}])
    return X, y, task, spec


def encode(X, y, task, spec, strategy='temporal'):
    model = SupervisedFeatures(spec, task, strategy)
    result = model.fit_transform(X, y, X)
    return result['__te_category_response'].to_numpy(), model


@pytest.mark.parametrize('periods', [1, 2, 3])
def test_M1_short_history_and_own_future_labels(periods):
    X, y, task, spec = temporal_data(periods)
    values, model = encode(X, y, task, spec)
    assert np.isfinite(values).all() and np.all(values[:12] == 0)
    assert model.report[0]['coverage'] == (periods - 1) / periods
    for period in range(periods):
        changed = y.copy()
        changed.loc[X.period >= period, 'response'] += 987654
        alternative, _ = encode(X, changed, task, spec)
        np.testing.assert_array_equal(values[X.period <= period], alternative[X.period <= period])
    # Unknown categories use a fit-only prior in the saved transform.
    prediction = model.transform(X.iloc[-2:].assign(category='never_seen'), X.iloc[-2:])
    assert np.isfinite(prediction['__te_category_response']).all()


@pytest.mark.parametrize('strategy,horizon', [('temporal_group', 1), ('temporal', 2), ('temporal_group', 2)])
def test_M1_uncovered_group_and_horizon_are_label_free(strategy, horizon):
    X, y, task, spec = temporal_data(5, grouped='group' in strategy, horizon=horizon)
    # No permitted training group for the first window: its whole held set
    # must be excluded, rather than only the group of each individual row.
    X.loc[X.period < 2, 'entity'] = np.tile([0, 1], 12)
    values, model = encode(X, y, task, spec, strategy)
    from automl_agent.research_evaluation import folds
    parts = folds(X, y, task, strategy, min_folds=0)
    covered = np.zeros(len(X), bool)
    for _, b in parts:
        covered[b] = True
    assert np.all(values[~covered] == 0)
    assert model.report[0]['uncovered_prior'] == 'neutral'
    if strategy == 'temporal_group':
        for a, b in parts:
            held = set(X.iloc[b].entity)
            changed = y.copy()
            changed.loc[X.entity.isin(held), 'response'] += 123456
            altered, _ = encode(X, changed, task, spec, strategy)
            np.testing.assert_array_equal(values[b], altered[b])


def test_M1_nested_smearing_frozen_refit_and_strict_minimum(tmp_path):
    from automl_agent.pipelines import DirectPipeline
    from automl_agent.research_evaluation import folds
    X, y, task, spec = temporal_data(3)
    spec = spec.model_copy(update={'target_transform': 'log1p', 'log_inverse': 'oof_smearing'})
    context = {'strategy': 'temporal', 'protocol_hash': 'fixed', 'cpu': 1, 'memory_mb': 1024,
        'budget_seconds': 30, 'workdir': str(tmp_path), 'split_id': 'fit', 'task': task.model_dump(mode='json')}
    model = DirectPipeline(spec, task)
    model.fit(X, y, context)
    frozen = model.frozen_spec()
    assert len(frozen.learned_inverse['response']['fold_factors']) >= 2
    refit = DirectPipeline(frozen, task)
    refit.fit(X, y, {**context, 'execution_mode': 'refit'})
    assert refit.inverse_state == model.inverse_state
    for strategy in ('temporal', 'group', 'exact_group', 'random'):
        with pytest.raises(ValueError):
            folds(X.iloc[:1], y.iloc[:1], task.model_copy(update={'group_column': 'entity'}), strategy)
    with pytest.raises(ValueError, match='(two|2|insufficient)'):
        DirectPipeline(spec, task).fit(X.iloc[:24], y.iloc[:24], context)


def test_M1_uncovered_long_history_strict_past_prior():
    X, y, task, spec = temporal_data(8)
    values, model = encode(X, y, task, spec)
    assert model.report[0]['uncovered_rows'] == 60
    for period in range(5):
        expected = y.loc[X.period < period, 'response'].mean() if period else 0.
        np.testing.assert_allclose(values[X.period == period], expected)
    changed = y.copy(); changed.loc[X.period >= 3, 'response'] = 987654.
    altered, _ = encode(X, changed, task, spec)
    np.testing.assert_array_equal(values[X.period <= 3], altered[X.period <= 3])


@pytest.mark.parametrize('strategy', ['weighted_oof', 'stacked_oof', 'final_decision'])
def test_M1_one_fold_is_not_enough_for_learned_composition(strategy):
    from automl_agent.learned_blend import fit_blend, fit_final_layer
    task, y, p = class_data(7)
    report = {'folds': [{'fold': 0}], 'folds_hash': 'one', 'row_keys': list(map(str, y.index)),
        'classes': task.classes, 'values': {'response': p.tolist()}}
    with pytest.raises(ValueError, match='insufficient_oof_support'):
        if strategy == 'final_decision':
            fit_final_layer(task, [report, report], y, PipelineSpec(engine='ensemble', members=['a', 'b']))
        else:
            fit_blend(task, [report, report], y, strategy, 42)


@pytest.mark.parametrize('strategy', ['random', 'group', 'exact_group'])
def test_M1_negative_controls_keep_fold_exclusions(strategy):
    from automl_agent.research_evaluation import folds
    X, y, task, spec = temporal_data(6, grouped=True)
    X.entity = np.arange(len(X)) % 20
    if strategy == 'exact_group':
        X.iloc[1] = X.iloc[0]
    values, _ = encode(X, y, task, spec, strategy)
    for a, b in folds(X, y, task, strategy):
        changed = y.copy(); changed.iloc[b, 0] += 987654.
        altered, _ = encode(X, changed, task, spec, strategy)
        np.testing.assert_array_equal(values[b], altered[b])
        if strategy == 'group':
            assert not set(X.iloc[a].entity) & set(X.iloc[b].entity)


def class_data(seed, k=4, n=500):
    rng = np.random.default_rng(seed)
    classes = [f'c{i}' for i in range(k)]
    logits = rng.normal(size=(n, k))
    truth = np.exp(logits - logits.max(1, keepdims=True)); truth /= truth.sum(1, keepdims=True)
    y = pd.DataFrame({'response': np.asarray(classes)[[rng.choice(k, p=row) for row in truth]]})
    biased = logits + np.resize([-.3, .2, .4, -.4], k)
    p = np.exp(biased - biased.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    task = TaskSpec(kind='classification', targets=['response'], feature_columns=['x'],
        classes={'response': classes}, metric={'name': 'f1', 'average': 'macro'},
        output={'columns': ['response'], 'prediction_type': 'labels'}, source_hash='fixture')
    return task, y, p


@pytest.mark.parametrize('seed', range(8))
def test_M4_canonical_permutations(seed, canonical_decision):
    fit_decision, decide = canonical_decision.fit_decision, canonical_decision.decide
    task, y, p = class_data(seed)
    original = fit_decision(task, y, p, range(len(y)))
    expected = decide(p, task.classes['response'], original)
    for permutation in ([3, 0, 1, 2], [2, 1, 0, 3]):
        classes = [task.classes['response'][i] for i in permutation]
        altered = task.model_copy(update={'classes': {'response': classes}})
        state = fit_decision(altered, y, p[:, permutation], range(len(y)))
        np.testing.assert_array_equal(decide(p[:, permutation], classes, state), expected)
        assert state['training_metric'] == pytest.approx(original['training_metric'], abs=1e-15)
        assert state['classes'] == classes and max(map(abs, state['offsets'])) <= 1


@pytest.mark.parametrize('maximize', [True, False])
def test_M3_partial_window_hash_protocol_and_no_pareto_exclusion(maximize):
    from automl_agent.contracts import SelectionPolicy
    sign = 1 if maximize else -1
    rows = [{'id': name, 'identity': name, 'score': sign * score, 'windows': windows,
        'windows_hash': window_hash, 'protocol': 'fixed', 'window_status': 'limited', 'cost': cost,
        'created_index': i} for i, (name, score, window, window_hash, cost) in enumerate([
            ('baseline', 100, 100, 'same', 1), ('leader', 110, 101, 'same', 1),
            ('validation', 109, 10, 'same', 1), ('window', 90, 99, 'same', 2),
            ('wrong_hash', 120, 1000, 'other', 1)])
        for windows in [{'late': sign * window}]]
    selected, audit = select_finalists(rows, 'baseline', 'leader', SelectionPolicy(maximize=maximize), ['early', 'late'])
    assert selected == ['baseline', 'leader', 'window']
    assert audit['mode'] == 'partial_common_window' and audit['windows_hash'] == 'same'
    assert audit['completeness']['baseline'] == 'limited'
    assert 'wrong_hash' not in [row['id'] for row in audit['pool']]


@pytest.mark.parametrize('common', [0, 1, 2, 3])
def test_M3_window_counts_ties_identity_and_limit(common):
    from automl_agent.contracts import SelectionPolicy
    windows = [f'w{i}' for i in range(common)]
    rows = [{'id': name, 'identity': identity, 'score': score, 'windows': {w: value for w in windows},
        'windows_hash': 'same', 'protocol': 'fixed', 'window_status': 'limited' if common < 3 else 'complete',
        'cost': cost, 'created_index': i} for i, (name, identity, score, value, cost) in enumerate([
            ('baseline', 'base', 100, 100, 1), ('leader', 'lead', 90, 90, 1),
            ('alt', 'alt', 99, 80, 1), ('duplicate', 'alt', 99, 80, .1), ('other', 'other', 99, 80, 2)])]
    selected, audit = select_finalists(rows, 'baseline', 'leader', SelectionPolicy(maximize=False), windows)
    assert selected == ['baseline', 'leader', 'duplicate' if common == 0 else 'alt']
    assert audit['mode'] == ('limited_validation' if common == 0 else 'partial_common_window' if common == 1 else 'common_temporal_windows')
    assert select_finalists(rows, 'baseline', 'leader', SelectionPolicy(maximize=False), windows, limit=2)[0] == ['baseline', 'leader']
    for row in rows[2:]:
        row['control_score'] = 987654  # Unread pre-freeze data cannot change selection.
    assert select_finalists(rows, 'baseline', 'leader', SelectionPolicy(maximize=False), windows)[0] == selected


def test_M8_effective_diff_ignores_learned_state_and_normalizes_aliases():
    from automl_agent.research import effective_change
    from automl_agent.contracts import Hypothesis
    parent = PipelineSpec(model='lightgbm', representation='native', early_stopping_rounds=100,
        hyperparameters={'num_iterations': 200})
    child = PipelineSpec(model='lightgbm', representation='tabular', early_stopping_rounds=None,
        hyperparameters={'n_estimators': 200}, learned_inverse={'response': {'factor': 1.2}})
    hypothesis = Hypothesis(observation='Missing values', expected_effect='Change imputation',
        changed_component='imputation', acceptance='Fixed validation')
    report = effective_change(child, parent, hypothesis)
    assert report['effective_fields'] == ['early_stopping_rounds', 'representation']
    assert report['claimed_field_unchanged'] and report['mixed_change']
    assert report['imputation']['indicators'] == parent.imputation.indicators


@pytest.mark.parametrize('outcome', ['error', 'import_error', 'memory_limit', 'insufficient_pilot_support', 'completed', 'request_only', 'corrupt', 'wrong_request', 'wrong_rows', 'wrong_artifact', 'wrong_request_hash'])
def test_M6_cached_outcomes_never_refit(tmp_path, outcome):
    from automl_agent.learned_blend import member_oof, oof_directory, OOFCacheError
    from automl_agent.util import file_hash
    X, y, task, pipeline = temporal_data()
    spec = SimpleNamespace(pipeline=pipeline, library_versions={'numpy': 'fixture'})
    calls = []
    c = SimpleNamespace(workdir=tmp_path, task=task, plan=SimpleNamespace(hash='fixed', fit=list(range(len(X)))),
        dataset=SimpleNamespace(source_hash='data', row_keys=X.index.tolist()), lookup=lambda identifier: (spec, None),
        stage=lambda *a, **k: calls.append('stage'), executor=SimpleNamespace(run=lambda *a: calls.append('fit')))
    directory = oof_directory(c, spec, c.plan.fit)
    directory.mkdir(parents=True)
    request = {'mode': 'oof', 'task': task.model_dump(mode='json'), 'pipeline': pipeline.model_dump(mode='json'),
        'context': {'data_hash': 'data', 'protocol_hash': 'fixed', 'oof_seed': 42, 'library_versions': spec.library_versions}}
    if outcome == 'wrong_request':
        request['context']['protocol_hash'] = 'foreign'
    (directory / 'request.json').write_text(canonical(request))
    for name in ('fit_X.json', 'fit_y.json'):
        (directory / name).write_text(canonical({'index': ['foreign'] if outcome == 'wrong_rows' else X.index.tolist()}))
    manifest = {'status': 'error', 'diagnostic_code': 'error', 'error': 'ValueError: Insufficient inner history',
        'diagnostic_artifact': str(directory / ('foreign.json' if outcome == 'wrong_artifact' else 'worker_result.json'))}
    known = outcome in {'error', 'import_error', 'memory_limit', 'insufficient_pilot_support'}
    if known:
        manifest.update(status=outcome, diagnostic_code=outcome)
    manifest['request_hash'] = 'foreign' if outcome == 'wrong_request_hash' else digest(request)
    if outcome == 'completed':
        (directory / 'oof_predictions.json').write_text(canonical({'values': {'response': [1, 2]}, 'folds': [1, 2]}))
        manifest = {'status': 'completed', 'hashes': {'oof_predictions.json': file_hash(directory / 'oof_predictions.json')}}
    if outcome != 'request_only':
        (directory / 'worker_result.json').write_text('{' if outcome == 'corrupt' else canonical(manifest))
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    for _ in range(2):
        if outcome == 'completed':
            assert member_oof(c, 'member', 5)['values']['response'] == [1, 2]
        else:
            with pytest.raises(OOFCacheError) as caught:
                member_oof(c, 'member', 5)
            detail = caught.value.detail
            assert detail['status'] == ('technical_failure' if known else 'outcome_unknown')
            assert detail['identity'] == directory.name and detail['rows_hash'] == digest(c.plan.fit)
            if known:
                assert detail['error_code'] == 'oof_known_terminal_error' and detail['terminal_status'] == outcome
                assert detail['diagnostic_ref'].endswith('worker_result.json')
    assert calls == []
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_M2_summary_priorities_utf8_classes_and_measured_feedback():
    from automl_agent.context_memory import experiment_summary
    from context_compressor import Source
    from automl_agent.contracts import ExperimentSpec, Hypothesis
    task, y, p = class_data(4)
    task = task.model_copy(update={'classes': {'response': ['длинное имя ' * 400 + str(i) for i in range(4)]}})
    pipeline = PipelineSpec(model='logistic', hyperparameters={'очень длинный параметр ' * 70: 1})
    spec = ExperimentSpec(experiment_id='member', task_hash=task.hash, data_version='data', validation_hash='fixed',
        pipeline=pipeline, budget_seconds=10, hypothesis=Hypothesis(observation='fixture', expected_effect='fixture',
            changed_component='imputation', acceptance='fixture'))
    stability = {'status': 'limited', 'scope': 'fit_only_stress_windows', 'windows_hash': 'a' * 64,
        'scores': {'expanding_2': 56.1009}, 'diagnostics': {'expanding_2': {'residual_bias': {'response': 16.5136}}}}
    feedback = {'classes': {'scope': 'external_validation', 'classes': [{'class': label, 'status': 'available',
        'f1': i / 10, 'support': i + 5, 'uncertain': True} for i, label in enumerate(task.classes['response'])]},
        'private': {'status': 'suppressed', 'content': 'CONTROL_SENTINEL'}}
    result = {'training_summary': {'temporal_stability': stability}, 'feedback': feedback, 'error': 'ошибка ' * 300}
    source = Source('experiment:fixture', 'version', 'permitted', 'registered_feedback')
    change = {'basis': 'normalized_parent_recipe', 'declared_component': 'imputation',
        'effective_fields': ['representation', 'early_stopping_rounds'], 'claimed_field_unchanged': True, 'mixed_change': True}
    summary = experiment_summary(SimpleNamespace(task=task), spec, result, source, change)
    assert len(canonical(summary).encode()) <= 1536
    assert summary['temporal_stability']['last_residual_bias'] == 16.5136
    assert [row['index'] for row in summary['class_feedback']['weakest']] == [0, 1, 2]
    assert all('support' in row and row['uncertain'] for row in summary['class_feedback']['weakest'])
    assert summary['effective_change']['claimed_field_unchanged']
    assert 'CONTROL_SENTINEL' not in canonical(summary)
    assert summary == experiment_summary(SimpleNamespace(task=task), spec, result, source, change)


@pytest.mark.parametrize('method', ['legacy', 'regularized'])
@pytest.mark.parametrize('k', [2, 3, 10])
def test_M4_sparse_classes_and_zero_objective(method, k, canonical_decision):
    fit_decision = canonical_decision.fit_decision
    from automl_agent.evaluation import score_predictions
    task, y, p = class_data(11, k=k, n=100)
    if k > 2:
        y.loc[y.response == f'c{k-1}', 'response'] = 'c0'
    state = fit_decision(task, y, p, range(len(y)), method=method)
    baseline = score_predictions(task, y, {'response': decide(p, task.classes['response'])}, 'labels')
    objective = state['training_metric'] - np.sum(np.asarray(state['penalty']) * np.asarray(state['offsets']) ** 2)
    assert objective >= baseline - 1e-12 and np.isfinite(state['offsets']).all()
    if method == 'regularized':
        assert state['permutation_guarantee'] == 'distinct_fit_keys_and_no_final_argmax_ties'


@pytest.fixture
def canonical_decision():
    """Rejected M4 hypothesis remains independently reproducible outside runtime."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'outputs/ml_M1_M8_v096_20261025/m4_candidate/decision.py'
    spec = importlib.util.spec_from_file_location('automl_agent._m4_candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ml_controller(tmp_path, monkeypatch):
    from automl_agent.controller import Controller
    from automl_agent.budget import Limits
    from conftest import write_task
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture-offline-key')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://fixture.invalid/v1')
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1), run_id='ml_fixes')
    c.prepare()
    yield c
    c.close()


@pytest.mark.parametrize('mode', ['native', 'json'])
def test_M2_M5_snapshot_routes_and_all_compaction_stages(ml_controller, mode):
    from automl_agent.controller import SYSTEM
    c = ml_controller
    spec, result = c.lookup(c.journal.get('baseline_id'))
    stability = {'status': 'limited', 'scope': 'fit_only_stress_windows', 'windows_hash': 'a' * 64,
        'scores': {'expanding_2': 56.1009}, 'diagnostics': {'expanding_2': {'residual_bias': {'answer': 16.5136}}}}
    c.journal.save_experiment(spec, result.model_copy(update={'training_summary': {'temporal_stability': stability}}))
    snapshot = c.context_memory.snapshot(SYSTEM)
    details = [json.loads(value) for value in snapshot.details]
    summary = next(item['summary'] for item in details if item['record_id'] == spec.experiment_id)
    assert summary['temporal_stability']['last_residual_bias'] == 16.5136
    for stage in range(4):
        prepared = c.context_memory.compressor.fit(snapshot, mode, budget=262144, minimum_stage=stage)
        body = json.loads(prepared.wire_json)
        context = json.loads(body['messages'][1]['content'])
        assert context['compaction']['stage'] == stage
        assert context['core']['reading_routes'] and context['core']['current_work']['leader_pipeline']
        instructions = body['messages'][0]['content']
        assert context['core']['current_work']['local_guides']['tool'] == 'read_run_records'
        policy = c.prompt_contract['guides']['research_policy']
        assert 'source_recipe_hash' in policy and 'Starter search ranges' in policy
        assert 'view=pipeline, include_code=false' in policy
        assert 'Copy identities' in instructions
        if stage == 0:
            assert '16.5136' in prepared.wire_json
        assert 'CONTROL_SENTINEL' not in prepared.wire_json


@pytest.mark.parametrize('scope,protocol,environment,status,expected', [
    ('full', 'same', 'same', 'completed', True), ('pilot', 'same', 'same', 'completed', False),
    ('full', 'foreign', 'same', 'completed', False), ('full', 'same', 'foreign', 'completed', False),
    ('full', 'same', 'same', 'error', False), ('unknown', 'same', 'same', 'completed', False)])
def test_M2_baseline_full_comparison_roles(tmp_path, scope, protocol, environment, status, expected):
    from context_compressor import ContextStore, Record, Source
    from context_compressor.core import _select_records
    store = ContextStore(tmp_path / 'context.sqlite')
    for identifier, evaluation_scope, parent in [('baseline', 'baseline', None), ('child', scope, 'baseline')]:
        source = Source(identifier, 'version', '{}', 'registered_feedback')
        store.put_source(source)
        store.publish_records([Record(identifier, identifier, 'version', 'registered_feedback',
            status='completed' if not parent else status, protocol='same' if not parent else protocol,
            environment='same' if not parent else environment, metric='f1', maximize=True,
            score=.5 if not parent else .6, parent_id=parent, evaluation_scope=evaluation_scope)])
    maximum = store.db.execute('SELECT max(seq) FROM cc_records').fetchone()[0]
    cards = _select_records(store, {'baseline_id': 'baseline', 'leader_id': 'child'}, None, maximum)
    child = next(card for card in cards if card['record_id'] == 'child')
    assert child['parent_comparison']['comparable'] is expected
    assert next(card for card in cards if card['record_id'] == 'baseline')['evaluation_scope'] == 'baseline'
    store.close()


def test_M7_command_errors_do_not_trigger_model_recovery(ml_controller, monkeypatch):
    from automl_agent.contracts import ToolCommand
    c = ml_controller
    calls, recoveries = [], []
    leader = c.journal.get('leader_id')
    monkeypatch.setattr(c, 'prepare', lambda: None)
    monkeypatch.setattr(c, 'finalize', lambda: None)
    monkeypatch.setattr(c, 'recover_search', lambda error: recoveries.append(str(error)))
    def next_command(*args):
        calls.append(1)
        if len(calls) <= 4:
            raise ValueError('Pipeline view requires exactly one experiment ID and include_code=false')
        repair = c.journal.get('last_tool_result')
        assert repair['status'] == 'command_validation'
        assert repair['read_routes'][-1]['arguments']['ids'] == [leader]
        return ToolCommand(operation_id='safe_read', tool='list_run_records', arguments={}), ''
    monkeypatch.setattr(c.qwen, 'next_command', next_command)
    monkeypatch.setattr(c, 'dispatch', lambda command: (c.freeze('Offline command repair scenario complete'), {'status': 'success'})[1])
    c.run()
    assert len(calls) == 5 and not recoveries
    failures = [json.loads(row[0]) for row in c.journal.db.execute("SELECT payload FROM events WHERE kind='qwen_failure'")]
    assert len(failures) == 4 and all(row['error_category'] == 'command_validation' for row in failures)


@pytest.mark.parametrize('mode', ['native', 'json'])
def test_M5_M7_provider_success_with_invalid_arguments_keeps_identity(ml_controller, monkeypatch, mode):
    import httpx
    from automl_agent.controller import SYSTEM
    from test_context_integration import response
    c = ml_controller
    c.qwen.transport = mode
    c.qwen.client.close()
    calls = []
    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=response(calls[-1], operation='invalid_pipeline_read',
            tool='read_run_records', arguments={'view': 'pipeline', 'include_code': True}))
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(handle))
    with pytest.raises(ValueError):
        c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
    assert len(calls) == 1
    assert not c.journal.operation('invalid_pipeline_read')
    assert not c.context_memory.pending()


def test_M6_known_error_has_permitted_read_route_and_build_result(ml_controller, monkeypatch):
    from automl_agent.learned_blend import member_oof, oof_directory, OOFCacheError
    from automl_agent.contracts import RunReadArgs, ToolCommand
    from automl_agent.tool_results import payload
    from automl_agent.util import atomic_json
    c = ml_controller
    identifier = c.journal.get('baseline_id')
    spec, _ = c.lookup(identifier)
    directory = oof_directory(c, spec, c.plan.fit)
    request = c.stage(directory, spec, c.plan.fit, {}, mode='oof')
    request['context']['oof_seed'] = 42
    atomic_json(directory / 'request.json', request)
    atomic_json(directory / 'worker_result.json', {'status': 'error', 'diagnostic_code': 'error',
        'error': 'ValueError: Insufficient inner history', 'traceback': 'PRIVATE_TARGET_SENTINEL',
        'diagnostic_artifact': str(directory / 'worker_result.json')})
    monkeypatch.setattr(c.executor, 'run', lambda *args: pytest.fail('Known errors never refit'))
    with pytest.raises(OOFCacheError) as caught:
        member_oof(c, identifier, 5)
    route = caught.value.detail['diagnostic_read']
    # The returned route is executable with the ordinary public read contract.
    read = payload(c.dispatch(ToolCommand(operation_id='read_oof_error', tool=route['tool'], arguments=route['arguments'])))
    assert 'PRIVATE_TARGET_SENTINEL' not in canonical(read)
    assert 'Insufficient inner history' in canonical(read)


@pytest.mark.parametrize('shift', ['none', 'importance_weighting'])
def test_M6_real_worker_binds_original_request_before_context_mutation(ml_controller, monkeypatch, shift):
    from automl_agent.learned_blend import member_oof, oof_directory, OOFCacheError
    c = ml_controller
    base, result = c.lookup(c.journal.get('baseline_id'))
    spec = base.model_copy(update={'experiment_id': 'bad_oof', 'pipeline': PipelineSpec(model='ridge',
        hyperparameters={'alpha': -1}, shift_strategy=shift)})
    original_lookup = c.lookup
    monkeypatch.setattr(c, 'lookup', lambda identifier: (spec, result) if identifier == 'bad_oof' else original_lookup(identifier))
    with pytest.raises(OOFCacheError) as first:
        member_oof(c, 'bad_oof', 15)
    assert first.value.detail['error_code'] == 'oof_known_terminal_error'
    directory = oof_directory(c, spec, c.plan.fit)
    request = json.loads((directory / 'request.json').read_text())
    manifest = json.loads((directory / 'worker_result.json').read_text())
    assert manifest['request_hash'] == digest(request)
    monkeypatch.setattr(c.executor, 'run', lambda *args: pytest.fail('Terminal worker error cannot refit'))
    with pytest.raises(OOFCacheError) as second:
        member_oof(c, 'bad_oof', 15)
    assert first.value.detail == second.value.detail


@pytest.mark.parametrize('failure', ['unknown', 'unavailable'])
def test_M7_terminal_model_outcomes_do_not_retry(ml_controller, monkeypatch, failure):
    from automl_agent.qwen import QwenOutcomeUnknown, QwenUnavailable
    c = ml_controller
    monkeypatch.setattr(c, 'prepare', lambda: None)
    monkeypatch.setattr(c, 'finalize', lambda: None)
    calls, recovery = [], []
    def fail(*args):
        calls.append(1)
        raise (QwenOutcomeUnknown if failure == 'unknown' else QwenUnavailable)('offline fixture')
    monkeypatch.setattr(c.qwen, 'next_command', fail)
    monkeypatch.setattr(c, 'recover_search', lambda error: recovery.append(str(error)))
    c.run()
    assert len(calls) == 1 and not recovery and c.state == 'frozen'
    assert c.journal.get('last_tool_result')['status'] == ('model_outcome_unknown' if failure == 'unknown' else 'model_unavailable')
