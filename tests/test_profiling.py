"""EDA regression checks: shared facts, permitted data, bounded worker and delivery."""
import json
import httpx
import numpy as np
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import DatasetReportArgs, ToolCommand
from automl_agent.controller import Controller, SYSTEM
from automl_agent.data import Dataset
from automl_agent.profiling import build_profile, correlations, numeric_statistics, profile_key
from automl_agent.util import canonical
from conftest import write_task


def cached_profile(c, scope="fit"):
    return c.eda.profile(c.plan, scope)


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-eda-fixture')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://fixture.invalid/v1')
    c = Controller(write_task(tmp_path), Limits(seconds=600, cpu=1), run_id='eda_test')
    c.prepare()
    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail('Unexpected API request')))
    yield c
    c.close()


def report(c, **kwargs):
    result = c.run_tools.dataset(DatasetReportArgs(max_chars=12000, **kwargs))
    if 'report' in result:
        return result['report']
    parts = [result['content']]
    while result['continuation']:
        result = c.run_tools.dataset(DatasetReportArgs(**result['continuation']['arguments']))
        parts.append(result['content'])
    return json.loads(''.join(parts))


def test_numeric_statistics_and_empty_values():
    s = numeric_statistics(pd.Series([0., 1., 2., 3., -1., np.nan, np.inf]))
    assert s['mean'] == s['median'] == 1
    assert s['zero_fraction'] == s['negative_fraction'] == .2
    assert s['q25'] == 0 and s['q75'] == 2
    assert numeric_statistics(pd.Series([np.nan])) == {'status': 'no_finite_values'}


def test_single_profile_drives_tool_context_knowledge_and_column_projection(agent, monkeypatch):
    c = agent
    original = c.eda._profile
    calls = []
    def calculate(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(c.eda, '_profile', calculate)
    reference, document = cached_profile(c)
    full = report(c, report='profile', include_target=True)
    subset = report(c, report='profile', columns=['value'])
    snapshot = c.context_memory.snapshot(SYSTEM)
    assert len(calls) == 1
    assert full['columns'] == document['columns']
    assert full['targets'] == document['targets']
    assert subset['columns'] == {'value': full['columns']['value']}
    assert not subset['targets']
    summary = json.loads(snapshot.profile_json)['summary']
    assert summary == reference['summary']
    assert len(summary['findings']) <= 3
    selection = c.knowledge.selection_context(summary)
    flags = document['observations'][0]['flags']
    assert all(selection.fit_profile[name] == value for name, value in flags.items())


def test_targets_cannot_escape_fit_and_cache_ignores_closed_labels(agent):
    c = agent
    before = build_profile(c.dataset, c.plan.fit).model_dump()
    c.dataset.train.loc[c.plan.validation + c.plan.control, 'answer'] = 1e100
    assert before == build_profile(c.dataset, c.plan.fit).model_dump()
    for scope in ['validation_features', 'test_features']:
        assert not report(c, scope=scope)['targets']
        with pytest.raises(ValueError):
            report(c, scope=scope, include_target=True)
        with pytest.raises(ValueError):
            report(c, scope=scope, report='correlations', columns=['answer'])


def test_filtered_profile_has_own_scope_and_does_not_poison_full_profile(agent):
    c = agent
    full = report(c, include_target=True)
    filtered = report(c, include_target=True, predicate={'column': 'value', 'op': 'gt', 'value': 0})
    assert filtered['rows'] < full['rows']
    assert filtered['selection']['predicate']['op'] == 'gt'
    assert filtered['targets']['answer'] != full['targets']['answer']
    assert report(c, include_target=True) == full


def test_cache_identity_tracks_roles_and_split_and_persisted_profile_survives_restart(agent, monkeypatch):
    c = agent
    old = profile_key(c.dataset, c.plan.fit, 'fit')
    assert profile_key(c.dataset, c.plan.fit[:-1], 'fit') != old
    assert profile_key(c.dataset, c.plan.fit, 'validation_features') != old
    cached_profile(c)
    monkeypatch.setattr(c.dataset, 'profile', lambda *a: pytest.fail('Persisted document recomputed'))
    cached_profile(c)
    c.task = c.task.model_copy(update={'categorical_columns': ['code', 'value']})
    c.dataset.task = c.task
    assert profile_key(c.dataset, c.plan.fit, 'fit') != old


def test_class_profile_covers_declared_absent_class_and_preserves_risk_rules(tmp_path):
    dataset = Dataset(write_task(tmp_path, kind='classification'))
    task, _ = dataset.draft()
    dataset.activate(task)
    dataset.task.classes['answer'].append('absent')
    profile = dataset.profile(list(range(100)))
    assert profile.targets['answer']['counts'][-1] == {'label': 'absent', 'rows': 0}
    flags = profile.observations[0]['flags']
    assert flags['rare_classes'] and flags['class_imbalance']


def test_correlations_pairwise_support_constants_and_nonlinear_monotone_relation():
    x = np.arange(60, dtype=float)
    frame = pd.DataFrame({'x': x, 'cube': x ** 3, 'constant': 1., 'rare': np.where(x < 29, x, np.nan)})
    result = correlations(frame)
    assert result['spearman']['x']['cube'] == pytest.approx(1)
    assert result['correlations']['x']['cube'] < .95
    assert result['pair_counts']['rare']['x'] == 29
    assert result['pair_status']['x']['constant'] == 'constant'
    assert result['pair_status']['x']['rare'] == 'insufficient_support'
    assert result['correlations']['x']['rare'] is None
    json.dumps(result, allow_nan=False)


def test_real_worker_excludes_categorical_codes_and_keeps_csv(agent):
    c = agent
    csv = c.publisher.output.read_bytes()
    result = report(c, report='correlations', include_target=True, columns=['value', 'answer', 'code'])
    assert result['status'] == 'available', result
    assert result['excluded_columns'] == {'code': 'unsupported_type'}
    assert result['pair_counts']['value']['answer'] >= 30
    assert result['sample_rows'] == len(c.plan.fit)
    assert c.publisher.output.read_bytes() == csv
    assert c.journal.get('active_worker') is None


def test_classification_target_and_numeric_category_are_not_correlated(agent):
    c = agent
    c.task = c.task.model_copy(update={'categorical_columns': ['code', 'value']})
    c.dataset.task = c.task
    result = report(c, report='correlations', columns=['value', 'code'])
    assert result['status'] == 'available', result
    assert result['excluded_columns'] == {'value': 'unsupported_type', 'code': 'unsupported_type'}
    assert result['correlations'] == {}


def test_failure_budget_and_pagination_do_not_repeat_worker(agent, monkeypatch):
    c = agent
    csv = c.publisher.output.read_bytes()
    calls = []
    def timeout(*args):
        calls.append(args[-1])
        return {'status': 'timeout'}
    monkeypatch.setattr(c.executor, 'run', timeout)
    result = report(c, report='correlations', columns=['value', 'empty'])
    assert result['status'] == 'technical_failure' and result['diagnostic_status'] == 'timeout'
    assert report(c, report='correlations', columns=['value', 'empty']) == result
    assert len(calls) == 1 and 0 < calls[0] <= 20
    monkeypatch.setattr(c.budget, 'final_reserve', lambda *a: c.budget.remaining + 1)
    assert report(c, report='correlations', columns=['value'])['status'] == 'budget_too_small'
    assert len(calls) == 1 and c.publisher.output.read_bytes() == csv


def test_pagination_reads_saved_profile_even_after_cache_is_cold(agent, monkeypatch):
    c = agent
    args = DatasetReportArgs(include_target=True, max_chars=2600)
    first = c.run_tools.dataset(args)
    assert first['continuation']
    monkeypatch.setattr(c.dataset, 'inspect', lambda *a, **k: pytest.fail('Page recomputed profile'))
    second = c.run_tools.dataset(DatasetReportArgs(**first['continuation']['arguments']))
    assert second['range']['start'] == first['range']['end']
    changed = dict(first['continuation']['arguments'], columns=['value'])
    with pytest.raises(ValueError):
        c.run_tools.dataset(DatasetReportArgs(**changed))


def test_findings_survive_both_wire_transports_under_pressure(agent):
    c = agent
    snapshot = c.context_memory.snapshot(SYSTEM)
    for mode in ['native', 'json']:
        prepared = c.context_memory.compressor.prepare(snapshot, mode, budget=131072)
        body = json.loads(prepared.wire_json)
        context = json.loads(body['messages'][1]['content'])
        assert 'findings' in canonical(context)
        assert 'Check feature representation' in canonical(context)
        assert 'read_run_records' in canonical(context)


def test_offline_research_cycle_profiles_correlates_trains_and_finishes(agent):
    c = agent
    steps = iter(['profile', 'correlations', 'train', 'finish'])
    def command():
        step = next(steps)
        if step in {'profile', 'correlations'}:
            return ToolCommand(operation_id='eda_' + step, tool='get_dataset_report', arguments={
                'report': step, 'columns': ['value', 'answer'], 'include_target': True}), 'Offline diagnostic hypothesis'
        if step == 'train':
            return ToolCommand(operation_id='eda_train', tool='train_model', arguments={
                'pipeline': {'model': 'ridge', 'hyperparameters': {'alpha': 2.}},
                'budget_seconds': 30,
                'hypothesis': {'observation': 'Fit-only profile and correlations are available; compare regularization.',
                    'expected_effect': 'Test validation RMSE', 'changed_component': 'alpha', 'acceptance': 'Compare same protocol'}}), 'Offline experiment'
        return ToolCommand(operation_id='eda_finish', tool='finish', arguments={
            'reason': 'Offline EDA integration completed', 'result_ids': [c.journal.get('leader_id')]}), 'Offline finish'
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        selected, narrative = command()
        if 'tools' in body:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'offline_' + selected.operation_id, 'type': 'function', 'function': {
                    'name': selected.tool, 'arguments': canonical({'operation_id': selected.operation_id, **selected.arguments})}}]}
        else:
            message = {'role': 'assistant', 'content': selected.model_dump_json()}
        return httpx.Response(200, json={'choices': [{'message': message,
            'finish_reason': 'tool_calls' if 'tools' in body else 'stop'}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}})
    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(handler))
    result = c.run()
    assert result['state'] == 'completed', result
    assert result['control_status'] == result['refit_status'] == 'completed'
    operations = [c.journal.operation('eda_' + name) for name in ['profile', 'correlations', 'train', 'finish']]
    assert all(op.status == 'succeeded' for op in operations)
    assert c.publisher.output.exists()
    assert len(requests) == 4


def test_sampling_is_bounded_reproducible_and_independent_of_target_values(agent, monkeypatch):
    c = agent
    extended = pd.concat([c.dataset.train] * 120, ignore_index=True)
    c.dataset.train = extended
    c.dataset.row_keys = [f'train:{i}' for i in range(len(extended))]
    monkeypatch.setattr(c.eda, 'finalist_costs', lambda: ())
    plan = c.plan.model_copy(update={'fit': list(range(len(extended)))})
    captured = []
    def stage(path, frame, *, deadline, memory_mb):
        assert deadline <= c.budget.deadline and memory_mb == c.executor.memory_mb
        captured.append(frame.copy())
    monkeypatch.setattr('automl_agent.eda.service.stage_frame', stage)
    monkeypatch.setattr(c.executor, 'run', lambda *a: {'status': 'completed', 'report': {}})
    args = DatasetReportArgs(report='correlations', columns=['value', 'answer'], include_target=True)
    first = c.eda.correlation_report(args, plan)
    c.dataset.train['answer'] = -c.dataset.train['answer']
    second = c.eda.correlation_report(args, plan)
    assert first['sampled'] and first['sample_rows'] == 20000 and first['source_rows'] == 21600
    assert second['sample_rows'] == 20000
    assert captured[0].index.equals(captured[1].index)
    pd.testing.assert_series_equal(captured[0]['value'], captured[1]['value'])


def test_wide_default_requires_explicit_selection_before_worker(agent, monkeypatch):
    c = agent
    columns = [f'col_{i}' for i in range(31)]
    for name in columns:
        c.dataset.train[name] = 1.
        c.dataset.test[name] = 1.
    c.dataset.task = c.task.model_copy(update={'feature_columns': columns, 'categorical_columns': []})
    monkeypatch.setattr(c.executor, 'run', lambda *a: pytest.fail('Wide correlation started a worker'))
    result, _ = c.eda.report(DatasetReportArgs(report='correlations'), c.plan)
    assert result['status'] == 'incompatible' and '30 columns' in result['error']


def test_optional_staging_failure_preserves_csv_and_search_state(agent, monkeypatch):
    c = agent
    before = c.publisher.output.read_bytes()
    def fail(*args, **kwargs):
        assert kwargs["deadline"] <= c.budget.deadline
        raise OSError('Synthetic local staging failure')
    monkeypatch.setattr('automl_agent.eda.service.stage_frame', fail)
    result = report(c, report='correlations', columns=['value'])
    assert result['status'] == 'technical_failure' and result['diagnostic_status'] == 'OSError'
    assert c.state == 'searching' and c.publisher.output.read_bytes() == before


def test_profile_cache_and_report_pages_survive_new_controller_instance(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-eda-fixture')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://fixture.invalid/v1')
    path = write_task(tmp_path)
    c = Controller(path, Limits(seconds=600, cpu=1), run_id='eda_restart')
    try:
        c.prepare()
        expected = cached_profile(c)
        first = c.run_tools.dataset(DatasetReportArgs(include_target=True, max_chars=2600))
        assert first['continuation']
    finally:
        c.close()
    resumed = Controller(path, Limits(seconds=600, cpu=1), run_id='eda_restart')
    try:
        resumed.prepare()
        monkeypatch.setattr(resumed.dataset, 'profile', lambda *a: pytest.fail('Restart recomputed cached profile'))
        monkeypatch.setattr(resumed.dataset, 'inspect', lambda *a, **k: pytest.fail('Restart recomputed report'))
        assert cached_profile(resumed) == expected
        following = resumed.run_tools.dataset(DatasetReportArgs(**first['continuation']['arguments']))
        assert following['range']['start'] == first['range']['end']
    finally:
        resumed.close()


def test_old_eda_runtime_requires_frozen_runtime_or_new_run(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-eda-fixture')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://fixture.invalid/v1')
    path = write_task(tmp_path)
    c = Controller(path, run_id='old_eda')
    c.journal.put('eda_profile_version', None)
    c.close()
    with pytest.raises(ValueError, match='frozen runtime'):
        Controller(path, run_id='old_eda')


def test_nonfinite_summary_statistics_are_explicit_and_json_safe():
    stats = numeric_statistics(pd.Series([1e308, 1e308, -1e308]))
    assert stats['status'] == 'partial'
    assert 'std' in stats['undefined_statistics']
    json.dumps(stats, allow_nan=False)


def test_flat_runtime_materializes_nested_eda_package(tmp_path, monkeypatch):
    # Exercise the actual packer on tiny project-owned fixtures, not a competition candidate.
    import hashlib
    import runpy
    import zipfile
    from tools import build_flat_submission as packer
    root = tmp_path / 'source'
    root.mkdir()
    files = ['solution.py', 'requirements.txt', 'automl_agent/__init__.py',
        'automl_agent/eda/__init__.py', 'automl_agent/eda/service.py',
        'ml_knowledge/__init__.py', 'data_for_participants/requirements.txt']
    files += ['context_compressor/' + name + '.py' for name in ['__init__', 'contracts', 'store', 'core', 'transport', 'assembler']]
    modules = ['core', 'lexical', 'runtime', 'memory', 'context']
    files += ['ml_knowledge/' + name + '.py' for name in modules]
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')
    marker = 'class EDAService: pass\n'
    (root / 'automl_agent/eda/service.py').write_text(marker)
    release = root / 'ml_knowledge/releases/ml_knowledge_v006.zip'
    release.parent.mkdir()
    with zipfile.ZipFile(release, 'w') as archive:
        for name in modules:
            archive.writestr('mlkb_' + name + '.py', '')
        archive.writestr('ml_knowledge_bundle.py', '')
    release.with_suffix('.json').write_text(json.dumps({'zip_sha256': hashlib.sha256(release.read_bytes()).hexdigest(), 'corpus': {}}))
    monkeypatch.setattr(packer, 'ROOT', root)
    destination = tmp_path / 'delivery'
    packer.build(destination)
    assert all(path.suffix == '.py' or path.name == 'requirements.txt' for path in destination.iterdir())
    bootstrap = runpy.run_path(str(destination / 'agent_bundle.py'))
    restored = bootstrap['materialize'](tmp_path / 'task')
    assert (restored / 'automl_agent/eda/service.py').read_text() == marker


def test_boolean_numeric_features_and_nonfit_row_counts(agent):
    assert numeric_statistics(pd.Series([True, False]))['median'] == .5
    _, doc = agent.eda.profile(agent.plan, 'validation_features')
    assert doc['rows']['profile'] == len(agent.plan.validation)
    assert 'fit' not in doc['rows'] and not doc['targets']
