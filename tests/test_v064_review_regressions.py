"""Regressions reproduced by the review of the v064 delivery."""
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest

from automl_agent.api import prediction_bundle
from automl_agent.budget import Limits
from automl_agent.contracts import ImputationPlan, MetricSpec, OutputSpec, PipelineSpec, TaskResolutionProposal, TaskSpec
from automl_agent.controller import Controller, SYSTEM
from automl_agent.data import Dataset, _explicit_validation_supported
from automl_agent.evaluation import align_predictions, score_predictions
from automl_agent.forecasting import ForecastPipeline
from automl_agent.preprocessing import FeatureMatrix
from automl_agent.publication import Publisher
from automl_agent.search import SearchPipeline
from automl_agent.validation import choose_validation
from context_compressor import BudgetExceeded
from conftest import write_task
from test_context_integration import controller, mock_client, response
from test_pipelines import context


def proposal(data, **updates):
    spec = json.loads((data.task_path.parent / 'task_spec.json').read_text())
    spec.update(contract_version=2, source_hash=data.source_hash,
        file_roles={name: name for name in ('train.csv', 'test.csv', 'sample_submission.csv')}, **updates)
    return TaskResolutionProposal.model_validate({'request_id': 'review', 'source_hash': data.source_hash,
        'status': 'proposed', 'task_spec': spec, 'evidence': {field: ['task', 'inventory'] for field in spec}})


@pytest.mark.parametrize('metric,kind,maximize', [('mape', 'regression', False), ('balanced_accuracy', 'classification', True)])
@pytest.mark.parametrize('mode', ['native', 'json'])
def test_supported_metrics_reach_a_real_compressed_request(tmp_path, monkeypatch, metric, kind, maximize, mode):
    path = write_task(tmp_path, kind=kind)
    spec_path = tmp_path / 'task_spec.json'
    spec = json.loads(spec_path.read_text())
    spec['metric'] = {'name': metric}
    spec_path.write_text(json.dumps(spec))
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-review-fixture')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://fixture.invalid/v1')
    c = Controller(path, Limits(seconds=180, cpu=1), run_id='metric', transport=mode)
    try:
        c.prepare()
        seen = []
        def handle(request):
            body = json.loads(request.content)
            core = json.loads(body['messages'][1]['content'])['core']
            seen.append(core)
            return httpx.Response(200, json=response(body))
        mock_client(c, handle)
        command, _ = c.qwen.next_command(SYSTEM, c.context_memory.snapshot(SYSTEM))
        assert command.tool == 'get_dataset_report' and len(seen) == 1
        assert seen[0]['task']['metric']['name'] == metric
        assert seen[0]['task']['maximize'] is maximize
        assert seen[0]['policies_resources']['selection']['maximize'] is maximize
    finally:
        c.close()


@pytest.mark.parametrize('strategy', ['random', 'group', 'temporal', 'temporal_group'])
@pytest.mark.parametrize('template', [
    'Do not use validation_strategy={}.',
    'Do not use `validation_strategy={}`.',
    '"validation_strategy={}" is prohibited.',
    'Не используйте validation_strategy={}.',
    'validation_strategy={0}. Do not use validation_strategy={0}.',
    'validation_strategy={0}, do not use validation_strategy={0}.',
])
def test_negated_protocol_assignments_are_not_instructions(strategy, template):
    assert not _explicit_validation_supported(strategy, template.format(strategy))
    assert _explicit_validation_supported(strategy, 'validation_strategy=' + strategy)


def test_negated_override_is_repairable_before_temporal_validation(tmp_path):
    path = write_task(tmp_path, temporal=True)
    path.write_text('Regression. RMSE. when is time; test is a future period. Do not use validation_strategy=random.')
    data = Dataset(path, deferred=True)
    with pytest.raises(ValueError, match='explicit_validation'):
        data.confirm_proposal(proposal(data, explicit_validation='random'), 'review')
    assert data.paths == {} and not hasattr(data, 'train')
    data.activate(data.confirm_proposal(proposal(data, explicit_validation=None), 'review'))
    plan = choose_validation(data)
    assert plan.strategy == 'temporal'
    assert data.train.iloc[plan.fit].when.max() < data.train.iloc[plan.validation].when.min()


@pytest.mark.parametrize('kind', ['classification', 'multioutput_classification', 'multilabel'])
@pytest.mark.parametrize('prediction_type', ['values', 'scores'])
@pytest.mark.parametrize('values', [[2, 2, 2], [0, 1, 0]])
def test_classification_numeric_bundles_cannot_be_scored_or_published(tmp_path, kind, prediction_type, values):
    task = TaskSpec(kind=kind, targets=['y'], feature_columns=['x'], classes={'y': [0, 1]},
        metric=MetricSpec(name='f1'), output=OutputSpec(columns=['y'], prediction_type='labels'), source_hash='data')
    X = pd.DataFrame({'x': [1, 2, 3]}, index=['a', 'b', 'c'])
    bundle = prediction_bundle({'y': values}, X, context(task), prediction_type)
    with pytest.raises(ValueError, match='Classification requires'):
        align_predictions(bundle, list(X.index), task, 'protocol', 'test')
    dataset = SimpleNamespace(task=task, task_path=tmp_path / 'task.md', test=X, test_keys=list(X.index))
    with pytest.raises(ValueError, match='Classification requires'):
        Publisher(dataset, SimpleNamespace(hash='protocol'), tmp_path, None).frame(bundle)
    for accepted, valid in [('labels', [0, 1, 0]), ('probabilities', [[1, 0], [0, 1], [1, 0]])]:
        bundle = prediction_bundle({'y': valid}, X, context(task), accepted)
        assert 'y' in align_predictions(bundle, list(X.index), task, 'protocol', 'test')


@pytest.mark.parametrize('mode', ['auto', 'json'])
def test_focus_uses_available_transport_when_native_overhead_exceeds_cap(controller, mode):
    c = controller
    unit = 'Улучшить качество, сохраняя метрику и независимую валидацию. '
    c.research_focus = unit
    c.context_memory.snapshot(SYSTEM)
    native_overhead = c.context_memory.compressor.transport.overhead_bytes('native')
    # Calibrate pressure against the actual prompt, whose size is versioned.
    # The test exercises fallback, not an incidental historical SYSTEM length.
    extra = max(0, (65536 - native_overhead) // len(unit.encode()) + 2)
    c.research_focus = unit * (1 + extra)
    c.qwen.transport = mode
    seen = []
    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=response(body))
    mock_client(c, handle)
    snapshot = c.context_memory.snapshot(SYSTEM)
    compressor = c.context_memory.compressor
    assert compressor.transport.overhead_bytes('native') > 65536
    assert compressor.transport.overhead_bytes('json') < 65536
    command, _ = c.qwen.next_command(SYSTEM, snapshot)
    assert command.tool == 'get_dataset_report' and c.qwen.transport == 'json'
    assert len(seen) == 1 and 'tools' not in seen[0]
    assert c.research_focus in seen[0]['messages'][0]['content']


def test_local_native_prepare_failure_falls_back_before_any_request(controller, monkeypatch):
    c = controller
    c.qwen.transport = 'auto'
    snapshot = c.context_memory.snapshot(SYSTEM)
    compressor = c.context_memory.compressor
    prepare = compressor.prepare
    modes = []
    def bounded(snapshot, mode, **options):
        modes.append(mode)
        if mode == 'native':
            raise BudgetExceeded('Native framing cannot fit this snapshot')
        return prepare(snapshot, mode, **options)
    monkeypatch.setattr(compressor, 'prepare', bounded)
    seen = []
    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=response(body))
    mock_client(c, handle)
    c.qwen.next_command(SYSTEM, snapshot)
    assert modes == ['native', 'json'] and len(seen) == 1 and 'tools' not in seen[0]


@pytest.mark.parametrize('crash_point,corrupt', [('before_result_commit', False), ('before_acceptance', False), ('before_result_commit', True)])
def test_completed_baseline_is_recovered_without_retraining(tmp_path, monkeypatch, crash_point, corrupt):
    path = write_task(tmp_path)
    first = Controller(path, Limits(seconds=300, cpu=1), run_id='baseline_crash')
    first.no_llm = True
    class SimulatedCrash(BaseException):
        pass
    save = first.journal.save_experiment
    def interrupt_save(spec, result=None):
        if result is not None:
            assert result.status == 'completed'
            raise SimulatedCrash
        save(spec, result)
    if crash_point == 'before_result_commit':
        monkeypatch.setattr(first.journal, 'save_experiment', interrupt_save)
    else:
        monkeypatch.setattr(first, 'accept_result', lambda *args: (_ for _ in ()).throw(SimulatedCrash()))
    try:
        with pytest.raises(SimulatedCrash):
            first.prepare()
        identifier = first.journal.get('baseline_id')
        directory = first.workdir / 'attempts' / identifier
        manifest = json.loads((directory / 'worker_result.json').read_text())
        assert manifest['status'] == 'completed'
        if corrupt and crash_point == 'before_result_commit':
            (directory / 'test_predictions.json').write_text('{}')
    finally:
        first.close()
    resumed = Controller(path, Limits(seconds=300, cpu=1), run_id='baseline_crash')
    resumed.no_llm = True
    def no_retraining(*args, **kwargs):
        pytest.fail('A completed baseline must not be launched again')
    monkeypatch.setattr(resumed.executor, 'run', no_retraining)
    try:
        resumed.prepare()
        result = resumed.lookup(identifier)[1]
        if corrupt and crash_point == 'before_result_commit':
            assert result.status == 'error' and 'checksum' in result.error
            assert resumed.journal.get('publication')['scope'] == 'fallback'
        else:
            assert result.status == 'completed' and result.score is not None
            assert resumed.journal.get('leader_id') == identifier
            assert resumed.journal.get('publication')['scope'] == 'fit'
            assert resumed.journal.get('publication')['experiment_id'] == identifier
        assert resumed.state == 'searching'
    finally:
        resumed.close()


def test_renamed_outputs_rejected_before_contract_is_frozen(tmp_path):
    path = write_task(tmp_path)
    sample_path = tmp_path / 'sample_submission.csv'
    sample = pd.read_csv(sample_path, dtype=str).rename(columns={'answer': 'prediction'})
    sample.to_csv(sample_path, index=False)
    data = Dataset(path, deferred=True)
    with pytest.raises(ValueError, match='Renamed output columns'):
        data.confirm_proposal(proposal(data, output={'columns': ['row', 'prediction'], 'prediction_type': 'values'}), 'review')
    assert data.paths == {} and not hasattr(data, 'train')


@pytest.mark.parametrize('positive,classes', [(None, None), ('alien', None), ('positive', ['negative', 'positive', 'third'])])
def test_binary_f1_missing_or_invalid_positive_label_is_repairable(tmp_path, positive, classes):
    data = Dataset(write_task(tmp_path, kind='classification'), deferred=True)
    with pytest.raises(ValueError, match='Binary F1 requires'):
        data.confirm_proposal(proposal(data, metric={'name': 'f1', 'average': 'binary', 'positive_label': positive},
            classes={'answer': classes} if classes else {}), 'review')
    assert data.paths == {} and not hasattr(data, 'train')
    data.activate(data.confirm_proposal(proposal(data, metric={'name': 'f1', 'average': 'binary', 'positive_label': 'positive'}), 'review'))
    assert score_predictions(data.task, data.train[['answer']], {'answer': data.train.answer.to_numpy()}, 'labels') == 1


@pytest.mark.parametrize('classification', [False, True])
def test_flaml_per_target_callback_does_not_apply_global_target_weights(classification):
    task = TaskSpec(kind='multioutput_classification' if classification else 'multioutput_regression',
        targets=['a', 'b'], feature_columns=['x'], source_hash='data',
        metric=MetricSpec(name='f1' if classification else 'rmse', aggregation='weighted', target_weights=[.2, .8]),
        classes={'a': ['no', 'yes'], 'b': ['no', 'yes']} if classification else {},
        output=OutputSpec(columns=['a', 'b'], prediction_type='labels' if classification else 'values'))
    pipeline = SearchPipeline(PipelineSpec(engine='flaml'), task)
    X = pd.DataFrame({'x': [1., 2., 3.]})
    estimator = SimpleNamespace(predict=lambda X: np.array([1., 2., 3.]), classes_=np.array([0, 1]),
        predict_proba=lambda X: np.array([[1., 0.], [0., 1.], [1., 0.]]))
    for target in task.targets:
        loss, detail = pipeline._flaml_metric(target, ['no', 'yes'] if classification else [])(
            X, np.array([0, 1, 0]) if classification else np.array([1., 2., 3.]), estimator)
        assert loss == (-1 if classification else 0)
        assert detail['primary'] == (1 if classification else 0)
    assert task.metric.target_weights == [.2, .8] and task.metric.aggregation == 'weighted'


@pytest.mark.parametrize('frequency', ['MS', 'ME', 'QS', 'YS'])
@pytest.mark.parametrize('gap', [1, 2])
@pytest.mark.parametrize('as_text', [False, True])
def test_calendar_forecasts_count_periods_instead_of_minimum_days(tmp_path, frequency, gap, as_text):
    dates = pd.date_range('2024-01-01', periods=4 + gap, freq=frequency)
    task = TaskSpec(kind='forecasting', targets=['y'], feature_columns=[], time_column='when', horizon=gap,
        metric=MetricSpec(name='rmse'), output=OutputSpec(columns=['y'], prediction_type='values'), source_hash='data')
    X = pd.DataFrame({'when': dates[:4].astype(str) if as_text else dates[:4]}, index=['a', 'b', 'c', 'd'])
    y = pd.DataFrame({'y': [10., 20., 30., 40.]}, index=X.index)
    pipeline = ForecastPipeline(PipelineSpec(model='exponential_smoothing', hyperparameters={'trend': 'add'}), task)
    ctx = context(task, tmp_path)
    pipeline.fit(X, y, ctx)
    before = list(pipeline.history[('all',)])
    future = str(dates[-1]) if as_text else dates[-1]
    predicted = pipeline.predict(pd.DataFrame({'when': [future]}, index=['test']), ctx).values['y'][0]
    expected = float(pipeline.models[(('all',), 'y')].forecast(steps=gap)[-1])
    assert predicted == pytest.approx(expected)
    assert pipeline.history[('all',)] == before
    with pytest.raises(ValueError, match='training frequency'):
        invalid = dates[-1] + pd.Timedelta(days=1)
        pipeline.predict(pd.DataFrame({'when': [str(invalid) if as_text else invalid]}, index=['test']), ctx)


@pytest.mark.parametrize('times,future,steps', [
    ([0, 1, 2, 3], 6, 3),
    ([0, 1, 3, 4], 20, 1),
    (['2024-01-01', '2024-01-03', '2024-02-15', '2024-04-01'], '2024-12-01', 1),
])
def test_forecast_fills_only_confirmed_regular_gaps(tmp_path, times, future, steps):
    task = TaskSpec(kind='forecasting', targets=['y'], feature_columns=[], time_column='when', horizon=steps,
        metric=MetricSpec(name='rmse'), output=OutputSpec(columns=['y'], prediction_type='values'), source_hash='data')
    X = pd.DataFrame({'when': times}, index=['a', 'b', 'c', 'd'])
    y = pd.DataFrame({'y': [10., 20., 30., 40.]}, index=X.index)
    pipeline = ForecastPipeline(PipelineSpec(model='exponential_smoothing', hyperparameters={'trend': 'add'}), task)
    ctx = context(task, tmp_path)
    pipeline.fit(X, y, ctx)
    predicted = pipeline.predict(pd.DataFrame({'when': [future]}, index=['test']), ctx).values['y'][0]
    assert predicted == pytest.approx(float(pipeline.models[(('all',), 'y')].forecast(steps=steps)[-1]))


@pytest.mark.parametrize('method', ['iterative_bayesian_ridge', 'iterative_trees'])
@pytest.mark.parametrize('text', [False, True])
def test_iterative_imputation_without_numeric_features_is_a_valid_noop(method, text):
    task = TaskSpec(kind='regression', targets=['y'], feature_columns=['x'], source_hash='data',
        text_columns=['x'] if text else [], categorical_columns=[] if text else ['x'],
        metric=MetricSpec(name='rmse'), output=OutputSpec(columns=['y'], prediction_type='values'))
    X = pd.DataFrame({'x': ['warm sunny', 'cold rainy', None, 'warm sunny', 'cold rainy']})
    matrix = FeatureMatrix(PipelineSpec(model='ridge', representation='mixed' if text else 'tabular',
        imputation=ImputationPlan(numeric=method)), task)
    transformed = matrix.fit_transform(X)
    assert transformed.shape[0] == len(X) and transformed.shape[1] > 0
    assert matrix.summary['actual_iterations'] == matrix.summary['auxiliary_fits'] == 0
    assert matrix.summary['model_imputation_executed'] is False
