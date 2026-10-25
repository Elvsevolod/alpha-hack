"""A scored boosting model must execute the schedule advertised by its recipe."""
import cloudpickle
import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import MetricSpec, OutputSpec, PipelineSpec, TaskSpec
from automl_agent.pipelines import DirectPipeline
from test_pipelines import context


def opposing_holdout():
    rng = np.random.default_rng(27)
    X = pd.DataFrame({'x': rng.normal(size=200)}, index=[f'fit:{i}' for i in range(200)])
    V = pd.DataFrame({'x': rng.normal(size=60)}, index=[f'valid:{i}' for i in range(60)])
    y = pd.DataFrame({'answer': 2 * X.x}, index=X.index)
    z = pd.DataFrame({'answer': -2 * V.x}, index=V.index)
    task = TaskSpec(kind='regression', targets=['answer'], feature_columns=['x'],
        metric=MetricSpec(name='rmse'), output=OutputSpec(columns=['answer'], prediction_type='values'),
        source_hash='synthetic_schedule')
    return X, y, V, z, task


def test_histogram_scored_model_restores_the_selected_state(tmp_path, monkeypatch):
    from sklearn.ensemble import HistGradientBoostingRegressor
    calls = []
    native_fit = HistGradientBoostingRegressor.fit
    def record_fit(model, *args, **kwargs):
        calls.append(model.max_iter)
        return native_fit(model, *args, **kwargs)
    monkeypatch.setattr(HistGradientBoostingRegressor, 'fit', record_fit)
    X, y, V, z, task = opposing_holdout()
    spec = PipelineSpec(model='hist_gradient_boosting',
        hyperparameters={'max_iter': 20, 'min_samples_leaf': 5}, early_stopping_rounds=2, evaluation_interval=2)
    c = {**context(task, tmp_path), 'schedule_selection': True, 'schedule_validation': (V, z)}
    pipeline = DirectPipeline(spec, task)
    summary = pipeline.fit(X, y, c)
    frozen = pipeline.frozen_spec()
    assert frozen.iteration_schedule['answer'] == 2
    assert pipeline.models['answer'].n_iter_ == frozen.iteration_schedule['answer']
    assert calls == [2, 4]  # Restore a snapshot; no extra training to recover it.
    # Rebuild on exactly the same permitted population: the scored state must
    # match its portable frozen recipe, not the last stale warm-start fit.
    replay = DirectPipeline(frozen, task)
    replay.fit(X, y, context(task, tmp_path))
    expected = replay.predict(V, context(task)).values['answer']
    actual = pipeline.predict(V, context(task)).values['answer']
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    restored = cloudpickle.loads(cloudpickle.dumps(pipeline))
    np.testing.assert_allclose(restored.predict(V, context(task)).values['answer'], expected)
    rmse = np.sqrt(np.mean((np.asarray(actual) - z.answer.to_numpy()) ** 2))
    assert rmse == pytest.approx(min(p['primary_metric'] for p in summary.learning_curve['answer']['curve']))


def test_histogram_curve_contains_executed_one_based_checkpoints(tmp_path):
    X, y, V, z, task = opposing_holdout()
    spec = PipelineSpec(model='hist_gradient_boosting', hyperparameters={'max_iter': 7},
        early_stopping_rounds=100, evaluation_interval=3)
    p = DirectPipeline(spec, task)
    p.fit(X, y, {**context(task, tmp_path), 'schedule_selection': True, 'schedule_validation': (V, z)})
    assert [row['iteration'] for row in p.schedules['answer']['curve']] == [3, 6, 7]


def test_histogram_macro_f1_restores_state_and_preserves_declared_class_order(tmp_path):
    from automl_agent.evaluation import align_predictions, score_predictions

    X, y, V, z, task = opposing_holdout()
    y['answer'] = np.where(y.answer > 0, 'alpha', 'zeta')
    z['answer'] = np.where(z.answer > 0, 'alpha', 'zeta')
    task = task.model_copy(update={'kind':'classification', 'classes':{'answer':['zeta', 'alpha']},
        'metric':MetricSpec(name='f1', average='macro'),
        'output':OutputSpec(columns=['answer'], prediction_type='labels')})
    spec = PipelineSpec(model='hist_gradient_boosting', hyperparameters={'max_iter':20, 'min_samples_leaf':5},
        early_stopping_rounds=2, evaluation_interval=2)
    c = {**context(task, tmp_path), 'schedule_selection':True, 'schedule_validation':(V, z)}
    p = DirectPipeline(spec, task)
    p.fit(X, y, c)
    frozen = p.frozen_spec()
    replay = DirectPipeline(frozen, task)
    replay.fit(X, y, context(task, tmp_path))
    actual, expected = p.predict(V, c), replay.predict(V, c)
    np.testing.assert_allclose(actual.values['answer'], expected.values['answer'], atol=1e-12)
    assert p.models['answer'].n_iter_ == frozen.iteration_schedule['answer']
    probabilities = align_predictions(actual, V.index.tolist(), task, 'protocol', 'test')
    score = score_predictions(task, z, probabilities, actual.prediction_type)
    assert score == pytest.approx(max(row['primary_metric'] for row in p.schedules['answer']['curve']))


@pytest.mark.parametrize('model,parameter', [('lightgbm', 'n_estimators'), ('xgboost', 'n_estimators'),
    ('catboost', 'iterations'), ('hist_gradient_boosting', 'max_iter')])
@pytest.mark.parametrize('field', ['iteration_schedule', 'frozen_iterations'])
def test_frozen_iterations_take_priority_over_inner_selection_flag(tabular, tmp_path, monkeypatch, model, parameter, field):
    import automl_agent.training_schedule as schedule
    X, y, task = tabular
    def forbidden(*args, **kwargs):
        pytest.fail('An already frozen schedule must not access the supplied holdout or select again')
    monkeypatch.setattr(schedule, 'fit_with_schedule', forbidden)
    spec = PipelineSpec(model=model, hyperparameters={parameter: 20}, early_stopping_rounds=2,
        **{field: {'answer': 3} if field == 'iteration_schedule' else 3})
    p = DirectPipeline(spec, task)
    # No schedule_validation is supplied intentionally: a fixed fit needs none.
    p.fit(X, y, {**context(task, tmp_path), 'schedule_selection': True})
    native = p.models['answer']
    count = native.n_iter_ if model == 'hist_gradient_boosting' else native.tree_count_ if model == 'catboost' else native.get_booster().num_boosted_rounds() if model == 'xgboost' else native.n_estimators_
    assert count == 3
    assert p.summary.auxiliary_fits == 0


def test_checkpoint_selection_is_not_overwritten_by_native_early_stopping(tabular, tmp_path, monkeypatch):
    import automl_agent.training_schedule as schedule
    X, y, task = tabular
    def forbidden(*args, **kwargs):
        pytest.fail('Explicit checkpoints have already selected the schedule')
    monkeypatch.setattr(schedule, 'fit_with_schedule', forbidden)
    spec = PipelineSpec(model='lightgbm', hyperparameters={'n_estimators': 12},
        checkpoint_counts=[2, 4], early_stopping_rounds=2)
    p = DirectPipeline(spec, task)
    p.fit(X, y, {**context(task, tmp_path), 'schedule_selection': True})
    assert p.schedules['answer']['scope'] == 'fit_inner_final_output_selection'
    assert p.models['answer'].n_estimators_ == p.frozen_spec().iteration_schedule['answer']


def test_histogram_hpo_score_matches_frozen_inner_replay_then_refits_without_search(tmp_path, monkeypatch):
    import optuna
    from automl_agent.pipelines import build_pipeline
    from automl_agent.validation import inner_split

    X, y, _, _, task = opposing_holdout()
    spec = PipelineSpec(engine='optuna', model='hist_gradient_boosting', n_trials=1,
        hyperparameters={'max_iter':20, 'min_samples_leaf':5}, early_stopping_rounds=2,
        evaluation_interval=2, search_space={'learning_rate':{'type':'categorical', 'choices':[.1]}})
    a, b = inner_split(X, y, task, 'random', spec.seed)
    y.iloc[b, 0] = -2 * X.iloc[b].x.to_numpy()
    c = {**context(task, tmp_path), 'budget_seconds':30}
    p = build_pipeline(spec, task, c)
    summary = p.fit(X, y, c)
    winner = next(t for t in summary.inner_scores['trials'] if t['state'] == 'completed')
    frozen = p.frozen_spec()
    replay = DirectPipeline(frozen, task)
    replay.fit(X.iloc[a], y.iloc[a], c)
    pred = np.asarray(replay.predict(X.iloc[b], c).values['answer'])
    expected = np.sqrt(np.mean((pred - y.iloc[b].answer.to_numpy()) ** 2))
    assert winner['value'] == pytest.approx(expected, abs=1e-12)
    assert winner['iteration_schedule'] == frozen.iteration_schedule
    restored = cloudpickle.loads(cloudpickle.dumps(p))
    np.testing.assert_allclose(restored.predict(X, c).values['answer'], p.predict(X, c).values['answer'])
    def forbidden(*args, **kwargs):
        pytest.fail('Final refit must not launch another HPO search')
    monkeypatch.setattr(optuna, 'create_study', forbidden)
    refit = restored.refit(X, y, {**c, 'execution_mode':'refit'})
    assert refit.rows == len(X) and refit.auxiliary_fits == 0
    assert restored.pipeline.models['answer'].n_iter_ == frozen.iteration_schedule['answer']


@pytest.mark.parametrize('model,parameter,checkpoints', [
    ('hist_gradient_boosting', 'max_iter', []), ('lightgbm', 'n_estimators', [2, 4])])
def test_worker_persists_the_executed_hpo_schedule(tmp_path, model, parameter, checkpoints):
    import json
    from pathlib import Path
    from automl_agent.budget import Limits
    from automl_agent.contracts import ToolCommand
    from automl_agent.controller import Controller
    from automl_agent.tool_results import payload
    from conftest import write_task

    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1), run_id='schedule_worker')
    try:
        c.prepare()
        result = payload(c.dispatch(ToolCommand(operation_id='scheduled_search', tool='tune_model',
            arguments={'pipeline':{'engine':'optuna', 'model':model, 'n_trials':1,
                'hyperparameters':{parameter:12}, 'early_stopping_rounds':2, 'evaluation_interval':2,
                'checkpoint_counts':checkpoints,
                'search_space':{'learning_rate':{'type':'categorical', 'choices':[.1]}}},
                'budget_seconds':60, 'hypothesis':{'observation':'Offline schedule consistency fixture',
                    'expected_effect':'Verify the selected iteration count reaches the saved worker model',
                    'changed_component':'schedule', 'acceptance':'Matching executed and frozen counts'}})))
        assert result['result']['status'] == 'completed', result
        _, saved = c.lookup(result['experiment']['experiment_id'])
        directory = Path(saved.artifacts['directory'])
        frozen = PipelineSpec.model_validate(json.loads((directory / 'worker_result.json').read_text())['frozen_pipeline'])
        with (directory / 'model.pkl').open('rb') as stream:
            fitted = cloudpickle.load(stream)
        count = frozen.iteration_schedule['answer']
        native = fitted.pipeline.models['answer']
        assert (native.n_iter_ if model == 'hist_gradient_boosting' else native.n_estimators_) == count
        assert fitted.frozen_spec().hash == frozen.hash
        if checkpoints:
            assert count in checkpoints
        assert c.budget.counter('llm_calls') == 0 and not c.journal.get('control_opened')
    finally:
        c.close()
