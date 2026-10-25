"""CatBoost's plural class_weights must retain confirmed label identity."""
import json

import cloudpickle
import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import MetricSpec, PipelineSpec
from automl_agent.pipelines import build_pipeline
from test_class_weight_contract import context, task_for


@pytest.mark.parametrize('labels', [[0, 1], [20, 10], ['no', 'yes'], ['1', '0']])
def test_catboost_dictionary_survives_json_predict_and_frozen_replay(tmp_path, labels):
    task = task_for(labels)
    X = pd.DataFrame({'x': np.linspace(-2, 2, 100)}, index=[f'r{i}' for i in range(100)])
    y = pd.DataFrame({'answer': np.where(X.x > 0, labels[1], labels[0])}, index=X.index)
    weights = {labels[0]:1, labels[1]:4}
    spec = PipelineSpec.model_validate_json(PipelineSpec(model='catboost',
        hyperparameters={'iterations':8, 'class_weights':weights}).model_dump_json())
    p = build_pipeline(spec, task, {})
    c = context(task, tmp_path)
    p.fit(X, y, c)
    bundle = p.predict(X.iloc[:12], c)
    native = p.models['answer']
    assert native.get_params()['class_weights'] == weights
    raw = native.predict_proba(p.preprocessor.transform(X.iloc[:12]))
    expected = raw[:, [native.classes_.tolist().index(label) for label in labels]]
    np.testing.assert_allclose(bundle.values['answer'], expected)
    restored = cloudpickle.loads(cloudpickle.dumps(p))
    np.testing.assert_allclose(restored.predict(X.iloc[:12], c).values['answer'], expected)
    frozen = PipelineSpec.model_validate_json(p.frozen_spec().model_dump_json())
    assert frozen.hyperparameters['class_weights'] == spec.hyperparameters['class_weights']
    replay = build_pipeline(frozen, task, {})
    replay.fit(X, y, {**c, 'execution_mode':'refit'})
    np.testing.assert_allclose(replay.predict(X.iloc[:12], c).values['answer'], expected)


@pytest.mark.parametrize('weights', [{'1':4}, {'no':1}, {'no':1, 'yes':-1},
    {'no':0, 'yes':0}, {'no':1, 'yes':True}, {'no':1, 'yes':float('inf')}])
def test_invalid_catboost_dictionary_fails_before_preprocessing(monkeypatch, weights):
    task = task_for(['no', 'yes'])
    monkeypatch.setattr('automl_agent.preprocessing.FeatureMatrix.fit',
        lambda *args: pytest.fail('Invalid weights must fail before preprocessing'))
    spec = PipelineSpec(model='catboost', hyperparameters={'class_weights':weights})
    with pytest.raises(ValueError, match='class_weights'):
        build_pipeline(spec, task, {})


def test_catboost_dictionary_and_auto_weights_are_not_combined():
    spec = PipelineSpec(model='catboost', hyperparameters={'class_weights':{'0':1, '1':4},
        'auto_class_weights':'Balanced'})
    with pytest.raises(ValueError, match='class_weights'):
        build_pipeline(spec, task_for([0, 1]), {})


def test_absent_inner_class_keeps_its_weight_for_full_refit(tmp_path):
    task = task_for([0, 1, 2])
    X = pd.DataFrame({'x':np.linspace(-2, 2, 90)}, index=[f'r{i}' for i in range(90)])
    y = pd.DataFrame({'answer':np.tile([0, 1, 2], 30)}, index=X.index)
    spec = PipelineSpec(model='catboost', hyperparameters={'iterations':5,
        'class_weights':{'0':1, '1':2, '2':9}})
    mask = y.answer != 2
    p = build_pipeline(spec, task, {})
    c = context(task, tmp_path)
    p.fit(X.loc[mask], y.loc[mask], c)
    assert p.models['answer'].get_params()['class_weights'] == {0:1, 1:2}
    frozen = p.frozen_spec()
    assert frozen.hyperparameters['class_weights']['2'] == 9
    full = build_pipeline(frozen, task, {})
    full.fit(X, y, {**c, 'execution_mode':'refit'})
    assert full.models['answer'].get_params()['class_weights'] == {0:1, 1:2, 2:9}
    assert np.asarray(full.predict(X.iloc[:3], c).values['answer']).shape == (3, 3)


def test_binary_stopping_metric_uses_catboost_dictionary_class_order(tmp_path):
    from automl_agent.evaluation import score_predictions
    rng = np.random.default_rng(391)
    X = pd.DataFrame({'x':rng.normal(size=220)}, index=[f'r{i}' for i in range(220)])
    y = pd.DataFrame({'answer':np.where(X.x + rng.normal(size=220) > .7, 20, 10)}, index=X.index)
    task = task_for([20, 10]).model_copy(update={'metric':MetricSpec(name='f1', average='binary', positive_label=20)})
    spec = PipelineSpec(model='catboost', hyperparameters={'iterations':20, 'depth':3,
        'class_weights':{20:1, 10:4}}, early_stopping_rounds=3, evaluation_interval=1)
    c = {**context(task, tmp_path), 'schedule_selection':True,
        'schedule_validation':(X.iloc[160:], y.iloc[160:])}
    p = build_pipeline(spec, task, {})
    p.fit(X.iloc[:160], y.iloc[:160], c)
    bundle = p.predict(X.iloc[160:], c)
    score = score_predictions(task, y.iloc[160:], {'answer':np.asarray(bundle.values['answer'])}, bundle.prediction_type)
    selected = p.frozen_spec().iteration_schedule['answer']
    point = next(row for row in p.schedules['answer']['curve'] if row['iteration'] == selected)
    assert point['primary_metric'] == pytest.approx(score)


def test_optuna_catboost_preserves_dictionary_and_frozen_predictions(tmp_path):
    task = task_for([20, 10])
    X = pd.DataFrame({'x':np.linspace(-2, 2, 100)}, index=[f'r{i}' for i in range(100)])
    y = pd.DataFrame({'answer':np.where(X.x > 0, 10, 20)}, index=X.index)
    spec = PipelineSpec(engine='optuna', model='catboost', n_trials=2,
        hyperparameters={'iterations':5, 'class_weights':{'20':1, '10':4}},
        search_space={'depth':{'type':'categorical', 'choices':[2, 3]}})
    p = build_pipeline(spec, task, {})
    c = context(task, tmp_path)
    p.fit(X, y, c)
    assert p.pipeline.models['answer'].get_params()['class_weights'] == {20:1, 10:4}
    expected = p.predict(X.iloc[:8], c).values['answer']
    frozen = PipelineSpec.model_validate_json(p.frozen_spec().model_dump_json())
    assert frozen.hyperparameters['class_weights'] == {'20':1, '10':4}
    replay = build_pipeline(frozen, task, {})
    replay.fit(X, y, {**c, 'execution_mode':'refit'})
    np.testing.assert_allclose(replay.predict(X.iloc[:8], c).values['answer'], expected)


def test_worker_receives_typed_catboost_weight_keys_and_rejects_unknown_labels(tmp_path):
    from pathlib import Path
    from automl_agent.budget import Limits
    from automl_agent.contracts import ToolCommand
    from automl_agent.controller import Controller
    from automl_agent.tool_results import payload
    from conftest import write_task

    path = write_task(tmp_path, kind='classification')
    frame = pd.read_csv(tmp_path / 'train.csv', dtype=str, keep_default_na=False)
    frame['answer'] = frame.answer.map({'negative':0, 'positive':1})
    frame.to_csv(tmp_path / 'train.csv', index=False)
    task = json.loads((tmp_path / 'task_spec.json').read_text())
    task['classes'] = {'answer':[0, 1]}
    (tmp_path / 'task_spec.json').write_text(json.dumps(task))
    c = Controller(path, Limits(seconds=180, cpu=1), run_id='catboost_weight_worker')
    hypothesis = dict(observation='Numeric labels are confirmed by this offline fixture',
        expected_effect='Keep label identity through JSON and a separate worker',
        changed_component='class_weights', acceptance='Valid probabilities or a preflight constraint')
    try:
        c.prepare()
        bad = payload(c.dispatch(ToolCommand(operation_id='unknown_catboost_weight', tool='train_model',
            arguments=dict(pipeline=dict(model='catboost', hyperparameters=dict(class_weights={'UNKNOWN':4})),
                hypothesis=hypothesis, budget_seconds=40))))
        assert bad.get('status') == 'rejected', bad
        good = payload(c.dispatch(ToolCommand(operation_id='valid_catboost_weight', tool='train_model',
            arguments=dict(pipeline=dict(model='catboost', hyperparameters=dict(iterations=5,
                class_weights={'0':1, '1':4})), hypothesis=hypothesis, budget_seconds=40))))
        assert good['result']['status'] == 'completed', good
        _, saved = c.lookup(good['experiment']['experiment_id'])
        with (Path(saved.artifacts['directory']) / 'model.pkl').open('rb') as stream:
            p = cloudpickle.load(stream)
        assert p.models['answer'].get_params()['class_weights'] == {0:1, 1:4}
        assert c.budget.counter('llm_calls') == 0 and not c.journal.get('control_opened')
    finally:
        c.close()


@pytest.mark.parametrize('labels', [[0, 1], ['no', 'yes']])
def test_native_positional_weights_remain_replayable(tmp_path, labels):
    task = task_for(labels)
    X = pd.DataFrame({'x':np.linspace(-2, 2, 80)}, index=[f'r{i}' for i in range(80)])
    y = pd.DataFrame({'answer':np.where(X.x > 0, labels[1], labels[0])}, index=X.index)
    spec = PipelineSpec(model='catboost', hyperparameters={'iterations':5, 'class_weights':[1, 4]})
    p = build_pipeline(spec, task, {})
    p.fit(X, y, context(task, tmp_path))
    assert p.models['answer'].get_params()['class_weights'] == [1, 4]
    assert np.isfinite(p.predict(X.iloc[:4], context(task, tmp_path)).values['answer']).all()
