import importlib.util
import hashlib
from pathlib import Path
import sys

import cloudpickle
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_agent.contracts import PipelineSpec, TaskSpec
from automl_agent.evaluation import score_predictions
from automl_agent.pipelines import DirectPipeline


def load(path, name):
    spec = importlib.util.spec_from_file_location('automl_agent.' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import automl_agent.decision as candidate
BASELINE = ROOT / 'tests/fixtures/legacy_decision_v096.py'
assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == '3041947b1d763f0bb503bd063cffac1a122d4233cfa758ec4162385a4158b602'
original = load(BASELINE, '_old_decision_test')


def data(seed=7, k=4, n=500):
    rng = np.random.default_rng(seed)
    classes = [f'c{i}' for i in range(k)]
    z = rng.normal(size=(n, k)); q = np.exp(z - z.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    actual = np.asarray(classes)[[rng.choice(k, p=row) for row in q]]
    z += np.resize([-.3, .2, .4, -.4], k)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    task = TaskSpec(kind='classification', targets=['response'], feature_columns=['x'], classes={'response': classes},
        metric={'name': 'f1', 'average': 'macro'}, output={'columns': ['response'], 'prediction_type': 'labels'}, source_hash='fixture')
    return task, pd.DataFrame({'response': actual}), p


@pytest.mark.parametrize('seed', range(8))
def test_permutation_and_renaming_and_zero_objective(seed):
    task, y, p = data(seed)
    labels = task.classes['response']
    state = candidate.fit_decision(task, y, p, range(len(y)))
    prediction = candidate.decide(p, labels, state)
    raw = score_predictions(task, y, {'response': candidate.decide(p, labels)}, 'labels')
    assert state['training_metric'] - np.sum(np.asarray(state['penalty']) * np.asarray(state['offsets']) ** 2) >= raw - 1e-12
    for order in ([3, 0, 1, 2], [2, 1, 0, 3]):
        classes = [labels[i] for i in order]
        other = task.model_copy(update={'classes': {'response': classes}})
        fitted = candidate.fit_decision(other, y, p[:, order], range(len(y)))
        np.testing.assert_array_equal(candidate.decide(p[:, order], classes, fitted), prediction)
        np.testing.assert_allclose(np.asarray(fitted['offsets'])[np.argsort(order)], state['offsets'], atol=1e-12)
        assert fitted['training_metric'] == pytest.approx(state['training_metric'], abs=1e-15)
    names = {label: f'другое имя {len(labels)-i}' for i, label in enumerate(labels)}
    renamed = task.model_copy(update={'classes': {'response': [names[label] for label in labels]}})
    renamed_y = y.replace({'response': names})
    fitted = candidate.fit_decision(renamed, renamed_y, p, range(len(y)))
    np.testing.assert_array_equal(candidate.decide(p, renamed.classes['response'], fitted), [names[label] for label in prediction])


@pytest.mark.parametrize('seed', range(8))
def test_legacy_is_identical(seed):
    task, y, p = data(seed)
    assert candidate.fit_decision(task, y, p, range(len(y)), method='legacy') == original.fit_decision(task, y, p, range(len(y)), method='legacy')
    assert PipelineSpec().decision_method == 'legacy'


@pytest.mark.parametrize('k', [2, 3, 10])
def test_absent_and_rare_classes_and_bounds(k):
    task, y, p = data(11, k=k)
    if k > 2:
        y.loc[y.response == f'c{k-1}', 'response'] = 'c0'
        y.loc[0, 'response'] = 'c1'
    state = candidate.fit_decision(task, y, p, range(len(y)))
    assert max(map(abs, state['offsets'])) <= 1 and abs(sum(state['offsets'])) < 1e-12
    order = list(reversed(range(k)))
    classes = [task.classes['response'][i] for i in order]
    other = task.model_copy(update={'classes': {'response': classes}})
    fitted = candidate.fit_decision(other, y, p[:, order], range(len(y)))
    np.testing.assert_array_equal(candidate.decide(p, task.classes['response'], state), candidate.decide(p[:, order], classes, fitted))


@pytest.mark.parametrize('k', [3, 10])
def test_exact_ties_are_explicit(k):
    task, y, _ = data(k=k, n=k*10)
    y.response = np.tile(task.classes['response'], 10)
    p = np.full((len(y), k), 1/k)
    state = candidate.fit_decision(task, y, p, range(len(y)))
    assert state['canonical_key_ties'] == k-1 and state['final_argmax_tie_rows'] == len(y)
    assert set(candidate.decide(p, task.classes['response'], state)) == {task.classes['response'][0]}


@pytest.mark.parametrize('invalid', ['shape', 'rows', 'nan', 'inf'])
def test_invalid_shapes_and_nonfinite_remain_errors(invalid):
    task, y, p = data()
    if invalid == 'shape': p = p[:, 0]
    elif invalid == 'rows': p = p[:-1]
    else: p[0, 0] = np.nan if invalid == 'nan' else np.inf
    with pytest.raises(ValueError):
        candidate.fit_decision(task, y, p, range(len(y)))


def test_mixed_label_types_preserve_exact_mapping():
    task, y, p = data(k=3)
    labels = [1, 'class', 2.5]
    mapping = dict(zip(task.classes['response'], labels))
    task = task.model_copy(update={'classes': {'response': labels}})
    y.response = y.response.map(mapping)
    state = candidate.fit_decision(task, y, p, range(len(y)))
    prediction = candidate.decide(p, labels, state)
    assert prediction.dtype == object and set(prediction) <= set(labels) and state['classes'] == labels
    order = [2, 0, 1]; reordered = [labels[i] for i in order]
    fitted = candidate.fit_decision(task.model_copy(update={'classes': {'response': reordered}}), y, p[:, order], range(len(y)))
    np.testing.assert_array_equal(candidate.decide(p[:, order], reordered, fitted), prediction)


def test_quantized_probability_ties_cannot_reduce_final_objective():
    rng = np.random.default_rng(6)
    raw = rng.integers(1, 5, size=(120, 3))
    probability = raw / raw.sum(1, keepdims=True)
    classes = ['a', 'b', 'c']
    actual = np.asarray(classes)[[rng.choice(3, p=row) for row in probability]]
    task = TaskSpec(kind='classification', targets=['response'], feature_columns=['x'], classes={'response': classes},
        metric={'name': 'f1'}, output={'columns': ['response'], 'prediction_type': 'labels'}, source_hash='ties')
    y = pd.DataFrame({'response': actual})
    baseline = score_predictions(task, y, {'response': candidate.decide(probability, classes)}, 'labels')
    state = candidate.fit_decision(task, y, probability, range(len(y)))
    objective = state['training_metric'] - np.sum(np.asarray(state['penalty']) * np.asarray(state['offsets']) ** 2)
    assert objective >= baseline - 1e-12
    assert state['zero_objective_guard_applied']


@pytest.mark.parametrize('fit_policy', ['inner_holdout', 'pooled_oof'])
def test_frozen_refit_keeps_learned_state(tmp_path, monkeypatch, fit_policy):
    task, y, _ = data(k=3, n=180)
    rng = np.random.default_rng(5)
    X = pd.DataFrame({'x': rng.normal(size=len(y))}, index=[f'row{i}' for i in range(len(y))])
    y.index = X.index
    monkeypatch.setattr('automl_agent.decision.fit_decision', candidate.fit_decision)
    monkeypatch.setattr('automl_agent.decision.decide', candidate.decide)
    spec = PipelineSpec(model='logistic', prediction_policy='class_offsets', decision_method='regularized', postprocess_fit_policy=fit_policy)
    context = {'strategy': 'random', 'protocol_hash': 'fixed', 'cpu': 1, 'memory_mb': 1024, 'budget_seconds': 30,
        'workdir': str(tmp_path), 'task': task.model_dump(mode='json'), 'split_id': 'fit'}
    model = DirectPipeline(spec, task); model.fit(X.iloc[:150], y.iloc[:150], context)
    frozen = PipelineSpec.model_validate_json(model.frozen_spec().model_dump_json())
    state = frozen.learned_decision['response']
    assert state['canonical_order'] and state['classes'] == task.classes['response']
    model = cloudpickle.loads(cloudpickle.dumps(DirectPipeline(frozen, task)))
    monkeypatch.setattr('automl_agent.decision.fit_decision', lambda *a, **k: pytest.fail('Frozen refit must not tune offsets'))
    model.fit(X, y, {**context, 'execution_mode': 'refit'})
    assert model.postprocessors['response'] == state
    assert set(model.predict(X.iloc[-20:], context).values['response']) <= set(task.classes['response'])
