"""Historical rejected hypothesis, not the production canonical implementation.

Run explicitly after restoring the original candidate; never substitute current
runtime code for an unavailable historical implementation.
"""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from test_ml_M1_M8_fixes import class_data
from automl_agent.decision import decide


@pytest.fixture
def canonical_decision():
    path = ROOT / 'outputs/ml_M1_M8_v096_20261025/m4_candidate/decision.py'
    if not path.is_file():
        pytest.fail('Historical rejected candidate absent; restore its original artifact before this research audit')
    spec = importlib.util.spec_from_file_location('automl_agent._m4_candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


@pytest.mark.parametrize('method', ['legacy', 'regularized'])
@pytest.mark.parametrize('k', [2, 3, 10])
def test_M4_sparse_classes_and_zero_objective(method, k, canonical_decision):
    from automl_agent.evaluation import score_predictions
    task, y, p = class_data(11, k=k, n=100)
    if k > 2:
        y.loc[y.response == f'c{k-1}', 'response'] = 'c0'
    state = canonical_decision.fit_decision(task, y, p, range(len(y)), method=method)
    baseline = score_predictions(task, y, {'response': decide(p, task.classes['response'])}, 'labels')
    objective = state['training_metric'] - np.sum(np.asarray(state['penalty']) * np.asarray(state['offsets']) ** 2)
    assert objective >= baseline - 1e-12 and np.isfinite(state['offsets']).all()
    if method == 'regularized':
        assert state['permutation_guarantee'] == 'distinct_fit_keys_and_no_final_argmax_ties'
