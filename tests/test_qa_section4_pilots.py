from dataclasses import asdict
from itertools import permutations
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from tools.qa_decision_rule_pilot import choose, decide
from tools import qa_pseudo_label_pilot as pseudo


@pytest.fixture
def source_task(tmp_path):
    rng = np.random.default_rng(123)
    X = pd.DataFrame(rng.normal(size=(30, 2)), columns=['x', 'z'], index=[f'train:{i}' for i in range(30)])
    y = pd.Series((X.x > 0).astype(int), index=X.index, name='y')
    X.assign(y=y).rename_axis('id').to_csv(tmp_path / 'labeled.csv')
    pd.DataFrame({'id': ['test:1', 'test:2'], 'x': [1., -1.], 'z': [0., 0.]}).to_csv(tmp_path / 'future.csv', index=False)
    pd.DataFrame({'id': [f'u:{i}' for i in range(10)], 'x': rng.normal(size=10), 'z': rng.normal(size=10)}).to_csv(tmp_path / 'unlabeled.csv', index=False)
    manifest = {'allow_pseudo_labels': True, 'file_roles': {'train': 'labeled.csv', 'test': 'future.csv', 'unlabeled': 'unlabeled.csv'},
                'id_column': 'id', 'target': 'y', 'features': ['x', 'z']}
    return tmp_path, X, y, manifest


@pytest.mark.parametrize('iterations', [0, 2, True, 1.0])
def test_pseudo_has_exactly_one_iteration(iterations):
    with pytest.raises(ValueError, match='one teacher-to-student'):
        pseudo.PseudoRecipe(.8, iterations=iterations)


@pytest.mark.parametrize('threshold,weight', [(float('nan'), .25), (.4, .25), (1.1, .25), (.8, 0), (.8, -1)])
def test_pseudo_rejects_invalid_frozen_settings(threshold, weight):
    with pytest.raises(ValueError):
        pseudo.PseudoRecipe(threshold, weight)


@pytest.mark.parametrize('kind', ['not_approved', 'test_alias', 'path_escape', 'target_present', 'test_id_overlap'])
def test_unlabeled_source_must_be_separate_and_explicit(source_task, kind):
    root, _, _, manifest = source_task
    if kind == 'not_approved':
        manifest['allow_pseudo_labels'] = False
    elif kind == 'test_alias':
        manifest['file_roles']['unlabeled'] = 'future.csv'
    elif kind == 'path_escape':
        manifest['file_roles']['unlabeled'] = '../outside.csv'
    else:
        path = root / 'unlabeled.csv'
        frame = pd.read_csv(path)
        if kind == 'target_present':
            frame['y'] = 0
        else:
            frame.loc[0, 'id'] = 'test:1'
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        pseudo.approved_source(root, manifest)


def test_phase_guard_rejects_rows_before_any_model_fit(source_task, monkeypatch):
    root, X, y, manifest = source_task
    source = pseudo.approved_source(root, manifest)
    monkeypatch.setattr(pseudo, 'estimator', lambda _: pytest.fail('Invalid phase trained a model'))
    with pytest.raises(ValueError, match='phase authority'):
        pseudo.fit_one_iteration(X, y, source, pseudo.PseudoRecipe(.8),
                                 pseudo.PhaseAuthority('research_fit', frozenset(X.index[:20])))


def test_source_mutation_cannot_reuse_previous_approval(source_task, monkeypatch):
    root, X, y, manifest = source_task
    source = pseudo.approved_source(root, manifest)
    source.features.iat[0, 0] += 1
    monkeypatch.setattr(pseudo, 'estimator', lambda _: pytest.fail('Mutated source trained a model'))
    with pytest.raises(ValueError, match='source changed'):
        pseudo.fit_one_iteration(X, y, source, pseudo.PseudoRecipe(.8),
                                 pseudo.PhaseAuthority('research_fit', frozenset(X.index)))


def test_teacher_student_weights_provenance_and_predict_only(source_task, monkeypatch):
    root, X, y, manifest = source_task
    source = pseudo.approved_source(root, manifest)
    observed = []
    original = Pipeline.fit

    def tracked(model, features, labels, **kwargs):
        observed.append({'rows': list(features.index), 'weights': kwargs.get('model__sample_weight')})
        return original(model, features, labels, **kwargs)

    monkeypatch.setattr(Pipeline, 'fit', tracked)
    model = pseudo.fit_one_iteration(X.iloc[:20], y.iloc[:20], source, pseudo.PseudoRecipe(.5),
                                     pseudo.PhaseAuthority('research_fit', frozenset(X.index[:20])))
    assert len(observed) == 2 and model.teacher is not model.student
    assert observed[0]['rows'] == list(X.index[:20])
    assert set(observed[1]['rows']) == set(X.index[:20]) | set(source.features.index)
    assert np.array_equal(observed[1]['weights'], [1.] * 20 + [.25] * 10)
    assert all(row['weight'] == .25 and row['teacher_sha256'] == model.teacher_hash and
               row['source_id'] == source.source_id for row in model.provenance)
    before = len(observed)
    assert np.array_equal(model.predict(X.iloc[20:]), model.predict(X.iloc[20:]))
    assert len(observed) == before


def test_multiclass_rule_preserves_input_and_requires_probabilities():
    probability = np.array([[.45, .5, .05]])
    original = probability.copy()
    assert decide(probability, ['a', 'b', 'c'], [2, 1, 1]).tolist() == ['a']
    assert np.array_equal(probability, original)
    for bad in (np.array([[.2, .2, .2]]), np.array([[np.nan, .5, .5]])):
        with pytest.raises(ValueError):
            decide(bad, ['a', 'b', 'c'], [1, 1, 1])


def test_stricter_tie_pilot_uses_exact_best_in_every_order():
    records = [{'variant': name, 'control_macro_f1': score, 'frozen_cost_seconds': cost, 'created_index': i}
               for i, (name, score, cost) in enumerate([('baseline', .8000, 0), ('cheap', .8009, 1), ('best', .8018, 2)])]
    for ordered in permutations(records):
        assert choose(ordered, .001) == 'cheap'
        assert choose(ordered, .0001) == 'best'


def test_full_train_demo_follows_frozen_control(tmp_path, monkeypatch):
    original = pseudo.fit_one_iteration
    output = tmp_path / 'pilot'
    full_refits = []

    def tracked(features, labels, source, recipe, authority):
        if authority.phase == 'full_train_refit':
            matching = [manifest.parent for manifest in output.glob('*/SOURCE_MANIFEST.json')
                        if pseudo.approved_source(manifest.parent, json.loads(manifest.read_text())).source_id == source.source_id]
            assert len(matching) == 1
            frozen = json.loads((matching[0] / 'CONTROL_FROZEN.json').read_text())
            assert frozen['frozen_recipe_hash'] == pseudo.digest(asdict(recipe))
            full_refits.append(source.source_id)
        return original(features, labels, source, recipe, authority)

    monkeypatch.setattr(pseudo, 'fit_one_iteration', tracked)
    assert pseudo.run(output)['passed'] and len(full_refits) == 3
