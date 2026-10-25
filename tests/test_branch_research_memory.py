"""Structural coverage and evidence projection, independent of gateway billing."""
import json
from types import SimpleNamespace

import pytest

from automl_agent.contracts import PipelineSpec, MetricSpec, SelectionPolicy
from automl_agent.research import experiment_review
from automl_agent.research_memory import branch_key, project
from test_research_learning import trial


@pytest.mark.parametrize('change', [
    {'representation': 'native'}, {'include_features': ['x']},
    {'transformations': [{'name': 'standard_scale'}]},
    {'imputation': {'numeric': 'knn'}}, {'target_transform': 'log1p'},
    {'prediction_policy': 'threshold', 'postprocess_fit_policy': 'inner_holdout'},
    {'shift_strategy': 'importance_weighting'},
])
def test_structural_choices_are_distinct_from_family(change):
    base = PipelineSpec(model='ridge')
    other = PipelineSpec.model_validate({**base.model_dump(), **change})
    assert branch_key(base) != branch_key(other)


def test_hpo_seed_and_fitted_state_do_not_invent_new_structural_branches():
    base = PipelineSpec(model='ridge', hyperparameters={'alpha': 1})
    other = base.model_copy(update={'hyperparameters': {'alpha': 10}, 'seed': 7,
        'learned_inverse': {'secret': 'not_an_identity'}})
    assert branch_key(base) == branch_key(other)


def test_same_family_with_other_preprocessing_remains_available(monkeypatch):
    from automl_agent import directions, completion
    s, r = trial('tested', 1)
    s = s.model_copy(update={'pipeline': PipelineSpec(model='ridge',
        transformations=[{'name': 'spline'}])})
    monkeypatch.setattr(directions, 'cards', lambda *_: {'plain': {'id': 'plain',
        'pipeline': PipelineSpec(model='ridge').model_dump(), 'applicable': True, 'apply': {}}})
    monkeypatch.setattr(completion, 'candidate_costs', lambda *_: None)
    c = SimpleNamespace(records=lambda: [(s, r)], plan=SimpleNamespace(hash='split', strategy='random'),
        task=None, dataset=SimpleNamespace(test=[]), finalist_costs=lambda: [],
        budget=SimpleNamespace(remaining=1000, final_reserve=lambda _: 10))
    assert directions.direction_map(c)['affordable_unattempted'][0]['id'] == 'plain'
    c.records = lambda: [(s.model_copy(update={'pipeline': PipelineSpec(model='ridge')}), r)]
    assert not directions.direction_map(c)['affordable_unattempted']


def rows_and_controller():
    task = SimpleNamespace(hash='task', metric=MetricSpec(name='rmse'))
    versions = {'scikit-learn': '1.5.2'}
    rows, old = [], []
    for i, score in enumerate([5., 8., 6., 4.]):
        s, r = trial('e' + str(i), score)
        if i == 2:
            r = r.model_copy(update={'status': 'timeout', 'score': None})
        review = experiment_review(s, r, old, task, SelectionPolicy(maximize=False))
        rows.append({'spec': s, 'review': review, 'id': 'review_' + str(i), 'sequence': i})
        old.append((s, r))
    c = SimpleNamespace(task=task, dataset=SimpleNamespace(source_hash='data'),
        plan=SimpleNamespace(hash='split'), capabilities=SimpleNamespace(versions=versions),
        context_memory=SimpleNamespace(store=SimpleNamespace(active_note=lambda: {
            'related_ids': ['e1'], 'question': 'regularization'})))
    return rows, c


def test_worse_hpo_variant_does_not_erase_best_measured_score(monkeypatch):
    from automl_agent import directions
    first, good = trial('best', 1.)
    second, worse = trial('latest', 3.)
    second = second.model_copy(update={'pipeline': first.pipeline.model_copy(
        update={'hyperparameters': {'alpha': 10}})})
    monkeypatch.setattr(directions, 'cards', lambda *_: {})
    c = SimpleNamespace(records=lambda: [(first, good), (second, worse)],
        plan=SimpleNamespace(hash='split', strategy='random'),
        task=SimpleNamespace(metric=MetricSpec(name='rmse')))
    rows = directions.direction_map(c)['directions']
    assert len(rows) == 1
    assert rows[0]['experiment_id'] == 'best'
    assert rows[0]['score'] == 1.
    assert rows[0]['latest_score'] == 3.
    assert rows[0]['variant_count'] == 2


def test_selected_branch_has_negative_result_and_at_most_two_comparable_experiences():
    rows, c = rows_and_controller()
    result = project(c, rows)
    assert result['current']['experiment_id'] == 'e1'
    assert result['current']['outcome'] == 'no_material_gain'
    assert len(result['previous']) == 2
    assert any(row['outcome'] == 'technical_failure' for row in result['previous'])
    assert all(row['tested_design']['source'] == 'agent_assumption' for row in result['previous'])
    assert rows[1]['review']['learning']['interpretation'] == 'unknown'


def test_changed_protocol_or_environment_cannot_leak_into_current_memory():
    rows, c = rows_and_controller()
    for field, value in [('validation_hash', 'closed'), ('data_version', 'elsewhere'),
                         ('task_hash', 'another'), ('library_versions', {})]:
        altered = [{**row, 'spec': row['spec'].model_copy(update={field: value})} for row in rows]
        result = project(c, altered)
        assert result['current'] is None and result['previous'] == []


def test_reviews_exclude_private_payloads_and_leave_acceptance_unknown():
    s, r = trial('private', 3)
    r = r.model_copy(update={'artifacts': {'labels': 'PRIVATE_SENTINEL'},
        'feedback': {'labels': 'PRIVATE_SENTINEL'}, 'training_summary': {'predictions': 'PRIVATE_SENTINEL'}})
    review = experiment_review(s, r, [], SimpleNamespace(metric=MetricSpec(name='rmse')),
        SelectionPolicy(maximize=False))
    assert 'PRIVATE_SENTINEL' not in json.dumps(review)
    assert review['learning']['acceptance'] == 'not_established_by_numeric_score'


def test_compact_memory_keeps_measured_diff_separate_from_declared_single_change():
    _, c = rows_and_controller()
    parent, old = trial('parent', 2., pipeline=PipelineSpec(model='ridge', hyperparameters={'alpha': 1}))
    child, measured = trial('child', 1., parent_id='parent',
        pipeline=PipelineSpec(model='lightgbm', hyperparameters={'n_estimators': 5}))
    review = experiment_review(child, measured, [(parent, old)], c.task, SelectionPolicy(maximize=False))
    c.context_memory.store.active_note = lambda: {'related_ids': ['child']}
    memory = project(c, [{'spec': child, 'review': review, 'id': 'review_child', 'sequence': 1}])
    facts = memory['current']['change_facts']
    assert facts['basis'] == 'normalized_parent_recipe'
    assert facts['field_count'] == len(review['effective_change']['effective_fields']) > 1
    assert facts['mixed'] is True
    assert memory['current']['tested_design']['source'] == 'agent_assumption'


@pytest.mark.parametrize('mode', ['native', 'json'])
def test_real_controller_delivers_negative_memory_after_maximum_compression(tmp_path, monkeypatch, mode):
    from automl_agent.controller import Controller
    from automl_agent.budget import Limits
    from conftest import write_task
    from test_layered_prompt_context import read_all
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1), run_id='branch_projection')
    c.no_llm = True
    c.qwen.disabled = True
    try:
        c.prepare()
        s, r = c.lookup(c.journal.get('baseline_id'))
        s = s.model_copy(update={'experiment_id': 'negative', 'parent_id': s.experiment_id, 'status': 'full'})
        r = r.model_copy(update={'experiment_id': 'negative', 'score': r.score + 100, 'scope': 'full',
            'training_summary': {**r.training_summary, 'hidden_labels': 'PRIVATE_MEMORY_SENTINEL'}})
        c.journal.save_experiment(s, r)
        snapshot = c.context_memory.snapshot(c.system_prompt)
        for stage in range(4):
            prepared = c.context_memory.compressor.fit(snapshot, mode, budget=262144, minimum_stage=stage)
            wire = json.loads(prepared.wire_json)
            context = json.loads(wire['messages'][1]['content'])
            memory = context['core']['current_work']['research_memory']
            if 'read' in memory:
                memory = json.loads(read_all(c, memory['read']))
            assert memory['current']['experiment_id'] == 'negative'
            assert memory['current']['outcome'] == 'no_material_gain'
            assert memory['current']['comparator_id'] == s.parent_id
            assert len(memory['previous']) <= 2
            assert 'PRIVATE_MEMORY_SENTINEL' not in json.dumps(memory)
        assert c.journal.get('control_status', 'not_opened') == 'not_opened'
    finally:
        c.close()


@pytest.mark.parametrize('version', [2, 3, 4, 5])
def test_saved_prompt_is_not_silently_upgraded(tmp_path, version):
    from automl_agent.prompting import bind_prompt
    from automl_agent.journal import Journal
    from automl_agent.util import digest
    j = Journal(tmp_path)
    try:
        predecessor = {'version': version, 'system': 'Pinned predecessor', 'system_hash': digest('Pinned predecessor'),
            'guides': {}, 'guides_hash': digest({}), 'sections': {}}
        j.put('llm_prompt_contract', predecessor)
        assert bind_prompt(j, 'legacy', existing_run=True) == predecessor
    finally:
        j.close()
