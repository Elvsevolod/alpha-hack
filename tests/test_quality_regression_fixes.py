"""Reproduce omitted time roles and fits refused only by a per-attempt estimate."""
import json
import time

import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.completion import CompletionCosts
from automl_agent.contracts import ExecutionPolicy, ExperimentResult, Hypothesis, TaskResolutionProposal, TrainArgs
from automl_agent.controller import Controller
from automl_agent.data import Dataset
from automl_agent.validation import choose_validation
from conftest import write_task


def time_proposal(data, time_column):
    spec = json.loads((data.task_path.parent / 'task_spec.json').read_text())
    spec.update(contract_version=2, source_hash=data.source_hash, time_column=time_column,
        file_roles={name: name for name in ('train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv')
            if name in data.catalog})
    return TaskResolutionProposal.model_validate({'request_id': 'r', 'source_hash': data.source_hash,
        'status': 'proposed', 'task_spec': spec,
        'evidence': {field: ['task', 'inventory'] for field in spec}})


@pytest.mark.parametrize('dictionary', [False, True])
def test_deferred_contract_cannot_drop_a_uniquely_documented_time_role(tmp_path, dictionary):
    path = write_task(tmp_path, temporal=True)
    path.write_text('Synthetic regression. ' + ('' if dictionary else
        '\n| when | Numeric period code | Ordinal year of the observation |\n'))
    if dictionary:
        pd.DataFrame({'feature': ['when'], 'description': ['Year of the observation; anonymized period code']}).to_csv(
            tmp_path / 'data_dictionary.csv', index=False)
    data = Dataset(path, deferred=True)
    with pytest.raises(ValueError, match='time_column.*when'):
        data.confirm_proposal(time_proposal(data, None), 'r')
    assert not hasattr(data, 'train') and data.paths == {}
    corrected = time_proposal(data, 'when')
    corrected.evidence['time_column'] = ['description_csv:data_dictionary.csv'] if dictionary else ['task']
    task = data.confirm_proposal(corrected, 'r')
    data.activate(task)
    plan = choose_validation(data)
    assert plan.strategy == 'temporal'
    assert data.train.iloc[plan.fit]['when'].astype(float).max() < data.train.iloc[plan.validation]['when'].astype(float).min()
    assert data.train.iloc[plan.validation]['when'].astype(float).max() < data.train.iloc[plan.control]['when'].astype(float).min()


@pytest.mark.parametrize('description', [
    'when is the construction year, not an observation year.',
    'when is not a time column; it is an ordinary numeric feature.',
    '| when | Числовой признак | Это не код периода наблюдения |',
])
def test_negated_time_description_does_not_force_a_time_role(tmp_path, description):
    path = write_task(tmp_path, temporal=True)
    path.write_text(description)
    data = Dataset(path, deferred=True)
    assert data.confirm_proposal(time_proposal(data, None), 'r').time_column is None


def test_construction_year_does_not_inherit_observation_role_from_another_column(tmp_path):
    path = write_task(tmp_path, temporal=True)
    for name in ('train.csv', 'test.csv'):
        frame = pd.read_csv(tmp_path / name, dtype=str, keep_default_na=False)
        frame['built'] = 1990
        frame.to_csv(tmp_path / name, index=False)
    spec = json.loads((tmp_path / 'task_spec.json').read_text())
    spec['feature_columns'].append('built')
    (tmp_path / 'task_spec.json').write_text(json.dumps(spec))
    path.write_text('| when | Observation year |\n| built | Year of construction |\n'
        'The columns when and built are expressed on different scales.\n')
    data = Dataset(path, deferred=True)
    with pytest.raises(ValueError, match='time_column.*when'):
        data.confirm_proposal(time_proposal(data, 'built'), 'r')


def test_time_word_in_feature_name_does_not_establish_semantics(tmp_path):
    path = write_task(tmp_path, temporal=True)
    for name in ('train.csv', 'test.csv'):
        frame = pd.read_csv(tmp_path / name, dtype=str, keep_default_na=False).rename(columns={'when': 'timestamp_count'})
        frame.to_csv(tmp_path / name, index=False)
    spec = json.loads((tmp_path / 'task_spec.json').read_text())
    spec['feature_columns'] = ['timestamp_count' if name == 'when' else name for name in spec['feature_columns']]
    (tmp_path / 'task_spec.json').write_text(json.dumps(spec))
    path.write_text('| timestamp_count | Numeric feature | Number of sensor messages |')
    data = Dataset(path, deferred=True)
    assert data.confirm_proposal(time_proposal(data, None), 'r').time_column is None


@pytest.fixture
def controller(tmp_path):
    c = Controller(write_task(tmp_path), Limits(seconds=600, cpu=1))
    c.prepare()
    try:
        yield c
    finally:
        c.close()


def test_affordable_candidate_gets_a_hard_bounded_attempt_after_pessimistic_pilot(controller, monkeypatch):
    c = controller
    baseline, _ = c.lookup(c.journal.get('baseline_id'))
    pipeline = baseline.pipeline.model_copy(deep=True)
    pipeline.hyperparameters['n_estimators'] = 27
    spec = c.make_spec(TrainArgs(pipeline=pipeline, budget_seconds=40, hypothesis=Hypothesis(
        observation='Pilot extrapolation may exceed the requested window', expected_effect='Measure a full bounded fit',
        changed_component='capacity', acceptance='Keep finalization protected')), 'bounded_after_pilot')
    proposal = CompletionCosts([], {'seconds': 60, 'candidate_fit_seconds': 80,
        'candidate_control_seconds': 20, 'candidate_refit_seconds': 30,
        'cost_basis': 'row_and_search_scaled_pilot', 'uncertainty': 'estimate'})
    estimates = iter([None, proposal, proposal])
    monkeypatch.setattr('automl_agent.completion.candidate_costs', lambda *a: next(estimates))
    original = c.executor.run
    calls = []
    def run(directory, request, seconds):
        calls.append((directory.name, seconds))
        assert seconds <= 40
        assert time.time() + seconds <= c.budget.deadline - c.budget.final_reserve(proposal) + .1
        return original(directory, request, seconds)
    monkeypatch.setattr(c.executor, 'run', run)
    result = c.execute_experiment(spec)
    assert result.status == 'completed', result.error
    assert [name for name, _ in calls][:3] == ['performance_1', 'performance_2', spec.experiment_id]
    assert not c.journal.get('control_opened')


def test_measured_recipe_with_short_request_is_attempted_without_extending_request(controller, monkeypatch):
    c = controller
    baseline, measured = c.lookup(c.journal.get('baseline_id'))
    proposal = CompletionCosts([], {'seconds': 60, 'candidate_fit_seconds': 80})
    monkeypatch.setattr('automl_agent.completion.candidate_costs', lambda *a: proposal)
    observed = []
    def execute(spec, automatic_pilot=True):
        observed.append(spec.budget_seconds)
        return measured.model_copy(update={'experiment_id': spec.experiment_id})
    monkeypatch.setattr(c, 'execute_experiment', execute)
    monkeypatch.setattr(c, 'accept_result', lambda *a: None)
    args = TrainArgs(pipeline=baseline.pipeline.model_copy(update={'seed': 987}), budget_seconds=20,
        hypothesis=Hypothesis(observation='A prior estimate exceeds a local request', expected_effect='Try within the hard cap',
            changed_component='seed', acceptance='No budget extension'))
    operation, _ = c.journal.begin_operation(c.run_id, 'short_measured', 'train_model', args.model_dump())
    c.train(args, operation)
    assert observed == [20]


def multiclass_task(root, metric='f1'):
    path = write_task(root, kind='classification')
    frame = pd.read_csv(root / 'train.csv', dtype=str, keep_default_na=False)
    frame['answer'] = ['low', 'middle', 'high'] * (len(frame) // 3)
    frame.to_csv(root / 'train.csv', index=False)
    spec = json.loads((root / 'task_spec.json').read_text())
    spec['metric'] = {'name': metric, 'average': 'macro'}
    (root / 'task_spec.json').write_text(json.dumps(spec))
    return path


@pytest.mark.parametrize('metric,expected', [('f1', True), ('accuracy', False)])
def test_metric_decision_challenger_is_bounded_preserves_anchor_and_resumes(tmp_path, monkeypatch, metric, expected):
    c = Controller(multiclass_task(tmp_path, metric), Limits(seconds=600, cpu=1), run_id='metric_challenger')
    calls = []
    def execute(spec, automatic_pilot=True):
        calls.append(spec)
        return ExperimentResult(experiment_id=spec.experiment_id, status='completed', validation_hash=spec.validation_hash,
            score=.5, scope=spec.status)
    monkeypatch.setattr(c.executor, 'probe', lambda *a: ExecutionPolicy(writable_directories=[], tests={}, limitations=[]))
    monkeypatch.setattr(c, 'execute_experiment', execute)
    monkeypatch.setattr(c, 'accept_result', lambda *a: None)
    monkeypatch.setattr(c, 'finalist_costs', lambda: [])
    try:
        c.prepare()
        assert len(calls) == (2 if expected else 1)
        assert c.journal.get('baseline_id') == 'baseline'
        if expected:
            challenger = calls[1]
            assert challenger.pipeline.prediction_policy == 'class_offsets'
            # A 600 s task leaves room for the more stable pooled OOF; the cheap
            # inner holdout is kept only when the measured baseline leaves no room.
            assert challenger.pipeline.postprocess_fit_policy == 'pooled_oof'
            assert challenger.parent_id == 'baseline' and challenger.budget_seconds <= 180
            assert challenger.pipeline.class_offsets is None
        c.prepare()
        assert len(calls) == (2 if expected else 1)
        assert not c.journal.get('control_opened')
    finally:
        c.close()
