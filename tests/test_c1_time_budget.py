"""C1: protect complete refit without real Qwen or long timed experiments."""
import time

import pytest

from automl_agent.budget import Limits
from automl_agent.completion import CompletionCosts, candidate_costs, forecast
from automl_agent.contracts import Hypothesis, PipelineSpec, TrainArgs
from automl_agent.controller import Controller
from conftest import write_task


@pytest.fixture
def controller(tmp_path):
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1))
    c.prepare()
    try:
        yield c
    finally:
        c.close()


def test_search_reserves_the_third_finalist(controller):
    c = controller
    spec, result = c.lookup(c.journal.get('baseline_id'))
    third = spec.model_copy(update={'experiment_id': 'third'})
    c.journal.save_experiment(third, result.model_copy(update={'experiment_id': 'third', 'seconds': 20}))
    c.journal.put('finalist_pool_ids', [spec.experiment_id, 'third'])
    expected = forecast(c, [spec.experiment_id, 'third']).forecast
    assert c.finalist_costs().forecast['seconds'] == expected['seconds']


def test_expensive_measured_candidate_is_rejected_before_training(controller, monkeypatch):
    c = controller
    spec, result = c.lookup(c.journal.get('baseline_id'))
    result.seconds = 50
    result.training_summary.update(rows=len(c.plan.fit), seconds=50, budget={'selected_fit_actual_seconds': 50})
    measured = spec.model_copy(update={'experiment_id': 'costly_history'})
    c.journal.save_experiment(measured, result.model_copy(update={'experiment_id': measured.experiment_id}))
    c.budget.deadline = time.time() + 150
    pipeline = spec.pipeline.model_copy(update={'seed': spec.pipeline.seed + 1})
    args = TrainArgs(pipeline=pipeline, budget_seconds=100, hypothesis=Hypothesis(
        observation='Measured costly recipe', expected_effect='Keep time for complete refit',
        changed_component='seed', acceptance='Reject before starting a worker'))
    operation, _ = c.journal.begin_operation(c.run_id, 'costly', 'train_model', args.model_dump())
    monkeypatch.setattr(c, 'execute_experiment', lambda *a, **kw: pytest.fail('Unaffordable fit started'))
    with pytest.raises(TimeoutError, match='reserve'):
        c.train(args, operation)


def test_unmeasured_recipe_is_piloted_and_rechecked_even_when_capability_verified(controller, monkeypatch):
    c = controller
    spec, _ = c.lookup(c.journal.get('baseline_id'))
    pipeline = spec.pipeline.model_copy(deep=True)
    pipeline.hyperparameters['n_estimators'] = 27
    args = TrainArgs(pipeline=pipeline, budget_seconds=80, hypothesis=Hypothesis(
        observation='Unknown cost of changed parameters', expected_effect='Protect the old CSV',
        changed_component='training_cost', acceptance='Stop after the measured pilot'))
    candidate = c.make_spec(args, 'expensive_after_pilot')
    assert candidate_costs(c, pipeline) is None
    previous = c.publisher.output.read_bytes()
    monkeypatch.setattr(c.capabilities, 'verified', lambda *a: True)
    original = c.executor.run
    calls = []
    def run(directory, request, seconds):
        calls.append(directory.name)
        assert directory.name == 'pilot', 'Full fit started despite unaffordable refit'
        assert request['pipeline']['hyperparameters']['n_estimators'] == 27
        raw = original(directory, request, seconds)
        assert raw['status'] == 'completed', raw
        raw['training_summary'].update(seconds=100, budget={'selected_fit_actual_seconds': 100})
        raw['wall_seconds'] = 100
        return raw
    monkeypatch.setattr(c.executor, 'run', run)
    result = c.execute_experiment(candidate)
    assert calls == ['pilot'] and result.status == 'timeout'
    diagnostic = result.training_summary['diagnostic']
    assert result.seconds > 0
    assert diagnostic['category'] == 'budget_admission'
    assert diagnostic['reason'] == 'pilot_completed_candidate_rejected'
    assert diagnostic['scope'] == 'fit_only_no_external_metric'
    assert diagnostic['pilot_seconds'] == 100
    assert diagnostic['candidate_fit_seconds_required'] > diagnostic['candidate_window_seconds_remaining']
    assert diagnostic['candidate_fit_shortfall_seconds'] > 0
    assert diagnostic['finalization_reserve_seconds'] > 0
    assert diagnostic['candidate_control_seconds'] > 0
    assert diagnostic['candidate_refit_seconds'] > 0
    assert 'not model-quality evidence' in result.error
    public = c.public_result(result)
    assert public['training_summary']['diagnostic']['category'] == 'budget_admission'
    custom_id = 'budget_custom_public_result'
    custom_pipeline = candidate.pipeline.model_copy(update={
        'engine': 'custom', 'custom_hash': 'a' * 64, 'custom_module': 'a' * 64})
    custom_spec = candidate.model_copy(update={'experiment_id': custom_id, 'pipeline': custom_pipeline})
    custom_result = result.model_copy(update={'experiment_id': custom_id})
    c.journal.save_experiment(custom_spec, custom_result)
    assert c.public_result(custom_result)['training_summary']['diagnostic']['category'] == 'budget_admission'
    assert c.publisher.output.read_bytes() == previous and not c.journal.get('control_opened')


def test_pilot_budget_rejection_explains_minimum_candidate_window(controller, monkeypatch):
    c = controller
    spec, _ = c.lookup(c.journal.get('baseline_id'))
    proposed = CompletionCosts([], {'seconds': 10, 'candidate_fit_seconds': .1,
        'candidate_control_seconds': 1, 'candidate_refit_seconds': 2,
        'cost_basis': 'row_scaled_pilot', 'uncertainty': 'row_scaling_is_an_estimate'})
    monkeypatch.setattr('automl_agent.controller.time.time', lambda: 1000.0)
    result = c.pilot_budget_rejection(spec, {'wall_seconds': .1}, proposed,
        1000.15, time.monotonic(), 'pilot_completed_candidate_window_exhausted')
    diagnostic = result.training_summary['diagnostic']
    assert diagnostic['reason'] == 'pilot_completed_candidate_window_exhausted'
    assert diagnostic['candidate_fit_seconds_required'] < diagnostic['candidate_window_seconds_remaining']
    assert diagnostic['minimum_candidate_window_seconds'] == .2
    assert diagnostic['candidate_window_floor_shortfall_seconds'] > 0
    assert 'below the 0.200s minimum start window' in result.error
    assert result.status == 'timeout' and result.seconds >= 0


def test_control_must_also_fit_publication_and_safety(controller, monkeypatch):
    c = controller
    previous = c.publisher.output.read_bytes()
    c.freeze('Control plus refit fit, but their completing operations do not')
    c.budget.deadline = time.time() + 25
    def estimate(controller, identifiers, rows=None, phase=None):
        nodes = {identifier: 10 for identifier in identifiers}
        return {'seconds': sum(nodes.values()), 'node_seconds': nodes, 'disk_bytes': 0}
    monkeypatch.setattr('automl_agent.completion.estimate', estimate)
    monkeypatch.setattr(c, '_prepare_finalist', lambda *a: pytest.fail('Completion margin was passed to a worker'))
    c.finalize()
    assert c.journal.get('control_status') == 'control_skipped_budget'
    assert not c.journal.get('control_opened') and c.publisher.output.read_bytes() == previous


def test_control_worker_cannot_borrow_publication_or_system_margin(controller, monkeypatch):
    c = controller
    c.freeze('Worker allocation protects all following phases')
    original = c._prepare_finalist
    seen = []
    def prepare(identifier, rows, phase, seconds):
        f = forecast(c, c.journal.get('finalization_plan')['finalists']).forecast
        margin = f['control_scoring_and_publication_seconds'] + f['safety_seconds']
        if phase == 'control':
            margin += f['refit']['seconds']
        # Permit the few milliseconds between allocation and this observation.
        assert seconds <= c.budget.remaining - margin + .1
        seen.append(phase)
        return original(identifier, rows, phase, seconds)
    monkeypatch.setattr(c, '_prepare_finalist', prepare)
    c.finalize()
    assert seen == ['control', 'refit']
    assert c.journal.get('control_status') == c.journal.get('refit_status') == 'completed'


def test_proposed_ensemble_counts_shared_dependencies_once(controller):
    c = controller
    spec, result = c.lookup(c.journal.get('baseline_id'))
    member = spec.model_copy(update={'experiment_id': 'member'})
    c.journal.save_experiment(member, result.model_copy(update={'experiment_id': member.experiment_id, 'seconds': 20}))
    inner = PipelineSpec(engine='ensemble', model='uniform_average', members=[spec.experiment_id, member.experiment_id])
    nested = spec.model_copy(update={'experiment_id': 'inner', 'pipeline': inner})
    c.journal.save_experiment(nested, result.model_copy(update={'experiment_id': 'inner', 'seconds': .1}))
    proposed = candidate_costs(c, PipelineSpec(engine='ensemble', model='uniform_average', members=['inner', member.experiment_id]))
    assert set(proposed.forecast['control']['node_seconds']) == {spec.experiment_id, member.experiment_id, 'inner'}
    assert proposed.forecast['candidate_refit_seconds'] > forecast(c, [member.experiment_id]).forecast['refit']['seconds']
