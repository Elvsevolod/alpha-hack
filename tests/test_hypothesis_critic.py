"""The critic gates execution, preserves learning scopes, and uses real transport contracts."""
import json
import time

import httpx
import pytest

from automl_agent.budget import Limits
from automl_agent.completion import CompletionCosts
from automl_agent.contracts import ExperimentResult, Hypothesis, PipelineSpec, ToolCommand
from automl_agent.controller import Controller
from automl_agent.critic import Critique, review_candidate
from automl_agent.qwen import QwenClient, QwenOutcomeUnknown
from automl_agent.tool_results import payload
from conftest import write_task


REAL_CRITIQUE = QwenClient.critique


def answer(request, verdict='approve'):
    return {'review_id': request['review_id'], 'verdict': verdict,
        'evaluation_scope': 'research_validation',
        'comparator_id': next((item['experiment_id'] for item in request.get('comparators', [])
            if item['status'] == 'completed' and item['scope'] != 'pilot' and item['score'] is not None), None),
        'evidence_assessment': 'Compare the proposed representation on the fixed fit data.',
        'comparison_assessment': 'Use the saved baseline on the same protocol.',
        'leakage_assessment': 'Learn preprocessing inside fit and inner folds.',
        'cost_assessment': 'The requested experiment has a hard wall cap.',
        'falsification_assessment': 'Reject the hypothesis if primary validation RMSE fails to improve.',
        'revision_requests': ['Specify one changed component and its comparator.'] if verdict == 'revise' else []}


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-critic-fixture')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://fixture.invalid/v1')
    monkeypatch.delenv('OPENAI_EXTRA_BODY', raising=False)
    c = Controller(write_task(tmp_path), Limits(seconds=600, cpu=1), run_id='critic')
    c.prepare()
    yield c
    c.close()


def command(identifier='candidate', seconds=120, observation='Nonlinear residuals may remain in fit.'):
    return ToolCommand(operation_id=identifier, tool='train_model', arguments={
        'pipeline': PipelineSpec(model='extra_trees', hyperparameters={'n_estimators': 5}).model_dump(mode='json'),
        'hypothesis': Hypothesis(observation=observation, expected_effect='Reduce RMSE against the incumbent.',
            changed_component='model family', acceptance='Same protocol; accept only a lower validation RMSE.').model_dump(),
        'budget_seconds': seconds})


def execution_spy(c, monkeypatch):
    seen = []
    def execute(spec):
        seen.append(spec)
        return ExperimentResult(experiment_id=spec.experiment_id, status='completed', validation_hash=c.plan.hash,
            score=1.0, seconds=.1, scope='full')
    monkeypatch.setattr(c, 'execute_experiment', execute)
    monkeypatch.setattr(c, 'accept_result', lambda *_: None)
    return seen


def test_cheap_candidate_does_not_request_a_review(controller, monkeypatch):
    c = controller
    seen = execution_spy(c, monkeypatch)
    monkeypatch.setattr(c.qwen, 'critique', lambda *_a, **_k: pytest.fail('Cheap fit called critic'))
    result = payload(c.dispatch(command(seconds=30)))
    assert result['result']['status'] == 'completed' and len(seen) == 1
    assert result['critic'] is None and not (c.workdir / 'hypothesis_reviews').exists()


def test_approved_expensive_candidate_runs_after_review(controller, monkeypatch):
    c = controller
    seen = execution_spy(c, monkeypatch)
    calls = []
    def critique(request, reserve_seconds):
        assert not seen and reserve_seconds >= c.budget.final_reserve()
        calls.append(request)
        return answer(request)
    monkeypatch.setattr(c.qwen, 'critique', critique)
    result = payload(c.dispatch(command()))
    assert result['result']['status'] == 'completed' and len(seen) == len(calls) == 1
    assert result['critic']['status'] == 'approved' and 'request' not in result['critic']
    stored = json.loads(next((c.workdir / 'hypothesis_reviews').glob('*.json')).read_text())
    assert stored['request']['candidate']['hypothesis'] == command().arguments['hypothesis']
    assert stored['status'] == 'approved'


def test_revision_preserves_incumbent_and_can_be_fixed_without_repeat_billing(controller, monkeypatch):
    c = controller
    seen = execution_spy(c, monkeypatch)
    calls = []
    def critique(request, reserve_seconds):
        calls.append(request)
        return answer(request, 'revise' if len(calls) == 1 else 'approve')
    monkeypatch.setattr(c.qwen, 'critique', critique)
    before = len(c.records()), c.budget.counter('candidate_count'), c.publisher.output.read_bytes(), c.plan.hash
    wrapped = c.dispatch(command())
    assert wrapped['status'] == 'hypothesis_revision_required'
    first = payload(wrapped)
    assert first['status'] == 'hypothesis_revision_required' and not first['experiment_started']
    assert not seen and before == (len(c.records()), c.budget.counter('candidate_count'), c.publisher.output.read_bytes(), c.plan.hash)
    assert payload(c.dispatch(command())) == first
    again = payload(c.dispatch(command('same_proposal')))
    assert again['critic']['review_id'] == first['critic']['review_id'] and len(calls) == 1
    fixed = payload(c.dispatch(command('revised', observation='Fit residuals suggest nonlinear effects; change only the model family.')))
    assert fixed['result']['status'] == 'completed' and len(calls) == 2 and len(seen) == 1


@pytest.mark.parametrize('bad', [None, {}, {'evidence_assessment': ''}, {'verdict': 'unknown'},
    {'review_id': 'wrong'}, {'revision_requests': ['Unresolved issue']},
    {'evaluation_scope': 'control'}, {'comparator_id': 'invented_result'}, {'comparator_id': None}])
def test_invalid_critic_cannot_authorize_training(controller, monkeypatch, bad):
    c = controller
    seen = execution_spy(c, monkeypatch)
    def critique(request, reserve_seconds):
        return bad if bad is None or not bad else {**answer(request), **bad}
    monkeypatch.setattr(c.qwen, 'critique', critique)
    result = payload(c.dispatch(command()))
    assert result['status'] == 'hypothesis_review_unavailable' and not seen
    assert c.budget.counter('candidate_count') == 0


@pytest.mark.parametrize('estimate,cap,required', [(10, 180, False), (180, 30, False),
    (120, 180, True), (float('nan'), 120, True), (-1, 120, True), (None, 120, True)])
def test_gate_uses_measured_cost_or_hard_cap(controller, monkeypatch, estimate, cap, required):
    c = controller
    args = c.context_memory.arguments(command(seconds=cap))
    spec = c.make_spec(args, 'cost_check')
    calls = []
    monkeypatch.setattr(c.qwen, 'critique', lambda request, **_k: calls.append(request) or answer(request))
    result = review_candidate(c, spec, CompletionCosts([], {'candidate_fit_seconds': estimate, 'seconds': 17}))
    assert bool(calls) is required and (result is not None) is required


def test_critic_context_does_not_expose_private_results_or_nonfit_profiles(controller, monkeypatch):
    c = controller
    old, measured = c.lookup(c.journal.get('baseline_id'))
    measured.feedback = {'hidden': 'PRIVATE_SENTINEL'}
    measured.artifacts['predictions'] = 'PRIVATE_SENTINEL'
    measured.training_summary['inner_labels'] = 'PRIVATE_SENTINEL'
    c.journal.save_experiment(old, measured)
    profile = c.dataset.profile
    scopes = []
    def fit_profile(indices, scope='fit'):
        scopes.append(indices)
        return profile(indices, scope)
    monkeypatch.setattr(c.dataset, 'profile', fit_profile)
    def critique(request, reserve_seconds):
        wire = json.dumps(request)
        assert 'PRIVATE_SENTINEL' not in wire and 'feedback' not in wire and 'artifacts' not in wire
        assert request['validation']['control'] == 'closed'
        return answer(request)
    monkeypatch.setattr(c.qwen, 'critique', critique)
    spec = c.make_spec(c.context_memory.arguments(command()), 'private_check')
    assert review_candidate(c, spec, [])['status'] == 'approved'
    assert scopes == [c.plan.fit]


def test_review_time_reduces_the_training_allocation(controller, monkeypatch):
    c = controller
    seen = execution_spy(c, monkeypatch)
    def critique(request, reserve_seconds):
        c.budget.deadline = time.time() + 70
        return answer(request)
    monkeypatch.setattr(c.qwen, 'critique', critique)
    result = payload(c.dispatch(command()))
    assert result['result']['status'] == 'completed'
    assert 0 < seen[0].budget_seconds < 70 - c.budget.final_reserve()


def test_unknown_provider_outcome_is_propagated_without_training(controller, monkeypatch):
    c = controller
    seen = execution_spy(c, monkeypatch)
    def fail(*_a, **_k):
        raise QwenOutcomeUnknown('Provider outcome unknown')
    monkeypatch.setattr(c.qwen, 'critique', fail)
    with pytest.raises(QwenOutcomeUnknown):
        c.dispatch(command())
    assert not seen
    stored = json.loads(next((c.workdir / 'hypothesis_reviews').glob('*.json')).read_text())
    assert stored['status'] == 'unavailable'


@pytest.mark.parametrize('mode', ['native', 'json'])
def test_separate_json_transport_is_bounded_accounted_and_restores_research_settings(controller, monkeypatch, mode):
    c = controller
    c.qwen.transport = mode
    c.qwen.select_reasoning('xhigh', 'research_decision')
    monkeypatch.setattr(c.qwen, 'critique', REAL_CRITIQUE.__get__(c.qwen))
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert 'tools' not in body and body['response_format'] == {'type': 'json_object'}
        assert body['reasoning']['effort'] == 'medium'
        assert request.extensions['timeout']['read'] <= 90
        context = json.loads(body['messages'][1]['content'])
        assert 'core' not in context and 'observations' not in context
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(answer(context))}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}})
    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(handler))
    spec = c.make_spec(c.context_memory.arguments(command()), 'transport_check')
    calls = c.budget.counter('llm_calls')
    assert review_candidate(c, spec, [])['status'] == 'approved'
    assert len(seen) == 1 and c.budget.counter('llm_calls') == calls + 1
    assert c.qwen.effort == 'xhigh' and c.qwen.reasoning_reason == 'research_decision'
    assert c.journal.get('token_usage')['total_tokens'] == 150
    assert c.qwen.transport == mode


def test_custom_code_is_visible_and_oversized_context_never_calls_qwen(controller, monkeypatch):
    c = controller
    spec = c.make_spec(c.context_memory.arguments(command()), 'custom_check')
    calls = []
    monkeypatch.setattr(c.qwen, 'critique', lambda request, **_k: calls.append(request) or answer(request))
    assert review_candidate(c, spec, [], custom_code='def fit(): pass')['status'] == 'approved'
    assert calls[0]['custom_code'] == 'def fit(): pass'
    assert review_candidate(c, spec, [], custom_code='x' * 100000)['status'] == 'unavailable'
    assert len(calls) == 1


def test_revise_verdict_requires_actionable_requests():
    with pytest.raises(ValueError, match='actionable'):
        Critique.model_validate({**answer({'review_id': 'check'}), 'verdict': 'revise'})


def test_shrinking_grant_does_not_rebill_the_same_rejected_proposal(controller, monkeypatch):
    c = controller
    seen = execution_spy(c, monkeypatch)
    calls = []
    monkeypatch.setattr(c.qwen, 'critique', lambda request, **_k: calls.append(request) or answer(request, 'revise'))
    first = payload(c.dispatch(command('first_large_request', seconds=600)))
    c.budget.deadline -= 20
    second = payload(c.dispatch(command('same_large_request', seconds=600)))
    assert first['critic']['review_id'] == second['critic']['review_id']
    assert len(calls) == 1 and not seen


def test_extra_annotations_never_change_the_candidate_or_permissions(controller, monkeypatch):
    c = controller
    seen = execution_spy(c, monkeypatch)
    protocol = c.plan.hash
    def critique(request, reserve_seconds):
        return {**answer(request), 'pipeline_override': {'model': 'OTHER_MODEL'},
            'cost_assessment_note': 'ANNOTATION_NOT_DELIVERED_TO_RESEARCHER'}
    monkeypatch.setattr(c.qwen, 'critique', critique)
    result = payload(c.dispatch(command()))
    assert result['critic']['status'] == 'approved' and c.plan.hash == protocol
    assert seen[0].pipeline.model == 'extra_trees'
    assert result['critic']['ignored_response_fields'] == ['cost_assessment_note', 'pipeline_override']
    assert 'ANNOTATION_NOT_DELIVERED_TO_RESEARCHER' not in json.dumps(result)
