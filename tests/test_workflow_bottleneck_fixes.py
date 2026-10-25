import copy
import json

import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import PipelineSpec
from automl_agent.controller import Controller
from automl_agent.data import ContractUnresolved
from automl_agent.qwen import QwenRetryableError, QwenOutcomeUnknown
from automl_agent.research_memory import branch_key, identity_version
from automl_agent.resolution_feedback import ResolutionFeedback
from conftest import write_task
from test_qa_B1_source_semantics import proposed
from test_layered_prompt_context import controller


@pytest.mark.parametrize('model,key,first,second', [
    ('catboost', 'class_weights', [1, 1], [1, 3]),
    ('catboost', 'auto_class_weights', 'Balanced', 'SqrtBalanced'),
    ('lightgbm', 'class_weight', {'a': 1, 'b': 1}, {'a': 1, 'b': 3}),
    ('xgboost', 'scale_pos_weight', 1., 3.),
    ('logistic', 'class_weight', None, 'balanced'),
])
def test_all_declared_weighting_controls_define_distinct_structural_variants(model, key, first, second):
    a = PipelineSpec(model=model, hyperparameters={key: first})
    b = PipelineSpec(model=model, hyperparameters={key: second})
    assert branch_key(a) != branch_key(b)


def test_legacy_weighting_key_and_new_hpo_grouping_are_stable():
    from types import SimpleNamespace
    a = PipelineSpec(model='catboost', hyperparameters={'class_weights': [1, 1], 'depth': 4})
    b = a.model_copy(update={'hyperparameters': {'class_weights': [1, 3], 'depth': 4}})
    hpo = a.model_copy(update={'hyperparameters': {'class_weights': [1, 1], 'depth': 8}, 'seed': 13})
    assert branch_key(a, version=1) == branch_key(b, version=1)
    assert branch_key(a) == branch_key(hpo)
    assert identity_version(SimpleNamespace(prompt_contract={'version': 5})) == 1
    assert identity_version(SimpleNamespace(prompt_contract={'version': 6})) == 2


@pytest.mark.parametrize('version,count', [(5, 1), (6, 2)])
def test_weighted_branch_identity_agrees_in_coverage_reviews_and_context(monkeypatch, version, count):
    from automl_agent import directions
    from automl_agent.research import experiment_review
    from automl_agent.research_memory import project
    from automl_agent.contracts import SelectionPolicy
    from test_branch_research_memory import rows_and_controller
    from test_research_learning import trial
    _, c = rows_and_controller()
    c.prompt_contract = {'version': version}
    c.context_memory.store.active_note = lambda: {'related_ids': ['weighted']}
    records = [trial('plain', 2., pipeline=PipelineSpec(model='catboost', hyperparameters={'class_weights': [1, 1]})),
        trial('weighted', 1., pipeline=PipelineSpec(model='catboost', hyperparameters={'class_weights': [1, 3]}))]
    c.records = lambda: records
    c.plan.strategy = 'random'
    monkeypatch.setattr(directions, 'cards', lambda *_: {})
    mapping = directions.direction_map(c)
    assert len(mapping['directions']) == count
    rows = []
    for i, (s, r) in enumerate(records):
        review = experiment_review(s, r, records[:i], c.task, SelectionPolicy(maximize=False),
            branch_version=identity_version(c))
        rows.append({'spec': s, 'review': review, 'id': 'review_' + str(i), 'sequence': i})
    current = project(c, rows)['current']
    assert current['branch_id'] == rows[-1]['review']['branch_id']
    assert current['branch_id'] == branch_key(records[-1][0].pipeline, version=identity_version(c))


def test_transport_error_cannot_erase_verifier_error_or_preserve_stale_verifier_errors():
    f = ResolutionFeedback('same-request')
    f.record(ValueError('first field needs evidence'), validated_proposal=True)
    f.record(QwenRetryableError('known truncation'), validated_proposal=False)
    assert f.payload() == {'validation_errors': ['first field needs evidence'], 'response_error': 'known truncation'}
    restored = ResolutionFeedback('same-request', f.state())
    assert restored.errors == f.errors
    restored.record(ValueError('another field needs evidence'), validated_proposal=True)
    assert restored.payload() == {'validation_errors': ['another field needs evidence'], 'response_error': None}
    assert ResolutionFeedback('different-request', restored.state()).errors == []


def test_parameter_repair_is_bounded_and_uses_the_pinned_public_contract(monkeypatch):
    from types import SimpleNamespace
    from automl_agent import decision_context
    from automl_agent.pipelines import UnsupportedParameters
    c = SimpleNamespace(prompt_contract={'version': 6,
        'guides': {'runtime_parameters': json.dumps({'models': {'ridge': ['alpha']}})}},
        context_memory=SimpleNamespace(store=None))
    monkeypatch.setattr(decision_context, 'guide_route', lambda *_: {'tool': 'read_run_records', 'arguments': {'ids': ['pinned']}})
    bad = UnsupportedParameters('ridge', ['x' * 10000 + str(i) for i in range(20)])
    repair = decision_context.parameter_repair(c, bad)
    assert len(json.dumps(repair).encode()) < 2000
    assert repair['unsupported_total'] == 20 and repair['unsupported_names_truncated']
    assert repair['accepted_parameter_names'] == ['alpha']
    c.prompt_contract['version'] = 5
    assert decision_context.parameter_repair(c, bad) is None


@pytest.mark.parametrize('version', [5, 6])
def test_semantic_rejection_then_known_truncation_delivers_error_until_corrected(
        tmp_path, monkeypatch, version):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    c = Controller(write_task(tmp_path), Limits(seconds=90, cpu=1))
    c.prompt_contract = {**c.prompt_contract, 'version': version}
    c.qwen.key, c.qwen.base_url = 'offline', 'http://fixture.invalid/v1'
    monkeypatch.setattr(c.dataset, 'local_contract', lambda: None)
    requests = []
    def resolve(system, request):
        requests.append(copy.deepcopy(request))
        if len(requests) == 2:
            raise QwenRetryableError('Truncated model response')
        reply = proposed(c.dataset, request['request_id'], metric='rmse')
        if len(requests) == 1:
            reply['task_spec']['explicit_validation'] = 'random'
            reply['evidence']['explicit_validation'] = ['task']
        return reply
    monkeypatch.setattr(c.qwen, 'resolve', resolve)
    try:
        task = c._task()
        assert len(requests) == 3 and task.explicit_validation is None
        assert 'explicit_validation' in requests[1]['validation_errors'][0]
        if version == 6:
            assert requests[2]['validation_errors'] == requests[1]['validation_errors']
            assert requests[2]['response_error'] == 'Truncated model response'
        else:
            assert requests[2]['validation_errors'] == ['Truncated model response']
        assert 'response_error' not in json.dumps(task.model_dump())
    finally:
        c.close()


def test_unknown_outcome_is_not_retried_to_repair_the_contract(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    c = Controller(write_task(tmp_path), Limits(seconds=90, cpu=1))
    c.qwen.key, c.qwen.base_url = 'offline', 'http://fixture.invalid/v1'
    monkeypatch.setattr(c.dataset, 'local_contract', lambda: None)
    calls = []
    def resolve(*_):
        calls.append(1)
        raise QwenOutcomeUnknown('No known outcome')
    monkeypatch.setattr(c.qwen, 'resolve', resolve)
    try:
        with pytest.raises(ContractUnresolved, match='unavailable'):
            c._task()
        assert len(calls) == 1
    finally:
        c.close()


def decoded_objects(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from decoded_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from decoded_objects(item)
    elif isinstance(value, str):
        try:
            obj = json.loads(value)
        except (ValueError, TypeError):
            return
        if isinstance(obj, (dict, list)):
            yield from decoded_objects(obj)


@pytest.mark.parametrize('mode', ['native', 'json'])
@pytest.mark.parametrize('version,expected_reads', [(5, 2), (6, 0)])
def test_http_contract_repair_avoids_catalog_reads_and_finishes_csv(controller, mode, version, expected_reads):
    """Same deterministic policy, actual dispatch/fit/finish, no provider calls.

    This compares delivery paths, not the independent judgment of real Qwen.
    """
    import httpx
    from automl_agent.contracts import ToolCommand
    from automl_agent.tool_results import payload
    c = controller
    c.prompt_contract = {**c.prompt_contract, 'version': version}
    c.qwen.context_tokens = 262144
    c.qwen.transport, c.qwen.key, c.qwen.base_url = mode, 'offline', 'http://fixture.invalid/v1'
    h = {'observation': 'The adapter rejected the requested parameter.',
        'changed_component': 'hyperparameters.num_leaves', 'expected_effect': 'Fit a supported declared recipe.',
        'acceptance': 'Execution and fixed-protocol score; no optimality claim.'}
    bad = payload(c.dispatch(ToolCommand(operation_id='unsupported', tool='train_model', arguments={
        'pipeline': {'model': 'lightgbm', 'hyperparameters': {'not_a_parameter': 'VALUE_SENTINEL'}},
        'hypothesis': h, 'budget_seconds': 45})))
    assert bad['status'] == 'rejected' and 'Unsupported parameters' in bad['error']
    assert 'VALUE_SENTINEL' not in json.dumps(bad)
    if version == 6:
        assert 'num_leaves' in bad['repair']['accepted_parameter_names']
        assert 'n_jobs' not in bad['repair']['accepted_parameter_names']
    else:
        assert 'repair' not in bad
    c.journal.put('last_tool_result', bad)
    c.journal.put('last_research_tool', 'train_model')
    actions = []
    def handler(request):
        wire = json.loads(request.content)
        context = json.loads(wire['messages'][1]['content'])
        assert (len(wire['tools']) if mode == 'native' else len(context['tool_argument_schemas'])) == 17
        objects = list(decoded_objects(context))
        if 'train_model' in actions:
            tool, args = 'finish', {'reason': 'Offline adapter-repair route verified; quality gains remain unmeasured.',
                'result_ids': ['baseline']}
        elif any('accepted_parameter_names' in x or x.get('source') == 'controller_adapter_allowlists' for x in objects):
            tool, args = 'train_model', {'parent_id': 'baseline', 'search_action': 'improve',
                'pipeline': {'model': 'lightgbm', 'hyperparameters': {'num_leaves': 7}},
                'hypothesis': h, 'budget_seconds': 45}
        elif any('guides' in x for x in objects):
            catalog = next(x for x in objects if 'guides' in x)
            route = next(x['read'] for x in catalog['guides'] if x['id'] == 'runtime_parameters')
            tool, args = route['tool'], route['arguments']
        else:
            route = context['core']['current_work']['local_guides']
            tool, args = route['tool'], route['arguments']
        actions.append(tool)
        assert len(actions) <= 5, 'The scripted route repeated a read without delivered contract facts'
        op = 'repair_' + str(len(actions))
        command = {'operation_id': op, 'tool': tool, 'arguments': args}
        if mode == 'native':
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': op, 'type': 'function',
                'function': {'name': tool, 'arguments': json.dumps({'operation_id': op, **args})}}]}
        else:
            message = {'role': 'assistant', 'content': json.dumps(command)}
        return httpx.Response(200, json={'choices': [{'message': message,
            'finish_reason': 'tool_calls' if mode == 'native' else 'stop'}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}})
    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(handler))
    report = c.run()
    assert actions.count('read_run_records') == expected_reads
    assert len(actions) == expected_reads + 2
    assert report['state'] == 'completed' and report['control_status'] == 'completed'
    assert report['refit_status'] == 'completed' and c.publisher.output.is_file()
    assert c.journal.get('finish_evidence') and not c.journal.get('qwen_unknown_outcome')
    assert any(s.parent_id == 'baseline' and r and r.status == 'completed' for s, r in c.records())
