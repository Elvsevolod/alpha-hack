import json
from copy import deepcopy

import httpx
import pytest

from automl_agent.eda.planning import action_window, experiment_context
from automl_agent.eda.agent import EDAAgent
from automl_agent.eda.deepseek import DeepSeekClient
from conftest import write_task


def test_final_action_reserves_closure_without_prescribing_earlier_analysis():
    tools = [{'function': {'name': name}} for name in ['finish', 'train_model', 'read_knowledge']]
    window, offered = action_window(2, tools)
    assert offered == tools and window['analysis_generations_remaining'] == 1
    window, offered = action_window(1, tools)
    assert [t['function']['name'] for t in offered] == ['finish']
    assert window['phase'] == 'conclude'


def test_comparison_preserves_changes_support_and_requires_matching_rows():
    report = {'score': 40., 'metric': 'rmse', 'fit_keys_hash': 'fit', 'evaluation_keys_hash': 'eval',
        'evaluation_rows': 24, 'evaluation_periods': 1, 'support_status': 'limited',
        'support_limitations': ['short horizon'], 'pipeline': {'model': 'lightgbm', 'target_transform': 'none',
        'hyperparameters': {'n_estimators': 300}}}
    def receipt(op, data):
        return {'command': {'operation_id': op, 'tool': 'train_model'}, 'result': {'status': 'success', 'data': data}}
    after = deepcopy(report); after['score'] = 33.; after['pipeline']['target_transform'] = 'log1p'
    after['pipeline']['hyperparameters']['n_estimators'] = 500
    context = experiment_context([receipt('base', report), receipt('test', after)])
    comparison = context['experiments'][1]['comparison']
    assert comparison['matched_rows_and_metric'] and comparison['score_delta'] == -7
    assert len(comparison['changes']) == 2 and comparison['causal_attribution'] == 'not_established'
    assert context['experiments'][1]['support_limitations'] == ['short horizon']
    after['evaluation_keys_hash'] = 'different'
    assert experiment_context([receipt('a', report), receipt('b', after)])['experiments'][1]['comparison']['score_delta'] is None


@pytest.mark.parametrize('transport', ['native', 'json'])
@pytest.mark.parametrize('honors_finish', [True, False])
def test_http_budget_reserves_model_written_finish_in_both_transports(tmp_path, monkeypatch, transport, honors_finish):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-fixture')
    task = write_task(tmp_path/'input')
    phases = []
    def response(request):
        body = json.loads(request.content); context = json.loads(body['messages'][1]['content'])
        if 'request_id' in context:
            spec = json.loads((task.parent/'task_spec.json').read_text())
            spec.update(contract_version=2, source_hash=context['source_hash'],
                file_roles={r:r for r in ['train.csv', 'test.csv', 'sample_submission.csv']})
            message = {'content': json.dumps({'request_id': context['request_id'], 'source_hash':context['source_hash'],
                'status':'proposed', 'task_spec':spec, 'evidence':{k:['task','inventory'] for k in spec}})}
        else:
            phase = context['action_window']['phase']; phases.append(phase)
            closing = phase == 'conclude'
            names = ({t['function']['name'] for t in body['tools']} if transport == 'native'
                else set(context['tool_argument_schemas']))
            assert names == {'finish'} if closing else len(names) == 9
            assert 'experiment_context' in context
            command = {'operation_id':f'op{len(phases)}', 'tool':'finish' if closing and honors_finish else 'get_dataset_report',
                'arguments':{'reason':'Measured missingness is recorded. Effect unknown. Next compare one imputation change on matched rows; accept lower inner RMSE subject to adequate support.'}
                    if closing and honors_finish else {'report':'missing', 'columns':['value']}}
            message = ({'tool_calls':[{'id':command['operation_id'], 'function':{'name':command['tool'],
                'arguments':json.dumps({'operation_id':command['operation_id'], **command['arguments']})}}]}
                if transport == 'native' else {'content':json.dumps(command)})
        return httpx.Response(200, json={'choices':[{'message':message}], 'usage':{'prompt_tokens':100,'completion_tokens':100}})
    agent = EDAAgent(task,tmp_path/'result',tmp_path/'ledger.json',transport=transport,
        client=httpx.Client(transport=httpx.MockTransport(response)))
    monkeypatch.setattr(agent.qwen,'resolve',DeepSeekClient.resolve.__get__(agent.qwen))
    try:
        result = agent.autonomous()
    finally:
        agent.close()
    assert result['state'] == ('completed' if honors_finish else 'partial'), result
    if not honors_finish:
        history = json.loads((tmp_path/'result'/'trajectory.json').read_text())
        assert history[-1]['result']['status'] == 'rejected'
        assert 'reserved for finish' in history[-1]['result']['error']
    assert result['model_generations'] == 7 and result['model_fits'] == 0
    assert phases[-1] == 'conclude' and phases.count('conclude') == 1
    assert not list((tmp_path/'result').rglob('submission.csv'))


@pytest.mark.parametrize('provider', ['deepseek', 'organizer'])
def test_native_schema_failure_replans_last_generation_before_json_fallback(tmp_path, monkeypatch, provider):
    import time
    from automl_agent.budget import Budget, Limits
    from automl_agent.journal import Journal
    from automl_agent.eda.deepseek import CostLedger
    from automl_agent.qwen import native_tools
    from tools.qwen_eda_live import GatewayClient, GatewayLedger, limits
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-fixture')
    journal = Journal(tmp_path/'run')
    budget = Budget(journal, Limits(seconds=120, max_calls=7), time.time())
    budget.counter('llm_calls', 5)
    bodies = []
    def response(request):
        body = json.loads(request.content); bodies.append(body)
        context = json.loads(body['messages'][1]['content'])
        if len(bodies) == 1:
            assert context['remaining']['generations'] == 2
            train = next(t for t in body['tools'] if t['function']['name'] == 'train_model')
            fields = train['function']['parameters']['$defs']['PipelineSpec']['properties']
            assert fields['fit_diagnostics']['const'] == 'none'
            assert fields['engine']['const'] == 'direct'
            # A known successful response with a schema error; no fit was executed.
            message = {'tool_calls':[{'id':'bad', 'function':{'name':'train_model',
                'arguments':json.dumps({'pipeline':'invalid encoded object', 'hypothesis':'invalid'})}}]}
        else:
            assert len(bodies) == 2
            assert context['remaining']['generations'] == 1
            assert context['action_window']['phase'] == 'conclude'
            assert set(context['tool_argument_schemas']) == {'finish'}
            assert context['command_schema']['properties']['tool']['enum'] == ['finish']
            assert 'transport_repair' in context
            message = {'content':json.dumps({'operation_id':'done','tool':'finish',
                'arguments':{'reason':'No second experiment executed. Effect unknown; next test one change with adequate support.'}})}
        return httpx.Response(200,headers={'x-litellm-response-cost':'.01'},json={
            'choices':[{'finish_reason':'stop','message':message}], 'usage':{'prompt_tokens':100,'completion_tokens':100}})
    http = httpx.Client(transport=httpx.MockTransport(response))
    if provider == 'deepseek':
        client = DeepSeekClient(journal,budget,CostLedger(tmp_path/'costs.json','test'),client=http)
    else:
        account = {'spend':0.,'total_spend':0.,'max_budget':600.,'blocked':False}
        monkeypatch.setattr(limits,'fetch_info',lambda key:dict(account))
        ledger = GatewayLedger(tmp_path/'costs.json','fixture',limits.snapshot(account,'fixture','before'),'test')
        client = GatewayClient(journal,budget,ledger,{'key':'fixture','model':'fixture','base_url':'https://fixture.invalid'},client=http)
    client.allowed_tools = [t for t in native_tools() if t['function']['name'] in {'train_model','finish'}]
    try:
        command, _ = client.next_command('Investigate and conclude.', {'action_window':{'phase':'investigate'}, 'remaining':{'generations':2}})
        assert command.tool == 'finish' and budget.counter('llm_calls') == 7
        assert budget.counter('eda_fits') == 0
    finally:
        client.close(); journal.close()


def test_known_malformed_json_returns_feedback_then_reserves_finish(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','offline-fixture')
    task=write_task(tmp_path/'input')
    seen=[]
    def response(request):
        body=json.loads(request.content); context=json.loads(body['messages'][1]['content']);seen.append(context)
        if 'request_id' in context:
            spec=json.loads((task.parent/'task_spec.json').read_text())
            spec.update(contract_version=2,source_hash=context['source_hash'],file_roles={r:r for r in ['train.csv','test.csv','sample_submission.csv']})
            message={'content':json.dumps({'request_id':context['request_id'],'source_hash':context['source_hash'],
                'status':'proposed','task_spec':spec,'evidence':{k:['task','inventory'] for k in spec}})}
        elif context['action_window']['phase']!='conclude':
            assert not context['recent_results'] or context['recent_results'][-1]['result']['executed'] is False
            message={'content':json.dumps({'operation_id':'bad','tool':'train_model','arguments':{
                'pipeline':{'model':'ridge'},'hypothesis':{'observation':'missing effect'}}})}
        else:
            assert context['recent_results'][-1]['result']['status']=='rejected'
            assert set(context['tool_argument_schemas'])=={'finish'}
            message={'content':json.dumps({'operation_id':'done','tool':'finish','arguments':{
                'reason':'No experiments were executed because earlier commands failed schema validation. Next compare a single supported pipeline on matched rows; effects unknown.'}})}
        return httpx.Response(200,json={'choices':[{'message':message}], 'usage':{'prompt_tokens':100,'completion_tokens':100}})
    agent=EDAAgent(task,tmp_path/'result',tmp_path/'ledger.json',transport='json',
        client=httpx.Client(transport=httpx.MockTransport(response)))
    monkeypatch.setattr(agent.qwen,'resolve',DeepSeekClient.resolve.__get__(agent.qwen))
    try: result=agent.autonomous()
    finally: agent.close()
    assert result['state']=='completed' and result['model_generations']==7 and result['model_fits']==0
    assert len(seen)==7
    assert (tmp_path/'result'/'evaluation_report.json').exists()
