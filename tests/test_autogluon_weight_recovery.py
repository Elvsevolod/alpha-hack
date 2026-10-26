"""An unsupported nested weight request must lead to an explicit new recipe."""
import json
from pathlib import Path
import cloudpickle
import httpx
import pandas as pd
import pytest
from automl_agent.budget import Limits
from automl_agent.controller import Controller
from conftest import write_task


@pytest.mark.parametrize('transport',['native','json'])
def test_native_weight_error_reaches_next_decision_and_verified_direct_fit(tmp_path,monkeypatch,transport):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    c=Controller(write_task(tmp_path,kind='classification'),Limits(seconds=180,cpu=1,max_calls=4),
        run_id='nested_weights',transport=transport)
    requests=[]
    hypothesis={'observation':'Offline nested-adapter contract fixture',
        'expected_effect':'Preserve requested class weighting through a supported adapter',
        'changed_component':'class weighting adapter','acceptance':'Completed weighted worker and full refit CSV'}
    def respond(request):
        body=json.loads(request.content);requests.append(body)
        if len(requests)==1:
            command={'operation_id':'bad_nested_weight','tool':'run_automl','arguments':{
                'pipeline':{'engine':'autogluon','model':'library_search','learners':['GBM'],
                    'autogluon_configurations':{'GBM':[{'class_weight':'balanced'}]}},
                'budget_seconds':30,'hypothesis':hypothesis}}
        elif len(requests)==2:
            work=json.loads(body['messages'][1]['content'])['core']['current_work']
            diagnostic=json.loads(work['diagnostic'])
            assert diagnostic['status']=='command_validation'
            assert 'direct/Optuna' in diagnostic['error']
            assert c.journal.operation('bad_nested_weight') is None
            command={'operation_id':'weighted_direct','tool':'train_model','arguments':{
                'pipeline':{'model':'lightgbm','hyperparameters':{'n_estimators':5,'class_weight':'balanced'}},
                'budget_seconds':30,'hypothesis':hypothesis}}
        else:
            command={'operation_id':'finish_weights','tool':'finish','arguments':{
                'result_ids':[c.journal.operation('weighted_direct').experiment_id],'reason':'Offline adapter recovery checked',
                'imputation_skip_reason':'Fixed fixture','skipped_directions':{'all':'Scripted integration test, no quality claim'}}}
        message={'role':'assistant','content':json.dumps(command)}
        if 'tools' in body:
            message={'role':'assistant','content':None,'tool_calls':[{'id':'call_'+str(len(requests)),
                'type':'function','function':{'name':command['tool'],'arguments':json.dumps({
                    'operation_id':command['operation_id'],**command['arguments']})}}]}
        return httpx.Response(200,json={'choices':[{'message':message,
            'finish_reason':'tool_calls' if 'tools' in body else 'stop'}],
            'usage':{'prompt_tokens':10,'completion_tokens':5}})
    c.qwen.client.close();c.qwen.client=httpx.Client(transport=httpx.MockTransport(respond))
    c.qwen.key,c.qwen.base_url='offline-fixture','http://fixture.invalid/v1'
    try:
        report=c.run()
        assert len(requests)==3,report
        assert all(('tools' in body)==(transport=='native') for body in requests)
        assert c.journal.operation('bad_nested_weight') is None
        result=next(r for s,r in c.records() if s.experiment_id==c.journal.operation('weighted_direct').experiment_id)
        assert result.status=='completed',result
        with (Path(result.artifacts['directory'])/'model.pkl').open('rb') as stream:
            fitted=cloudpickle.load(stream)
        assert fitted.models['answer'].get_params()['class_weight']=='balanced'
        assert report['state']=='completed' and report['refit_status']=='completed',report
        assert report['publication']['scope']=='full_train_refit'
        output=pd.read_csv(c.publisher.output,dtype={'row':str})
        sample=pd.read_csv(tmp_path/'sample_submission.csv',dtype={'row':str})
        assert list(output)==list(sample) and output.row.tolist()==sample.row.tolist()
        assert set(output.answer)<={'negative','positive'}
    finally:
        c.close()
