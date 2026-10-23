import json

from automl_agent.contracts import PipelineSpec,Hypothesis
from automl_agent.qwen import QwenClient
from tools.live_acceptance import run_case


def test_safe_trained_csv_does_not_pass_without_failure_response(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','unit-fixture')
    monkeypatch.setenv('OPENAI_BASE_URL','http://127.0.0.1:1')
    monkeypatch.setenv('OPENAI_MODEL','unit-fixture')
    calls={'count':0}
    code='def build_pipeline(context):\n    raise RuntimeError("candidate fixture")\n'
    def response(self,messages,tools=None,*,prepared=None):
        calls['count']+=1
        if calls['count']==1:
            name='run_custom_experiment'
            args={'operation_id':'first','pipeline':PipelineSpec(engine='custom',model='ridge').model_dump(mode='json'),
                'hypothesis':Hypothesis(observation='Fixture',expected_effect='Fixture',changed_component='Fixture',acceptance='Fixture').model_dump(),
                'code':code,'budget_seconds':30}
        else:
            name='finish';args={'operation_id':'done','reason':'All done successfully. Keep the currently available model.',
                'result_ids':['baseline'], 'imputation_skip_reason':'No additional hypothesis in this technical test'}
        message={'content':None,'tool_calls':[{'id':str(calls['count']),'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
        return {'choices':[{'finish_reason':'tool_calls','message':message}]} if prepared is not None else message
    monkeypatch.setattr(QwenClient,'_request',response)
    result=run_case('custom_error',0,tmp_path,90)
    assert result['valid_csv'] and result['trained_result'] and result['successful_live_requests']==2
    assert not result['checks']['repair_or_reasoned_rejection'] and not result['passed']
