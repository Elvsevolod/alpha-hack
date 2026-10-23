from automl_agent.tool_results import payload as tool_payload
from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis,PipelineSpec,ToolCommand
from automl_agent.controller import Controller
from conftest import write_task
import pytest


VALID_CODE='''import pandas as pd
class Features:
    def fit(self,X,y=None):return self
    def transform(self,X):
        result=X.copy()
        result['square']=pd.to_numeric(result['value'],errors='coerce')**2
        return result
def build_transformer(context):return Features()
'''


@pytest.mark.parametrize('engine', ['custom', 'direct'])
def test_custom_component_save_load_control_and_refit_with_dependency_guard(tmp_path, engine):
    c=Controller(write_task(tmp_path),Limits(seconds=120,cpu=2),run_id='custom')
    try:
        c.prepare()
        args={'pipeline':PipelineSpec(engine=engine,model='ridge',custom_component='transformer').model_dump(mode='json'),
            'hypothesis':Hypothesis(observation='Fit numeric feature',expected_effect='Represent curvature',changed_component='square',acceptance='Fixed metric').model_dump(),
            'code':VALID_CODE,'budget_seconds':30}
        outcome=tool_payload(c.dispatch(ToolCommand(operation_id='component',tool='run_custom_experiment',arguments=args)))
        assert outcome['result']['status']=='completed',outcome
        c.journal.put('leader_id',outcome['experiment']['experiment_id'])
        c.freeze('Test frozen custom graph');c.finalize()
        assert c.journal.get('control_status')=='completed'
        assert c.journal.get('refit_status')=='completed'
    finally:c.close()
