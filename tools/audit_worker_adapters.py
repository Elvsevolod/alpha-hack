"""No Qwen: exercise real adapter admission, isolated workers and frozen refit."""
import json
import tempfile
from pathlib import Path
from automl_agent.controller import Controller
from automl_agent.contracts import PipelineSpec,ToolCommand
from automl_agent.budget import Limits
from automl_agent.tool_results import payload
from automl_agent.util import read_json
from tools.offline_acceptance import fixture


def main():
    reports=[]
    for engine,profile in [('lightautoml','selection'),('autogluon','bagging'),('autogluon','stacking')]:
        with tempfile.TemporaryDirectory(prefix='alpha-worker-') as directory:
            root=Path(directory);task=fixture(root)
            c=Controller(task,Limits(seconds=600,cpu=2,memory_mb=4000),run_id='worker_adapter')
            try:
                assert not c.qwen.configured
                c.prepare()
                c.qwen.critique = lambda request, reserve_seconds: {
                    'review_id':request['review_id'], 'verdict':'approve','evaluation_scope':'research_validation',
                    'comparator_id':c.journal.get('baseline_id'), 'evidence_assessment':'Offline synthetic adapter verification.',
                    'comparison_assessment':'Fixed local protocol.', 'leakage_assessment':'Fit-only workers.',
                    'cost_assessment':'Bounded technical test.', 'falsification_assessment':'Fail on incomplete fit/refit.',
                    'revision_requests':[]}
                pipeline=PipelineSpec(engine=engine,model=engine,learners=['lgb'] if engine=='lightautoml' else ['GBM'],autogluon_profile=profile,bag_folds=2,n_trials=1)
                result=payload(c.dispatch(ToolCommand(operation_id='verify_'+engine,tool='run_automl',arguments={
                    'pipeline':pipeline.model_dump(),'budget_seconds':180,
                    'hypothesis':{'observation':'Offline synthetic fixture','expected_effect':'Verify full worker adapter',
                        'changed_component':engine+profile,'acceptance':'Fit, saved probabilities, serialisation, complete refit'}})))
                if result.get('result',{}).get('status')!='completed':
                    for failure in c.workdir.rglob('worker_result.json'):
                        raw=read_json(failure)
                        if raw.get('traceback'):print(raw['traceback'],flush=True)
                    raise AssertionError(result)
                spec,measured=c.lookup(result['experiment']['experiment_id'])
                artifact=Path(measured.artifacts['directory'])
                raw=read_json(artifact/'worker_result.json')
                frozen=PipelineSpec.model_validate(raw['frozen_pipeline'])
                destination=c.workdir/'adapter_full_refit'
                request=c.stage(destination,spec,list(range(len(c.dataset.train))),{'test':[]},pipeline=frozen,mode='refit',model_path=artifact/'model.pkl')
                outcome=c.executor.run(destination,request,120)
                assert outcome['status']=='completed',outcome
                assert outcome['fit_rows']==len(c.dataset.train)
                reports.append({'engine':engine,'profile':profile,'research_rows':measured.training_summary['rows'],
                    'full_refit_rows':outcome['fit_rows'],'status':outcome['status'],'real_qwen_calls':0})
                print(json.dumps(reports[-1]),flush=True)
            finally:c.close()
    Path('outputs/audit_plan_v074_20261024/pinned_worker_adapters.json').write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__':main()
