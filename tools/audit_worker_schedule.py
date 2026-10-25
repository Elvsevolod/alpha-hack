"""Pinned worker test of native schedules and fixed-iteration full refit."""
import json
import tempfile
from pathlib import Path
from automl_agent.controller import Controller
from automl_agent.budget import Limits
from automl_agent.contracts import PipelineSpec,TrainArgs,Hypothesis
from automl_agent.util import read_json
from tools.offline_acceptance import fixture


def main():
    reports=[]
    for model,parameter in [('lightgbm','n_estimators'),('xgboost','n_estimators'),('catboost','iterations'),('hist_gradient_boosting','max_iter')]:
        with tempfile.TemporaryDirectory(prefix='schedule-worker-') as directory:
            root=Path(directory);path=fixture(root)
            c=Controller(path,Limits(seconds=180,cpu=2,memory_mb=4000))
            try:
                c.prepare();assert not c.qwen.configured
                pipeline=PipelineSpec(model=model,hyperparameters={parameter:40},early_stopping_rounds=5,evaluation_interval=5)
                spec=c.make_spec(TrainArgs(pipeline=pipeline,budget_seconds=45,hypothesis=Hypothesis(
                    observation='Offline schedule test',expected_effect='Verify schedule/refit contract',changed_component=model,acceptance='Complete worker fit and fixed refit')),'schedule')
                work=c.workdir/'schedule_fit';request=c.stage(work,spec,c.plan.fit,{'validation':c.plan.validation})
                result=c.executor.run(work,request,45)
                assert result['status']=='completed',result
                frozen=PipelineSpec.model_validate(result['frozen_pipeline'])
                assert 1<=frozen.iteration_schedule[c.task.targets[0]]<=40
                work=c.workdir/'schedule_refit';request=c.stage(work,spec,list(range(len(c.dataset.train))),{'test':[]},pipeline=frozen,mode='refit')
                refit=c.executor.run(work,request,45)
                assert refit['status']=='completed',refit
                assert refit['fit_rows']==len(c.dataset.train) and refit['training_summary']['auxiliary_fits']==0
                reports.append({'model':model,'iterations':frozen.iteration_schedule,'refit_rows':refit['fit_rows'],'status':'completed'})
                print(json.dumps(reports[-1]),flush=True)
            finally:c.close()
    Path('outputs/audit_plan_v074_20261024/pinned_worker_schedules.json').write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__':main()
