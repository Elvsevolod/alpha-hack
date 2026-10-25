"""No network, no pytest dependency: checks in the organizer-pinned image."""
import cloudpickle
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from automl_agent.contracts import TaskSpec,MetricSpec,OutputSpec,PipelineSpec
from automl_agent.pipelines import build_pipeline


def main():
    random=np.random.default_rng(42)
    X=pd.DataFrame({'value':random.normal(size=300),'category':np.tile(['a','b','c'],100)},index=['row_'+str(i) for i in range(300)])
    y=pd.DataFrame({'answer':X.value*3+random.normal(size=300)*.1},index=X.index)
    task=TaskSpec(kind='regression',targets=['answer'],feature_columns=list(X),categorical_columns=['category'],
        metric=MetricSpec(name='rmse'),output=OutputSpec(columns=['answer'],prediction_type='values'),source_hash='fixture')
    results=[]
    configurations=[('lightautoml','selection'),('autogluon','bagging'),('autogluon','stacking')]
    for engine,profile in configurations:
        with tempfile.TemporaryDirectory() as directory:
            context={'cpu':2,'memory_mb':4000,'strategy':'random','split_id':'test','protocol_hash':'p',
                'task':task.model_dump(mode='json'),'workdir':directory,'budget_seconds':100,'deadline':9999999999,
                'phase_estimates':{'selected_fit_seconds':1,'prediction_seconds':1,'publication_seconds':1}}
            spec=PipelineSpec(engine=engine,model=engine,learners=['lgb'] if engine=='lightautoml' else ['GBM'],
                autogluon_profile=profile,bag_folds=2,n_trials=1)
            pipeline=build_pipeline(spec,task,context)
            summary=pipeline.fit(X,y,context)
            restored=cloudpickle.loads(cloudpickle.dumps(pipeline))
            a=pipeline.predict(X.iloc[:12],context);b=restored.predict(X.iloc[:12],context)
            assert np.allclose(a.values['answer'],b.values['answer'])
            extra=X.copy();extra.index=['extra_'+str(i) for i in range(len(extra))]
            labels=y.copy();labels.index=extra.index
            combined=pd.concat([X,extra]);combined_y=pd.concat([y,labels])
            refit=restored.refit(combined,combined_y,context)
            assert refit.rows==len(combined)
            if engine=='lightautoml':
                assert pipeline.selected['answer']['parameters']['num_trees'] <= 300
                assert restored.selected['answer']['parameters']['num_trees']==pipeline.selected['answer']['parameters']['num_trees']
            else:
                predictor=restored.models['answer'][1]
                sizes={name:info.get('num_samples') for name,info in predictor.info()['model_info'].items() if name.endswith('_FULL')}
                print(json.dumps({'engine':engine,'profile':profile,'model_rows':sizes}),flush=True)
                assert restored.selected['answer']['full_refit_rows']==len(combined)
                assert sizes and all(rows==len(combined) for name,rows in sizes.items() if 'WeightedEnsemble' not in name)
            assert np.isfinite(restored.predict(extra.iloc[:12],context).values['answer']).all()
            results.append({'engine':engine,'profile':profile,'save_load':True,'research_rows':summary.rows,
                'refit_rows':refit.rows,'refit':refit.model_dump(mode='json')})
            print(json.dumps(results[-1]),flush=True)
    Path('outputs/audit_plan_v074_20261024/pinned_native_adapters.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
