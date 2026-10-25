"""Pinned offline classification/group contracts for the new AutoML paths."""
import cloudpickle
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from automl_agent.contracts import TaskSpec,MetricSpec,OutputSpec,PipelineSpec
from automl_agent.pipelines import build_pipeline


def main():
    rng=np.random.default_rng(12);n=360
    X=pd.DataFrame({'x':rng.normal(size=n),'kind':np.tile(['red','green','blue'],120),'entity':np.tile(np.arange(30),12)},index=['row_'+str(i) for i in range(n)])
    y=pd.DataFrame({'answer':np.where(X.x<-.5,'z',np.where(X.x>.5,'m','a'))},index=X.index)
    task=TaskSpec(kind='classification',targets=['answer'],feature_columns=['x','kind'],categorical_columns=['kind'],group_column='entity',
        classes={'answer':['z','a','m']},metric=MetricSpec(name='f1'),output=OutputSpec(columns=['answer'],prediction_type='labels'),source_hash='fixture')
    reports=[]
    for engine,profile in [('lightautoml','selection'),('autogluon','bagging'),('autogluon','stacking')]:
        with tempfile.TemporaryDirectory() as directory:
            c={'cpu':2,'memory_mb':4000,'strategy':'group','split_id':'test','protocol_hash':'p','task':task.model_dump(),'workdir':directory,
                'budget_seconds':100,'deadline':9999999999,'phase_estimates':{'selected_fit_seconds':1,'prediction_seconds':1,'publication_seconds':1}}
            spec=PipelineSpec(engine=engine,model=engine,learners=['lgb'] if engine=='lightautoml' else ['GBM'],autogluon_profile=profile,bag_folds=2,n_trials=1)
            model=build_pipeline(spec,task,c);summary=model.fit(X,y,c)
            restored=cloudpickle.loads(cloudpickle.dumps(model))
            bundle=restored.predict(X.iloc[:15],c)
            probabilities=np.asarray(bundle.values['answer'])
            assert probabilities.shape==(15,3) and np.allclose(probabilities.sum(axis=1),1,atol=1e-5)
            assert bundle.classes==task.classes
            split=summary.inner_scores.get('split',summary.inner_scores)
            a,b=split['fit_keys'],split.get('validation_keys',split.get('heldout_keys'))
            assert set(X.loc[a,'entity']).isdisjoint(X.loc[b,'entity'])
            reports.append({'engine':engine,'profile':profile,'classes':bundle.classes,'group_inner_holdout_disjoint':True,'rows':summary.rows,'save_load':True})
            print(json.dumps(reports[-1]),flush=True)
    Path('outputs/audit_plan_v074_20261024/pinned_classification.json').write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__':main()
