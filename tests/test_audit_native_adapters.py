import importlib.util
import cloudpickle
import numpy as np
import pytest
from automl_agent.contracts import PipelineSpec
from automl_agent.pipelines import build_pipeline
from test_pipelines import context


@pytest.mark.skipif(importlib.util.find_spec('lightautoml') is None,reason='LightAutoML checked in pinned Docker')
def test_lightautoml_fit_save_load_complete_refit(tabular,tmp_path):
    X,y,task=tabular
    X=X.drop(columns=['empty']);task=task.model_copy(update={'feature_columns':list(X)})
    c={**context(task,tmp_path),'budget_seconds':40}
    spec=PipelineSpec(engine='lightautoml',model='lightautoml',learners=['lgb'])
    pipeline=build_pipeline(spec,task,c)
    summary=pipeline.fit(X,y,c)
    assert summary.rows==len(X)
    assert summary.selected_parameters['answer']['parameters']['early_stopping_rounds']==0
    restored=cloudpickle.loads(cloudpickle.dumps(pipeline))
    assert np.allclose(pipeline.predict(X.iloc[:5],c).values['answer'],restored.predict(X.iloc[:5],c).values['answer'])
    added=X.copy();added.index=['new_'+str(i) for i in range(len(X))]
    labels=y.copy();labels.index=added.index
    summary=restored.refit(added,labels,c)
    assert summary.budget['selected_fit_rows']==len(added)
    assert summary.budget['refit_policy']=='frozen_single_family_fixed_iterations_all_permitted_rows'


@pytest.mark.skipif(importlib.util.find_spec('autogluon') is None,reason='AutoGluon checked in pinned Docker')
@pytest.mark.parametrize('profile',['bagging','stacking'])
def test_autogluon_profiles_full_refit(tabular,tmp_path,profile):
    X,y,task=tabular
    c={**context(task,tmp_path),'budget_seconds':70,'phase_estimates':{'selected_fit_seconds':1,'prediction_seconds':1}}
    spec=PipelineSpec(engine='autogluon',model='autogluon',learners=['GBM'],autogluon_profile=profile,bag_folds=2,n_trials=1)
    pipeline=build_pipeline(spec,task,c)
    summary=pipeline.fit(X,y,c)
    assert summary.rows==len(X)
    assert pipeline.selected['answer']['bagging']
    assert pipeline.selected['answer']['stacking']==(profile=='stacking')
    assert np.isfinite(pipeline.predict(X.iloc[:5],c).values['answer']).all()
    restored=cloudpickle.loads(cloudpickle.dumps(pipeline))
    assert np.allclose(pipeline.predict(X.iloc[:5],c).values['answer'],restored.predict(X.iloc[:5],c).values['answer'])
