import cloudpickle
import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import PipelineSpec, Transformation, MetricSpec, OutputSpec
from automl_agent.pipelines import DirectPipeline
from automl_agent.preprocessing import FeatureMatrix
from test_pipelines import context


@pytest.mark.parametrize('model,parameter', [('lightgbm','n_estimators'),('xgboost','n_estimators'),('catboost','iterations'),('hist_gradient_boosting','max_iter')])
def test_schedule_frozen_full_refit(tabular, model, parameter):
    X,y,task=tabular
    spec=PipelineSpec(model=model, hyperparameters={parameter:40}, early_stopping_rounds=5, evaluation_interval=5)
    p=DirectPipeline(spec,task)
    summary=p.fit(X,y,context(task))
    frozen=p.frozen_spec()
    assert 1<=frozen.iteration_schedule['answer']<=40
    assert summary.learning_curve['answer']['scope']=='fit_inner_holdout'
    refit=DirectPipeline(frozen,task);refit.fit(X,y,context(task))
    assert refit.summary.auxiliary_fits==0
    restored=cloudpickle.loads(cloudpickle.dumps(refit))
    assert np.allclose(refit.predict(X.iloc[:5],context(task)).values['answer'],restored.predict(X.iloc[:5],context(task)).values['answer'])


def test_unique_target_encoder_is_oof(tabular):
    X,y,task=tabular;X=X.copy();X['category']=[f'unique_{i}' for i in range(len(X))]
    spec=PipelineSpec(model='ridge',transformations=[Transformation(name='target_encode',parameters={'columns':['category']})])
    p=DirectPipeline(spec,task);p.fit(X,y,context(task))
    report=p.summary.preprocessing['supervised_features'][0]
    assert report['scope']=='external_split_aware_oof' and report['coverage']==1
    assert np.isfinite(p.predict(X.iloc[:5],context(task)).values['answer']).all()


def test_history_excludes_equal_and_future_targets(tabular):
    X,y,task=tabular;X=X.copy();X['entity']='a';X['when']=np.arange(len(X))//2
    task=task.model_copy(update={'feature_columns':list(X),'categorical_columns':['category','entity'],'time_column':'when'})
    spec=PipelineSpec(model='ridge',transformations=[Transformation(name='history',parameters={'columns':['entity']})])
    from automl_agent.supervised_features import SupervisedFeatures
    a=SupervisedFeatures(spec,task,'temporal').fit_transform(X,y,X)
    changed=y.copy();changed.iloc[20:,0]=999999
    b=SupervisedFeatures(spec,task,'temporal').fit_transform(X,changed,X)
    columns=[c for c in a if c.startswith('__history')]
    assert np.array_equal(a.iloc[:22][columns],b.iloc[:22][columns])
    assert a.filter(like='_count').iloc[0,0]==0


def test_group_target_encoding_excludes_all_group_labels(tabular):
    from automl_agent.supervised_features import SupervisedFeatures
    X,y,task=tabular;X=X.copy();X['entity']=np.tile(['a','b','c','d','e','f'],30)
    task=task.model_copy(update={'feature_columns':list(X),'group_column':'entity','categorical_columns':['category','entity']})
    spec=PipelineSpec(model='ridge',transformations=[Transformation(name='target_encode',parameters={'columns':['entity']})])
    a=SupervisedFeatures(spec,task,'group').fit_transform(X,y,X)
    changed=y.copy();changed.loc[X.entity=='a','answer']=99999
    b=SupervisedFeatures(spec,task,'group').fit_transform(X,changed,X)
    columns=[c for c in a if c.startswith('__te_')]
    assert np.array_equal(a.loc[X.entity=='a',columns],b.loc[X.entity=='a',columns])


def test_history_reports_known_and_cold_quality_without_declared_group(tabular):
    X,y,task=tabular;X=X.copy()
    X['entity']=np.where(np.arange(len(X))>=len(X)-12,'new','known')
    X['when']=np.arange(len(X))//30
    task=task.model_copy(update={'feature_columns':list(X),'categorical_columns':['category','entity'],'time_column':'when'})
    spec=PipelineSpec(model='ridge',transformations=[Transformation(name='history',parameters={'columns':['entity']})])
    model=DirectPipeline(spec,task)
    summary=model.fit(X,y,{**context(task),'strategy':'temporal'})
    diagnostic=summary.fit_diagnostics['entity_history']
    groups=next(iter(diagnostic['reports'].values()))
    assert task.group_column is None and diagnostic['scope']=='fit_inner_holdout'
    assert groups['cold']['rows']==12 and groups['known']['rows']>0
    assert sum(row['rows'] for row in groups.values())==diagnostic['evaluation_rows']
    assert all(np.isfinite(row['score']) and np.isfinite(row['bias']['answer']) for row in groups.values())
    refit=DirectPipeline(model.frozen_spec(),task)
    refit.fit(X,y,{**context(task),'strategy':'temporal','execution_mode':'refit'})
    assert refit.summary.fit_diagnostics is None and refit.summary.auxiliary_fits==0
