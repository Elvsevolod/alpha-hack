import cloudpickle
import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import ImputationPlan, MetricSpec, OutputSpec, PipelineSpec, Transformation
from automl_agent.evaluation import align_predictions
from automl_agent.pipelines import build_pipeline


def context(task,tmp_path=None):
    return {'cpu':1,'memory_mb':2000,'task':task.model_dump(mode='json'),'strategy':'random','split_id':'test','protocol_hash':'protocol',
        'workdir':str(tmp_path or '.'),'budget_seconds':20,'deadline':9999999999}


@pytest.mark.parametrize('model', ['ridge','elastic_net','sgd','random_forest','extra_trees','hist_gradient_boosting','mlp','catboost','lightgbm','xgboost'])
def test_model_fit_predict_and_save_load(tabular,model):
    X,y,task=tabular
    parameters={'n_estimators':10} if model in {'random_forest','extra_trees','lightgbm','xgboost'} else {'iterations':10} if model=='catboost' else {'max_iter':10} if model in {'mlp','hist_gradient_boosting'} else {}
    p=build_pipeline(PipelineSpec(model=model,hyperparameters=parameters),task,{})
    c=context(task); p.fit(X,y,c)
    restored=cloudpickle.loads(cloudpickle.dumps(p)); selected=X.iloc[:15].copy(); selected['category']='unseen'
    a=p.predict(selected,c); b=restored.predict(selected,c)
    assert np.allclose(a.values['answer'],b.values['answer'])
    assert np.isfinite(align_predictions(b,selected.index.tolist(),task,'protocol','test')['answer']).all()


@pytest.mark.parametrize('method',['median','iterative_bayesian_ridge','iterative_trees','knn','native'])
def test_imputation_preserves_empty_and_new_missing(tabular,method):
    X,y,task=tabular
    X=X.drop(columns=['category']); task=task.model_copy(update={'feature_columns':list(X),'categorical_columns':[]})
    X['second']=np.arange(len(X),dtype=float)
    spec=PipelineSpec(model='lightgbm' if method=='native' else 'ridge',imputation=ImputationPlan(numeric=method))
    pipeline=build_pipeline(spec,task,{})
    c=context(task); pipeline.fit(X,y,c)
    test=X.iloc[:4].copy(); test['second']=np.nan
    b=pipeline.predict(test,c)
    assert np.isfinite(b.values['answer']).all()
    assert 'empty' in pipeline.preprocessor.numeric_columns
    assert pipeline.preprocessor.summary['training_scope']=='fit_only_no_target'
    if method in {'iterative_bayesian_ridge', 'iterative_trees'}:
        assert pipeline.preprocessor.summary['model_imputation_executed']
        assert pipeline.preprocessor.summary['actual_iterations'] > 0
        assert pipeline.preprocessor.summary['auxiliary_fits'] > 0


def test_text_vocabulary_and_numeric_combination(tabular):
    X,y,task=tabular; X=X.drop(columns=['category']); X['notes']=['warm sunny' if i%2 else 'cold rainy' for i in range(len(X))]
    task=task.model_copy(update={'feature_columns':list(X),'categorical_columns':[],'text_columns':['notes']})
    p=build_pipeline(PipelineSpec(model='ridge',representation='mixed'),task,{})
    p.fit(X,y,context(task)); new=X.iloc[:2].copy();new['notes']='totallyunseen'
    p.predict(new,context(task))
    assert 'totallyunseen' not in p.preprocessor.vectorizers['notes'].vocabulary_


def test_multioutput_regression(tabular):
    X,y,task=tabular;y['other']=y.answer*2
    task=task.model_copy(update={'kind':'multioutput_regression','targets':['answer','other'],
        'output':OutputSpec(columns=['identifier','answer','other'],prediction_type='values')})
    p=build_pipeline(PipelineSpec(model='ridge'),task,{})
    p.fit(X,y,context(task)); b=p.predict(X.iloc[:3],context(task))
    assert set(b.values)=={'answer','other'}


def test_mlp_dense_memory_guard(tabular):
    X,y,task=tabular
    p=build_pipeline(PipelineSpec(model='mlp',hyperparameters={'hidden_layer_sizes':[1000000]}),task,{})
    with pytest.raises(MemoryError):p.fit(X,y,{**context(task),'memory_mb':10})


@pytest.mark.parametrize('model',['logistic','sgd','extra_trees','catboost','lightgbm','xgboost','mlp'])
def test_multiclass_probability_alignment(tabular,model):
    X,y,t=tabular
    y['answer']=np.where(y.answer>1,'third',np.where(y.answer>0,'first','second'))
    task=t.model_copy(update={'kind':'classification','classes':{'answer':['first','second','third']},'metric':MetricSpec(name='f1'),
        'output':OutputSpec(columns=['identifier','answer'],prediction_type='labels')})
    params={'n_estimators':8} if model in {'extra_trees','lightgbm','xgboost'} else {'iterations':8} if model=='catboost' else {'max_iter':10} if model=='mlp' else {}
    p=build_pipeline(PipelineSpec(model=model,hyperparameters=params),task,{})
    p.fit(X,y,context(task));b=p.predict(X.iloc[:4],context(task))
    values=align_predictions(b,X.index[:4].tolist(),task,'protocol','test')['answer']
    assert values.shape==(4,3) and np.allclose(values.sum(axis=1),1)


def test_lightgbm_special_feature_names_preserve_values_categories_and_roundtrip(tabular):
    X,y,task=tabular
    mapping={'number':'number,mean','second':'number:mean','category':'category[code]'}
    X=X.rename(columns=mapping)
    before=X.copy(deep=True)
    task=task.model_copy(update={'feature_columns':list(X),'categorical_columns':['category[code]']})
    spec=PipelineSpec(model='lightgbm',representation='native',hyperparameters={'n_estimators':10})
    pipeline=build_pipeline(spec,task,{})
    c=context(task)
    pipeline.fit(X,y,c)
    prediction=pipeline.predict(X.iloc[:30],c)
    restored=cloudpickle.loads(cloudpickle.dumps(pipeline))
    assert np.isfinite(prediction.values['answer']).all()
    assert np.allclose(prediction.values['answer'],restored.predict(X.iloc[:30],c).values['answer'])
    assert pipeline.model_feature_names['feature_0']=='number,mean'
    assert len(set(pipeline.model_feature_names))==pipeline.preprocessor.width
    assert set(mapping.values())<=set(pipeline.model_feature_names.values())
    assert pipeline.summary.preprocessing['model_feature_name_policy']=='positional_safe_aliases'
    pd.testing.assert_frame_equal(X,before)
