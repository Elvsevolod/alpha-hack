import cloudpickle
import importlib.util
import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import MetricSpec, OutputSpec, PipelineSpec, TaskSpec
from automl_agent.evaluation import align_predictions, score_predictions
from automl_agent.pipelines import build_pipeline
from automl_agent.search import suggest_parameters
from test_pipelines import context


@pytest.mark.parametrize('engine',['optuna','flaml',pytest.param('autogluon',
    marks=pytest.mark.skipif(importlib.util.find_spec('autogluon') is None,
        reason='AutoGluon unavailable locally; requires organizer dependency environment'))])
@pytest.mark.parametrize('classification',[False,True])
def test_search_fit_save_load_and_fixed_refit(tabular,tmp_path,engine,classification):
    X,y,task=tabular
    if classification:
        y['answer']=np.where(y.answer>0,'yes','no')
        task=task.model_copy(update={'kind':'classification','classes':{'answer':['no','yes']},'metric':MetricSpec(name='f1'),
            'output':OutputSpec(columns=['identifier','answer'],prediction_type='labels')})
    model='logistic' if classification else 'ridge'
    pipeline=PipelineSpec(engine=engine,model=model,search_owner=engine,n_trials=2,
        learners=['lgbm'] if engine=='flaml' else ['GBM'] if engine=='autogluon' else [],
        search_space={'C' if classification else 'alpha':{'type':'float','low':.1,'high':2.}} if engine=='optuna' else {})
    c=context(task,tmp_path); c['budget_seconds']=20
    p=build_pipeline(pipeline,task,c); summary=p.fit(X.iloc[:140],y.iloc[:140],c)
    b=p.predict(X.iloc[140:150],c)
    pred=align_predictions(b,X.index[140:150].tolist(),task,'protocol','test')
    assert np.isfinite(score_predictions(task,y.iloc[140:150],pred,b.prediction_type))
    restored=cloudpickle.loads(cloudpickle.dumps(p))
    assert restored.predict(X.iloc[140:150],c).values==b.values
    directory=tmp_path/'fixed_refit'; directory.mkdir()
    restored.refit(X,y,{**c,'workdir':str(directory)})
    assert restored.training_keys==X.index.tolist() or engine=='optuna'
    if engine=='optuna': assert restored.frozen_spec().engine=='direct'
    assert np.isfinite(np.asarray(restored.predict(X.iloc[:4],c).values['answer'],dtype=float)).all()


@pytest.mark.parametrize('model',['naive','lightgbm','sarimax','exponential_smoothing'])
def test_forecast_multiple_series_causality_and_short_history(model,tmp_path):
    rows=[]; values=[]
    for group,length in [('long',35),('short',2)]:
        for t in range(length):rows.append({'series':group,'time':t});values.append(float(t)+1)
    X=pd.DataFrame(rows,index=[f'train:{i}' for i in range(len(rows))]); y=pd.DataFrame({'target':values},index=X.index)
    task=TaskSpec(kind='forecasting',targets=['target'],feature_columns=['series','time'],categorical_columns=['series'],
        time_column='time',series_columns=['series'],horizon=3,metric=MetricSpec(name='rmse'),
        output=OutputSpec(columns=['target'],prediction_type='values'),source_hash='fixture')
    spec=PipelineSpec(model=model,representation='lags',hyperparameters={'n_estimators':10} if model=='lightgbm' else {})
    c={**context(task,tmp_path),'strategy':'temporal'}
    p=build_pipeline(spec,task,c); p.fit(X,y,c)
    test=pd.DataFrame({'series':['long','short','long','new'],'time':[36,3,35,2]},index=['a','b','c','d'])
    before=cloudpickle.dumps(p.history)
    b=p.predict(test,c)
    assert b.row_keys==test.index.tolist() and np.isfinite(b.values['target']).all()
    assert cloudpickle.dumps(p.history)==before
    # A prediction cannot use a target from inside its own future horizon.
    with pytest.raises(ValueError):p.predict(pd.DataFrame({'series':['long'],'time':[1]},index=['x']),c)
    with pytest.raises(ValueError,match='one forecast row'):
        p.predict(pd.DataFrame({'series':['long','long'],'time':[36,36]},index=['x','y']),c)


def test_optuna_forecasting_fails_before_pipeline_construction(tmp_path):
    task=TaskSpec(kind='forecasting',targets=['target'],feature_columns=['series','time'],categorical_columns=['series'],
        time_column='time',series_columns=['series'],horizon=2,metric=MetricSpec(name='rmse'),
        output=OutputSpec(columns=['target'],prediction_type='values'),source_hash='fixture')
    spec=PipelineSpec(engine='optuna',search_owner='optuna',model='ridge',
        search_space={'alpha':{'type':'float','low':.1,'high':2.}})
    with pytest.raises(ValueError,match='optuna_forecasting_requires_causal_forecast_adapter'):
        build_pipeline(spec,task,{'strategy':'temporal'})


def test_ranking_shuffled_groups_preserves_rows(tmp_path):
    rng=np.random.default_rng(7);n=120
    X=pd.DataFrame({'query':np.repeat(np.arange(12),10),'x':rng.normal(size=n)},index=[str(i) for i in range(n)])
    y=pd.DataFrame({'rel':np.tile(np.arange(10)%4,12)},index=X.index)
    order=rng.permutation(n);X,y=X.iloc[order],y.iloc[order]
    task=TaskSpec(kind='ranking',targets=['rel'],feature_columns=['query','x'],group_column='query',metric=MetricSpec(name='ndcg',k=5),
        output=OutputSpec(columns=['rel'],prediction_type='scores'),source_hash='fixture')
    p=build_pipeline(PipelineSpec(model='lgbm_ranker',hyperparameters={'n_estimators':10}),task,{})
    c=context(task,tmp_path);c['strategy']='group';p.fit(X,y,c);b=p.predict(X,c)
    pred=align_predictions(b,X.index.tolist(),task,'protocol','test')
    assert 0<=score_predictions(task,y,pred,b.prediction_type,X['query'].to_numpy())<=1


def test_multilabel_constant_output_and_resampling(tabular):
    X,y,t=tabular
    y['answer']=(y.answer>0).astype(int);y['constant']=np.zeros(len(y),dtype=int)
    task=t.model_copy(update={'kind':'multilabel','targets':['answer','constant'],'classes':{'answer':[0,1],'constant':[0]},'metric':MetricSpec(name='f1'),
        'output':OutputSpec(columns=['identifier','answer','constant'],prediction_type='labels')})
    p=build_pipeline(PipelineSpec(model='logistic'),task,{});c=context(task);p.fit(X,y,c);b=p.predict(X.iloc[:20],c)
    pred=align_predictions(b,X.index[:20].tolist(),task,'protocol','test')
    assert pred['constant'].shape==(20,1) and np.allclose(pred['constant'],1)
    assert score_predictions(task,y.iloc[:20],pred,b.prediction_type)>=0


@pytest.mark.parametrize('method',['random_over','smote'])
def test_resampling_is_fit_only(tabular,method):
    X,y,t=tabular;y['answer']=np.where(y.answer>1,'rare','common')
    task=t.model_copy(update={'kind':'classification','classes':{'answer':['common','rare']},'metric':MetricSpec(name='f1'),
        'output':OutputSpec(columns=['identifier','answer'],prediction_type='labels')})
    p=build_pipeline(PipelineSpec(model='logistic',resampling=method),task,{})
    p.fit(X,y,context(task));b=p.predict(X.iloc[:6],context(task))
    assert len(b.row_keys)==6 and p.summary.rows==len(X)
