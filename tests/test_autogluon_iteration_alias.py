"""Native iteration aliases must not override AutoGluon's selected fit rounds."""
import json
import numpy as np
import pandas as pd
import pytest
from automl_agent.contracts import TaskSpec,MetricSpec,OutputSpec,PipelineSpec
from automl_agent.search import SearchPipeline


@pytest.mark.parametrize('alias',['n_estimators','num_iterations','num_boost_round'])
@pytest.mark.parametrize('kind',['regression','classification'])
def test_explicit_round_cap_does_not_override_selected_rounds_in_refit(tmp_path,monkeypatch,record_property,alias,kind):
    pytest.importorskip('autogluon.tabular')
    from autogluon.tabular.models.lgb.lgb_model import LGBModel
    history=[]
    original=LGBModel._fit
    def watch(self,X,y,*args,**kwargs):
        result=original(self,X,y,*args,**kwargs)
        history.append({'rows':len(X),'actual_rounds':self.model.current_iteration(),
            'recorded_rounds':self.params_trained['num_boost_round'],
            'input_parameters':self._get_model_params(),'validation':kwargs.get('X_val') is not None})
        return result
    monkeypatch.setattr(LGBModel,'_fit',watch)
    monkeypatch.setattr('automl_agent.search.inner_split',lambda *args:(np.arange(60),np.arange(60,100)))
    X=pd.DataFrame({'x':np.arange(120,dtype=float)},index=['train:'+str(i) for i in range(120)])
    y=pd.DataFrame({'answer':np.r_[np.arange(60),-np.arange(60,120)]},index=X.index)
    task=TaskSpec(kind='regression',targets=['answer'],id_columns=['id'],feature_columns=['x'],
        metric=MetricSpec(name='rmse'),output=OutputSpec(columns=['id','answer'],prediction_type='values'),source_hash='synthetic')
    if kind=='classification':
        y['answer']=np.r_[np.where(np.arange(60)>30,'high','low'),np.full(60,'low')]
        task=task.model_copy(update={'kind':'classification','classes':{'answer':['low','high']},
            'metric':MetricSpec(name='f1',average='macro'),
            'output':OutputSpec(columns=['id','answer'],prediction_type='labels')})
    spec=PipelineSpec(engine='autogluon',model='library_search',learners=['GBM'],
        autogluon_configurations={'GBM':[{alias:30,'min_data_in_leaf':1,'learning_rate':.5}]})
    model=SearchPipeline(spec,task)
    context={'strategy':'random','cpu':1,'budget_seconds':45,'memory_mb':2048,'workdir':str(tmp_path),
        'task':task.model_dump(mode='json'),'protocol_hash':'fixed','data_hash':'synthetic','split_id':'fit'}
    model.fit(X.iloc[:100],y.iloc[:100],context)
    model.refit(X,y,{**context,'execution_mode':'refit'})
    record_property('fit_history',json.dumps(history,sort_keys=True))
    assert len(history)==3,history
    assert history[0]['actual_rounds']<30,history
    assert [h['rows'] for h in history]==[60,100,120],history
    assert len({h['actual_rounds'] for h in history})==1,history
    assert all(h['actual_rounds']==h['recorded_rounds'] for h in history),history


@pytest.mark.parametrize('other',['n_estimators','num_iterations'])
def test_native_round_alias_conflicts_are_rejected_and_equal_values_merge(tabular,other):
    from automl_agent.search import autogluon_portfolio
    task=tabular[2]
    spec=PipelineSpec(engine='autogluon',model='library_search',learners=['GBM'],
        autogluon_configurations={'GBM':[{'num_boost_round':7,other:8}]})
    with pytest.raises(ValueError,match='Conflicting'):
        autogluon_portfolio(spec,task,['GBM'])
    equal=spec.model_copy(update={'autogluon_configurations':{'GBM':[{'num_boost_round':7,other:7}]}})
    emitted=autogluon_portfolio(equal,task,['GBM'])['GBM'][0]
    assert emitted['num_boost_round']==7
    assert 'n_estimators' not in emitted and 'num_iterations' not in emitted
    assert equal.autogluon_configurations['GBM'][0][other]==7
