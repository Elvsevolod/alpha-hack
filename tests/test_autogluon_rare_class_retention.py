"""Native AutoGluon must not discard allowed rows/classes under Macro F1."""
import json
import numpy as np
import pandas as pd
import pytest
from automl_agent.contracts import TaskSpec,MetricSpec,OutputSpec,PipelineSpec
from automl_agent.search import SearchPipeline
from automl_agent.validation import inner_split


@pytest.mark.parametrize('profile',['selection','bagging','stacking'])
def test_macro_f1_keeps_rare_class_and_every_permitted_refit_row(tmp_path,monkeypatch,record_property,profile):
    pytest.importorskip('autogluon.tabular')
    from autogluon.tabular.models.lgb.lgb_model import LGBModel
    history=[];original=LGBModel._fit
    def observe(self,X,y,*args,**kwargs):
        history.append({'rows':len(X),'missing_labels':int(pd.isna(y).sum()),
            'encoded_class_counts':{str(k):int(v) for k,v in pd.Series(y).value_counts().items()}})
        return original(self,X,y,*args,**kwargs)
    monkeypatch.setattr(LGBModel,'_fit',observe)
    n=1200
    labels=np.where(np.arange(n)%2,'B','A').astype(object);labels[np.arange(0,n,200)]='rare'
    X=pd.DataFrame({'x':np.arange(n)%7,'rare_hint':(labels=='rare').astype(int)},index=['train:'+str(i) for i in range(n)])
    y=pd.DataFrame({'answer':labels},index=X.index)
    task=TaskSpec(kind='classification',targets=['answer'],id_columns=['id'],feature_columns=list(X),
        classes={'answer':['rare','B','A']},metric=MetricSpec(name='f1',average='macro'),
        output=OutputSpec(columns=['id','answer'],prediction_type='labels'),source_hash='synthetic_rare')
    spec=PipelineSpec(engine='autogluon',model='library_search',learners=['GBM'],autogluon_profile=profile,bag_folds=2,
        autogluon_configurations={'GBM':[{'n_estimators':5,'min_child_samples':1,'num_leaves':7}]})
    context={'strategy':'random','cpu':1,'budget_seconds':45,'memory_mb':2048,'workdir':str(tmp_path),
        'task':task.model_dump(mode='json'),'protocol_hash':'fixed','data_hash':'synthetic_rare','split_id':'fit'}
    model=SearchPipeline(spec,task)
    model.fit(X.iloc[:1000],y.iloc[:1000],context)
    fit,_=inner_split(X.iloc[:1000],y.iloc[:1000],task,'random',spec.seed)
    record_property('fit_history',json.dumps(history,sort_keys=True))
    predictor=model.models['answer'][1]
    assert set(predictor.class_labels_internal_map)==set(task.classes['answer']),predictor.class_labels_internal_map
    if profile=='selection':
        assert [h['rows'] for h in history]==[len(fit),1000],history
    else:
        assert history[-1]['rows']==1000,history
    model.refit(X,y,{**context,'execution_mode':'refit'})
    record_property('complete_fit_history',json.dumps(history,sort_keys=True))
    assert history[-1]['rows']==1200 and all(h['missing_labels']==0 for h in history),history
    if profile=='selection':
        assert len(history)==3,history
    bundle=model.predict(X.iloc[:10],context)
    assert bundle.classes==task.classes
    probability=np.asarray(bundle.values['answer'])
    assert probability.shape==(10,3) and np.allclose(probability.sum(axis=1),1)
    assert probability[:,0].max()>0


@pytest.mark.parametrize('profile',['bagging','stacking'])
def test_singleton_class_rejected_before_native_predictor_construction(tabular,tmp_path,monkeypatch,profile):
    ag=pytest.importorskip('autogluon.tabular')
    called=[]
    def forbidden(**kwargs):
        called.append(True)
        raise AssertionError('Native predictor must not drop the singleton')
    monkeypatch.setattr(ag,'TabularPredictor',forbidden)
    X,y,task=tabular
    y['answer']=np.where(np.arange(len(y))%2,'A','B');y.iloc[0,0]='rare'
    task=task.model_copy(update={'kind':'classification','classes':{'answer':['A','B','rare']},
        'metric':MetricSpec(name='f1',average='macro'),'output':OutputSpec(columns=['answer'],prediction_type='labels')})
    spec=PipelineSpec(engine='autogluon',model='library_search',learners=['GBM'],autogluon_profile=profile,bag_folds=2)
    with pytest.raises(ValueError,match='two inner-fit rows'):
        SearchPipeline(spec,task).fit(X,y,{'strategy':'random','cpu':1,'budget_seconds':30,'memory_mb':2048,'workdir':str(tmp_path)})
    assert not called


def test_undeclared_refit_class_is_rejected_before_native_fit_and_preserves_original(tabular,tmp_path,monkeypatch):
    pytest.importorskip('autogluon.tabular')
    from autogluon.tabular.models.lgb.lgb_model import LGBModel
    X,y,task=tabular
    y['answer']=np.where(np.arange(len(y))%2,'A','B');y.iloc[140:,0]='rare'
    task=task.model_copy(update={'kind':'classification','classes':{'answer':['A','B']},
        'metric':MetricSpec(name='f1',average='macro'),'output':OutputSpec(columns=['answer'],prediction_type='labels')})
    calls=[];original=LGBModel._fit
    def observe(self,X,y,*args,**kwargs):
        calls.append({'rows':len(X),'missing_labels':int(pd.isna(y).sum())})
        return original(self,X,y,*args,**kwargs)
    monkeypatch.setattr(LGBModel,'_fit',observe)
    spec=PipelineSpec(engine='autogluon',model='library_search',learners=['GBM'],
        autogluon_configurations={'GBM':[{'n_estimators':3,'min_child_samples':1}]})
    context={'strategy':'random','cpu':1,'budget_seconds':30,'memory_mb':2048,'workdir':str(tmp_path),
        'task':task.model_dump(mode='json'),'protocol_hash':'fixed','data_hash':'fixture','split_id':'fit'}
    model=SearchPipeline(spec,task);model.fit(X.iloc[:140],y.iloc[:140],context)
    predictor=model.models['answer'][1]
    names=predictor.model_names();before=model.predict(X.iloc[:8],context).values
    with pytest.raises(ValueError,match='outside the confirmed class space'):
        model.refit(X,y,context)
    assert len(calls)==2 and all(item['missing_labels']==0 for item in calls)
    assert model.models['answer'][1] is predictor and predictor.model_names()==names
    assert model.predict(X.iloc[:8],context).values==before
