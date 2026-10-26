"""Refit a selected native learner on new, already-declared labels without HPO."""
import cloudpickle
import numpy as np
import pandas as pd
import pytest
from automl_agent.contracts import TaskSpec,MetricSpec,OutputSpec,PipelineSpec
from automl_agent.search import SearchPipeline


@pytest.mark.parametrize('initial_classes',[2,3])
@pytest.mark.parametrize('numeric',[False,True])
@pytest.mark.parametrize('family',['GBM','CAT','XGB'])
def test_declared_new_class_rebuilds_label_space_with_one_fixed_fit(tmp_path,monkeypatch,record_property,family,initial_classes,numeric):
    ag=pytest.importorskip('autogluon.tabular')
    from autogluon.tabular.models.lgb.lgb_model import LGBModel
    from autogluon.tabular.models.catboost.catboost_model import CatBoostModel
    from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel
    import automl_agent.search as search
    cls={'GBM':LGBModel,'CAT':CatBoostModel,'XGB':XGBoostModel}[family]
    fits=[];searches=[];splits=[]
    oldfit,oldsearch,oldsplit=cls._fit,ag.TabularPredictor.fit,search.inner_split
    def fit(self,X,y,*args,**kwargs):
        result=oldfit(self,X,y,*args,**kwargs)
        fits.append({'rows':len(X),'missing':int(pd.isna(y).sum()),'classes':len(pd.unique(y)),
            'validation':kwargs.get('X_val') is not None,'rounds':dict(self.params_trained),
            'actual_rounds':self.model.current_iteration() if family=='GBM' else self.model.tree_count_ if family=='CAT' else self.model.get_booster().num_boosted_rounds()})
        return result
    def selection(self,*args,**kwargs):
        searches.append(True);return oldsearch(self,*args,**kwargs)
    def split(*args,**kwargs):
        splits.append(True);return oldsplit(*args,**kwargs)
    monkeypatch.setattr(cls,'_fit',fit);monkeypatch.setattr(ag.TabularPredictor,'fit',selection)
    monkeypatch.setattr(search,'inner_split',split)
    n=180;index=['train:'+str(i) for i in range(n)]
    X=pd.DataFrame({'x':np.arange(n)%11,'kind':np.where(np.arange(n)%2,'odd','even')},index=index)
    names=['A','B','C'][:initial_classes]
    y=pd.DataFrame({'answer':np.asarray(names)[np.arange(n)%initial_classes]},index=index);y.iloc[140:,0]='rare'
    classes=['rare',*reversed(names)]
    if numeric:
        mapping={'A':2,'B':10,'C':30,'rare':99}
        y['answer']=y.answer.map(mapping);classes=[mapping[c] for c in classes]
    task=TaskSpec(kind='classification',targets=['answer'],feature_columns=list(X),categorical_columns=['kind'],
        classes={'answer':classes},metric=MetricSpec(name='f1',average='macro'),
        output=OutputSpec(columns=['answer'],prediction_type='labels'),source_hash='declared_class_fixture')
    params={'GBM':{'n_estimators':5,'min_child_samples':1},'CAT':{'iterations':5,'depth':2},'XGB':{'n_estimators':5,'max_depth':2}}[family]
    spec=PipelineSpec(engine='autogluon',model='library_search',learners=[family],autogluon_configurations={family:[params]})
    ctx={'strategy':'random','cpu':1,'memory_mb':2048,'budget_seconds':45,'workdir':str(tmp_path),
        'task':task.model_dump(mode='json'),'protocol_hash':'fixed','data_hash':'synthetic','split_id':'fit'}
    model=SearchPipeline(spec,task);initial=model.fit(X.iloc[:140],y.iloc[:140],ctx)
    assert initial.budget['class_work']['targets']['answer']=={'observed_classes':initial_classes,'declared_classes':initial_classes+1}
    original=model.models['answer'][1];names=original.model_names();before=original.predict_proba(X.iloc[:8]).copy()
    selected_rounds=fits[-1]['actual_rounds']
    refitted=model.refit(X,y,{**ctx,'execution_mode':'refit'})
    assert model.summary.rows==refitted.rows
    assert model.summary.budget==refitted.budget
    assert refitted.budget['class_work']['targets']['answer']=={'observed_classes':initial_classes+1,'declared_classes':initial_classes+1}
    assert len(fits)==3 and fits[-1]['rows']==180 and fits[-1]['classes']==initial_classes+1
    assert fits[-1]['missing']==0 and not fits[-1]['validation']
    assert fits[-1]['actual_rounds']==selected_rounds
    assert len(searches)==len(splits)==1
    result=model.predict(X.iloc[-8:],ctx)
    assert result.classes==task.classes
    probability=np.asarray(result.values['answer'])
    assert probability.shape==(8,initial_classes+1) and np.allclose(probability.sum(axis=1),1)
    assert (probability[:,0]>0).all()
    restored=cloudpickle.loads(cloudpickle.dumps(model))
    assert restored.predict(X.iloc[-8:],ctx).values==result.values
    restored.refit(X,y,{**ctx,'execution_mode':'refit'})
    assert len(fits)==4 and fits[-1]['actual_rounds']==selected_rounds and not fits[-1]['validation']
    assert len(searches)==len(splits)==1
    assert original.model_names()==names and original.predict_proba(X.iloc[:8]).equals(before)
    record_property('native_fits',__import__('json').dumps(fits,sort_keys=True))
