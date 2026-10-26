import json,time,hashlib,importlib.metadata
from pathlib import Path
import numpy as np,pandas as pd
from autogluon.tabular.models.lgb.lgb_model import LGBModel
from autogluon.tabular.models.catboost.catboost_model import CatBoostModel
from autogluon.core.data import LabelCleaner
from automl_agent.contracts import TaskSpec,MetricSpec,OutputSpec,PipelineSpec
from automl_agent.search import autogluon_portfolio
root=Path('outputs/autogluon_weight_audit_20261026')
rng=np.random.default_rng(2026)
X=pd.DataFrame(rng.normal(size=(180,3)),columns=['a','b','c'])
y=pd.Series(np.where(X.a+rng.normal(size=180)>.8,2,10),name='answer')
task=TaskSpec(kind='classification',targets=['answer'],id_columns=[],feature_columns=list(X),classes={'answer':[2,10]},metric=MetricSpec(name='f1',average='macro'),output=OutputSpec(columns=['answer'],prediction_type='labels'),source_hash='fixed_synthetic')
cleaner=LabelCleaner.construct(problem_type='binary',y=y,positive_class=2)
encoded=cleaner.transform(y)
results={};pred={}
for name,family,modelclass,p in [('gbm_default','GBM',LGBModel,{}),('gbm_balanced','GBM',LGBModel,{'class_weight':'balanced'}),('gbm_dict','GBM',LGBModel,{'class_weight':{'2':8,'10':1}}),('cat_dict','CAT',CatBoostModel,{'class_weights':{'2':8,'10':1}})]:
 params={'n_estimators':4,'min_child_samples':1,'verbosity':-1} if family=='GBM' else {'iterations':4,'depth':2}
 params.pop('verbosity',None)
 params.update(p)
 spec=PipelineSpec(engine='autogluon',model='library_search',learners=[family],autogluon_configurations={family:[params]})
 start=time.monotonic()
 try:
  emitted=autogluon_portfolio(spec,task,[family])[family][0]
  emitted.pop('ag_args',None)
  model=modelclass(path=str((root/name).resolve()),name=name,problem_type='binary',eval_metric='f1_macro',hyperparameters=emitted)
  model.fit(X=X,y=encoded,num_cpus=1,num_gpus=0,time_limit=20,verbosity=0)
  probability=model.predict_proba(X)
  pred[name]=probability
  results[name]={'status':'completed','prediction_sha256':hashlib.sha256(probability.tobytes()).hexdigest()}
 except Exception as error:
  results[name]={'status':'error','error_type':type(error).__name__,'error':str(error)[:700]}
 results[name]['seconds']=time.monotonic()-start
if 'gbm_default' in pred and 'gbm_balanced' in pred:results['gbm_balanced']['max_abs_difference_from_default']=float(np.max(np.abs(pred['gbm_default']-pred['gbm_balanced'])))
result={'source_commit':'3b37cd9','label_map':{str(k):int(v) for k,v in cleaner.inv_map.items()},'class_counts':{str(k):int(v) for k,v in y.value_counts().items()},'results':results,'api_calls':0,'versions':{n:importlib.metadata.version(n) for n in ['autogluon.tabular','autogluon.core','pandas','numpy','scikit-learn','lightgbm','catboost']}}
(root/'observed.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
