"""Fresh guarded worker, serialization and CSV publication for declared labels."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import cloudpickle
import numpy as np
import pandas as pd
import pytest
from automl_agent.contracts import TaskSpec,MetricSpec,OutputSpec,PipelineSpec,PredictionBundle
from automl_agent.search import SearchPipeline
from automl_agent.executor import Executor,stage_frame
from automl_agent.journal import Journal
from automl_agent.publication import Publisher


@pytest.mark.parametrize('family',['GBM','CAT','XGB'])
def test_class_expansion_survives_guarded_worker_and_csv(tmp_path,record_property,family):
    pytest.importorskip('autogluon.tabular')
    import autogluon.tabular
    import pydantic
    # The development overlay is explicitly readable; production dependencies
    # live under sys.prefix. No worker policy or production launcher is patched.
    roots=[str(Path(autogluon.tabular.__file__).resolve().parents[2]),str(Path(pydantic.__file__).resolve().parents[1])]
    off_prefix=any(not Path(root).is_relative_to(Path(sys.prefix).resolve()) for root in roots)
    if off_prefix and not Path(sys.prefix,'lib/python3.12/site-packages/audit_paths.pth').exists():
        pytest.skip('Requires the isolated development overlay registered for Python -I')
    index=['train:'+str(i) for i in range(180)]
    X=pd.DataFrame({'x':np.arange(180)%11},index=index)
    y=pd.DataFrame({'answer':np.where(np.arange(180)%2,2,10)},index=index);y.iloc[140:,0]=99
    task=TaskSpec(kind='classification',targets=['answer'],id_columns=['row'],feature_columns=['x'],
        classes={'answer':[99,10,2]},metric=MetricSpec(name='f1',average='macro'),
        output=OutputSpec(columns=['row','answer'],prediction_type='labels'),source_hash='worker_classes')
    parameters={'GBM':{'n_estimators':3,'min_child_samples':1},'CAT':{'iterations':3,'depth':2},'XGB':{'n_estimators':3,'max_depth':2}}[family]
    spec=PipelineSpec(engine='autogluon',model='library_search',learners=[family],autogluon_configurations={family:[parameters]})
    source=tmp_path/'source';source.mkdir()
    context={'cpu':1,'memory_mb':2048,'budget_seconds':30,'strategy':'random','workdir':str(source),
        'task':task.model_dump(mode='json'),'protocol_hash':'fixed','data_hash':'worker_classes','split_id':'fit'}
    pipeline=SearchPipeline(spec,task);pipeline.fit(X.iloc[:140],y.iloc[:140],context)
    with (source/'model.pkl').open('wb') as stream:cloudpickle.dump(pipeline,stream)
    attempt=tmp_path/'attempt';attempt.mkdir()
    test=X.iloc[-8:].copy();test.index=['test:'+str(i) for i in range(8)]
    stage_frame(attempt/'fit_X.json',X);stage_frame(attempt/'fit_y.json',y);stage_frame(attempt/'test_X.json',test)
    request={'mode':'refit','pipeline':spec.model_dump(mode='json'),'task':task.model_dump(mode='json'),
        'model_path':str(source/'model.pkl'),'read_dependencies':[str(source),*roots],
        'context':context.copy(),'predict_splits':['test']}
    journal=Journal(tmp_path/'controller')
    try:
        result=Executor(1,2048,time.time()+60,journal).run(attempt,request,45)
        assert result['status']=='completed',result
        assert result['serialization_verified'] and result['fit_rows']==180
        bundle=PredictionBundle.model_validate_json((attempt/'test_predictions.json').read_text())
        source.rename(tmp_path/'retired_source')
        with (attempt/'model.pkl').open('rb') as stream:restored=cloudpickle.load(stream)
        assert restored.predict(test,{**context,'split_id':'test'}).values==bundle.values
        assert restored.summary.budget==result['training_summary']['budget']
        assert restored.summary.budget['class_work']['targets']['answer']=={'observed_classes':3,'declared_classes':3}
        ids=[str(i).zfill(6) for i in range(8)]
        data=SimpleNamespace(task=task,test=test.assign(row=ids),test_keys=list(test.index),
            task_path=tmp_path/'task.md',source_hash=task.source_hash)
        publisher=Publisher(data,SimpleNamespace(hash='fixed'),tmp_path,journal)
        publisher.publish(bundle,'candidate','full_train_refit',details={'rows':180})
        csv=pd.read_csv(publisher.output,dtype={'row':str})
        assert list(csv)==['row','answer'] and csv.row.tolist()==ids and set(csv.answer)<={2,10,99}
        record_property('worker_measurement',json.dumps({k:result[k] for k in ['wall_seconds','peak_memory_mb','fit_rows','serialization_verified']}))
    finally:
        journal.close()
