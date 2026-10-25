"""Small offline reproductions of role, resource and cache transmission defects."""
from types import SimpleNamespace

import pandas as pd
import pytest

from automl_agent.contracts import PipelineSpec,TaskSpec
from automl_agent.util import atomic_json,digest,file_hash


@pytest.mark.parametrize('storage',['float64','Int64'])
@pytest.mark.parametrize('model',['ridge','lightgbm'])
def test_numeric_storage_does_not_override_declared_categorical_role(storage,model):
    from automl_agent.preprocessing import FeatureMatrix
    task=TaskSpec(kind='regression',targets=['y'],feature_columns=['code','value'],categorical_columns=['code'],
        metric={'name':'rmse'},output={'columns':['y'],'prediction_type':'values'},source_hash='fixture')
    X=pd.DataFrame({'code':pd.Series([10,20,10,None],dtype=storage),'value':[1.,2.,3.,4.]})
    matrix=FeatureMatrix(PipelineSpec(model=model),task)
    matrix.fit(X,pd.DataFrame({'y':[0.,1.,0.,1.]}))
    assert matrix.categorical_columns==['code']
    assert matrix.numeric_columns==['value']
    assert matrix.transform(X.assign(code=999)).shape[1]==matrix.width


def test_startup_is_fixed_when_old_summary_has_no_execution_phases():
    from automl_agent.cost_model import phase_forecast
    summary={'rows':100,'seconds':5.,'budget':{'selected_fit_actual_seconds':1.,'selected_fit_rows':100,
        'worker_initialization_seconds':3.}}
    without={**summary,'budget':{**summary['budget'],'worker_initialization_seconds':0.}}
    assert phase_forecast(summary,6.,1000,100)-phase_forecast(without,6.,1000,100)==pytest.approx(3.*1.25)


@pytest.mark.parametrize('mismatch',['protocol','pipeline','rows','request_hash'])
def test_successful_oof_cache_rejects_foreign_identity_without_refitting(tmp_path,mismatch):
    from automl_agent.learned_blend import cached_oof,OOFCacheError
    task=TaskSpec(kind='regression',targets=['y'],feature_columns=['x'],metric={'name':'rmse'},
        output={'columns':['y'],'prediction_type':'values'},source_hash='fixture')
    spec=SimpleNamespace(pipeline=PipelineSpec(model='ridge'),library_versions={'numpy':'fixture'})
    c=SimpleNamespace(task=task,plan=SimpleNamespace(hash='ours'),dataset=SimpleNamespace(source_hash='fixture',row_keys=['a','b']),workdir=tmp_path)
    request={'mode':'oof','task':task.model_dump(mode='json'),'pipeline':spec.pipeline.model_dump(mode='json'),
        'context':{'data_hash':'fixture','protocol_hash':'ours','oof_seed':42,'library_versions':spec.library_versions}}
    if mismatch=='protocol':request['context']['protocol_hash']='foreign'
    if mismatch=='pipeline':request['pipeline']['hyperparameters']={'alpha':77.}
    atomic_json(tmp_path/'request.json',request)
    for name in ['fit_X','fit_y']:atomic_json(tmp_path/(name+'.json'),{'index':['foreign'] if mismatch=='rows' else ['a','b']})
    atomic_json(tmp_path/'oof_predictions.json',{'values':{'y':[3.,4.]}})
    atomic_json(tmp_path/'worker_result.json',{'status':'completed','request_hash':'foreign' if mismatch=='request_hash' else digest(request),
        'hashes':{'oof_predictions.json':file_hash(tmp_path/'oof_predictions.json')}})
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(OOFCacheError):cached_oof(c,'member',spec,[0,1],tmp_path)
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
