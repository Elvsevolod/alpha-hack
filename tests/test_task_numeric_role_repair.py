import json

import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import TaskResolutionProposal
from automl_agent.controller import Controller
from automl_agent.data import Dataset
from conftest import write_task


def proposal(data, request_id, categorical):
    spec=json.loads((data.task_path.parent/'task_spec.json').read_text())
    spec.update(contract_version=2,source_hash=data.source_hash,
        file_roles={n:n for n in ('train.csv','test.csv','sample_submission.csv')},
        categorical_columns=['code'] if categorical else [])
    return {'request_id':request_id,'source_hash':data.source_hash,'status':'proposed','task_spec':spec,
        'evidence':{field:['task','inventory']for field in spec}}


def test_numeric_role_mismatch_is_rejected_before_contract_confirmation(tmp_path):
    data=Dataset(write_task(tmp_path),deferred=True)
    original_sources=data.sources
    with pytest.raises(ValueError,match='Non-numeric value in numeric feature code'):
        data.confirm_proposal(TaskResolutionProposal.model_validate(proposal(data,'request',False)),'request')
    assert not hasattr(data,'train') and data.task is None and data.sources is original_sources
    corrected=data.confirm_proposal(TaskResolutionProposal.model_validate(proposal(data,'request',True)),'request')
    assert data.activate(corrected).categorical_columns==['code']


def test_controller_requests_semantic_repair_instead_of_failing_after_confirmation(tmp_path,monkeypatch):
    c=Controller(write_task(tmp_path),Limits(seconds=120,cpu=1));requests=[]
    monkeypatch.setattr(c.qwen,'key','offline-fixture');monkeypatch.setattr(c.qwen,'base_url','http://offline')
    def resolve(system,request):
        requests.append(request)
        if len(requests)==2:
            assert 'Non-numeric value in numeric feature code' in request['validation_errors'][0]
        assert len(requests)<=2
        return proposal(c.dataset,request['request_id'],len(requests)==2)
    monkeypatch.setattr(c.qwen,'resolve',resolve)
    try:
        task=c._task()
        assert len(requests)==2 and task.categorical_columns==['code']
        assert (tmp_path/'submission.csv').read_bytes()==(tmp_path/'sample_submission.csv').read_bytes()
    finally:c.close()


@pytest.mark.parametrize('file_name',['train.csv','test.csv'])
def test_all_invalid_feature_names_are_reported_together_without_values(tmp_path,file_name):
    import pandas as pd
    path=write_task(tmp_path)
    frame=pd.read_csv(tmp_path/file_name,dtype=str,keep_default_na=False)
    frame.loc[0,'value']='PRIVATE_FEATURE_SENTINEL'
    frame.to_csv(tmp_path/file_name,index=False)
    data=Dataset(path,deferred=True)
    with pytest.raises(ValueError)as error:
        data.confirm_proposal(TaskResolutionProposal.model_validate(proposal(data,'request',False)),'request')
    assert 'value' in str(error.value) and 'code' in str(error.value)
    assert 'PRIVATE_FEATURE_SENTINEL' not in str(error.value) and not hasattr(data,'train')
