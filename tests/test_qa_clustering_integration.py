import json
import time

import numpy as np
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.clustering import ClusteringContract, execute, silhouette_diagnostic
from automl_agent.executor import Executor


def fixture(root,degenerate=False):
    rng=np.random.default_rng(42)
    points=np.zeros(160) if degenerate else np.r_[rng.normal(-5,.1,80),rng.normal(5,.1,80)]
    pd.DataFrame({'id':[f'{i:05d}' for i in range(160)],'x':points,'empty':np.nan}).to_csv(root/'observations.csv',index=False)
    (root/'nested').mkdir()
    pd.DataFrame({'id':['00900','00700','00800'],'x':[-5.,5.,-5.1],'empty':np.nan}).to_csv(root/'nested/future.csv',index=False)
    spec=ClusteringContract(train_file='observations.csv',test_file='nested/future.csv',features=['x','empty'],id_columns=['id'],n_clusters=2)
    path=root/'clustering.json';path.write_text(spec.model_dump_json())
    return path


def test_target_free_bounded_worker_csv_and_verified_recovery(tmp_path,monkeypatch):
    path=fixture(tmp_path)
    report=execute(tmp_path,path,Limits(seconds=60,cpu=1,memory_mb=1024),time.time(),'exact')
    assert report['state']=='completed' and report['validation']['score']>.9
    actual=pd.read_csv(tmp_path/'submission.csv',dtype={'id':str})
    assert list(actual)==['id','cluster'] and actual.id.tolist()==['00900','00700','00800']
    assert actual.cluster.iloc[0]==actual.cluster.iloc[2]!=actual.cluster.iloc[1]
    process=json.loads((tmp_path/'outputs/clustering-exact/attempt/diagnostic_process_result.json').read_text())
    assert process['exit_code']==0 and not process['reason']
    expected=(tmp_path/'submission.csv').read_bytes();(tmp_path/'submission.csv').unlink()
    def forbidden(*args,**kwargs):raise AssertionError('Resume attempted another worker fit')
    monkeypatch.setattr(Executor,'run',forbidden)
    again=execute(tmp_path,path,Limits(seconds=60,cpu=1),time.time(),'exact')
    assert again['publication']==report['publication'] and (tmp_path/'submission.csv').read_bytes()==expected


def test_interrupted_publication_recovers_saved_result(tmp_path,monkeypatch):
    from automl_agent import util
    path=fixture(tmp_path);original=util.atomic_bytes
    def interrupted(path,content,mode=None):
        if path==tmp_path/'submission.csv':raise OSError('Injected publication interruption')
        return original(path,content,mode)
    monkeypatch.setattr(util,'atomic_bytes',interrupted)
    with pytest.raises(OSError,match='Injected'):
        execute(tmp_path,path,Limits(seconds=60,cpu=1),time.time(),'recover')
    monkeypatch.setattr(util,'atomic_bytes',original)
    monkeypatch.setattr(Executor,'run',lambda *a,**kw:pytest.fail('Unexpected refit'))
    assert execute(tmp_path,path,Limits(seconds=60,cpu=1),time.time(),'recover')['state']=='completed'


def test_degenerate_clusters_have_diagnostic_not_fake_score(tmp_path):
    path=fixture(tmp_path,degenerate=True)
    report=execute(tmp_path,path,Limits(seconds=60,cpu=1),time.time())
    assert report['state']=='completed'
    assert report['validation']['status']=='undefined_silhouette' and report['validation']['score'] is None
    assert silhouette_diagnostic(np.array([[0.],[1.],[10.],[11.]]),[0,0,1,1])['score']==pytest.approx(359/399)


@pytest.mark.parametrize('field,value',[('targets',['fake']),('train_file','../source.csv'),('test_file','submission.csv'),('features',['id']),('test_file','observations.csv'),('train_file',''),('test_file','.')])
def test_contract_rejects_fake_labels_and_role_overlap(tmp_path,field,value):
    path=fixture(tmp_path);spec=json.loads(path.read_text());spec[field]=value
    with pytest.raises(ValueError):ClusteringContract.model_validate(spec)


@pytest.mark.parametrize('link_kind',['symlink','hardlink'])
def test_source_alias_is_rejected_before_staging_and_fit(tmp_path,monkeypatch,link_kind):
    import os
    path=fixture(tmp_path)
    alias=tmp_path/'alias.csv'
    if link_kind=='symlink':alias.symlink_to(tmp_path/'observations.csv')
    else:os.link(tmp_path/'observations.csv',alias)
    spec=json.loads(path.read_text());spec['test_file']='alias.csv';path.write_text(json.dumps(spec))
    monkeypatch.setattr(Executor,'run',lambda *a,**kw:pytest.fail('Aliased input reached training'))
    with pytest.raises(ValueError,match='same source'):
        execute(tmp_path,path,Limits(seconds=60,cpu=1),time.time())
    assert not (tmp_path/'outputs').exists()


def test_source_change_and_expired_budget_are_not_retrained(tmp_path,monkeypatch):
    path=fixture(tmp_path)
    monkeypatch.setattr(Executor,'run',lambda *a,**kw:pytest.fail('No budget for worker'))
    report=execute(tmp_path,path,Limits(seconds=1,cpu=1),time.time()-10,'expired')
    assert report['error_code']=='time_budget_exhausted'
    source=tmp_path/'observations.csv';source.write_text(source.read_text().replace('-4.9695','-4.9696',1)+'\n')
    with pytest.raises(ValueError,match='another source'):
        execute(tmp_path,path,Limits(seconds=60,cpu=1),time.time(),'expired')
