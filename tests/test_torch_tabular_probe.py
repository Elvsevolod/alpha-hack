import joblib
import numpy as np
import pandas as pd
import pytest

from tools.torch_tabular_probe import InputTransform,TorchMLP,stopping_strategy


@pytest.mark.parametrize('encoding',['standard','piecewise','quantile'])
def test_fit_only_transform_unknown_missing_constant_and_labels(encoding):
    frame=pd.DataFrame({'x':[0.,1.,2.,np.nan], 'constant':[4.]*4, 'empty':[np.nan]*4,
                        'kind':['a','b','a',None], 'target':[100,200,300,400]})
    t=InputTransform(['x','constant','empty','kind'],['kind'],encoding).fit(frame)
    q=pd.DataFrame({'x':[1e9,np.nan], 'constant':[4.,4.], 'empty':[np.nan,np.nan],
                    'kind':['new',None], 'target':[-1,-2]})
    n,c=t.transform(q)
    assert c[:,0].tolist()==[0,1]
    assert np.isfinite(n).all()
    altered=q.assign(target=[9999,9999])
    nn,cc=t.transform(altered)
    np.testing.assert_array_equal(nn,n);np.testing.assert_array_equal(cc,c)
    assert t.maps['kind']=={'a':2,'b':3}
    np.testing.assert_array_equal(t.center,[1.,4.,0.])


@pytest.mark.parametrize('classification',[False,True])
def test_torch_from_scratch_roundtrip_query_labels_and_determinism(classification,tmp_path):
    pytest.importorskip('torch')
    X=pd.DataFrame({'x':np.linspace(-2,2,96),'category':['a','b']*48})
    labels=np.where(X.x>0,'P','N') if classification else 3*X.x.to_numpy()+5
    classes=['N','P'] if classification else None
    kwargs=dict(columns=['x','category'],categorical=['category'],classes=classes,
                encoding='piecewise',epochs=2,seconds=10,seed=42)
    a=TorchMLP(**kwargs).fit(X,labels)
    b=TorchMLP(**kwargs).fit(X,labels)
    q=X.assign(target=999)
    p=a.predict(q)
    np.testing.assert_allclose(b.predict(q),p,rtol=1e-6)
    joblib.dump(a,tmp_path/'state.joblib')
    np.testing.assert_allclose(joblib.load(tmp_path/'state.joblib').predict(q),p,rtol=1e-6)
    assert np.isfinite(p).all() and a.epochs_run==2
    if classification:
        assert p.shape==(96,2)
        np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-6)


def test_current_agent_boundary_is_explicit_not_a_claimed_live_run():
    from automl_agent.custom import validate_code
    from automl_agent.dependency_policy import permitted_module
    assert not permitted_module('torch')
    with pytest.raises(ValueError,match='outside the candidate policy'):
        validate_code('import torch\ndef build_pipeline(context):\n    return None\n','pipeline')


@pytest.mark.parametrize('mode,outer,inner',[
    ('holdout','temporal','temporal'),('exact_group','random','exact_group'),
    ('low_overlap','random','exact_group'),('holdout','group','group')])
def test_neural_stopping_preserves_declared_dependency(mode,outer,inner):
    from types import SimpleNamespace
    assert stopping_strategy(SimpleNamespace(evaluation_mode=mode,strategy=outer))==inner


@pytest.mark.parametrize('classification',[False,True])
def test_shared_weight_heads_train_all_members_and_average_at_inference(classification):
    torch=pytest.importorskip('torch')
    from tools.torch_tabular_probe import network
    X=pd.DataFrame({'x':np.linspace(-2,2,64),'cat':['A','B']*32})
    classes=['negative','positive'] if classification else None
    y=np.where(X.x>0,'positive','negative') if classification else X.x.to_numpy()*3
    model=TorchMLP(['x','cat'],['cat'],classes,encoding='quantile',heads=4,width=16,depth=2,epochs=2)
    model.fit(X,y)
    p=model.predict(X)
    net=network(*model.shape,width=16,depth=2,heads=4)
    net.load_state_dict({k:torch.from_numpy(v) for k,v in model.state.items()})
    net.eval();a,b=model.transformer.transform(X)
    with torch.inference_mode():
        z=net(torch.from_numpy(a),torch.from_numpy(b))
        expected=torch.softmax(z,dim=-1).mean(dim=1).numpy() if classification else z[:,:,0].mean(dim=1).numpy()*model.scale+model.mean
    np.testing.assert_allclose(p,expected,rtol=1e-6)
    assert len([k for k in model.state if k.endswith('.r')])==3


def test_dual_numeric_keeps_numeric_coordinate_and_learns_fit_only_tokens():
    x=pd.DataFrame({'number':[0.,1.,2.,np.nan], 'continuous':[.1,.2,.3,.4], 'cat':['A','B','A','B']})
    t=InputTransform(list(x),['cat'],dual_numeric=True).fit(x)
    n,c=t.transform(pd.DataFrame({'number':[1.,99.,np.nan],'continuous':[.2,.5,.8],'cat':['A','new',None]}))
    assert t.numeric==['number','continuous']
    assert t.encoded_categorical==['cat','number','continuous']
    assert c[:,1].tolist()==[3,0,1]
    assert n.shape==(3,4) and c.shape==(3,3)


def test_temporal_repetition_ablation_is_explicit_role_driven_and_fit_only():
    from tools.torch_temporal_features import temporal_feature_group
    frame=pd.DataFrame({'when':np.repeat([1,2],6),'category':['A','B']*6,
        'low':[0,1]*6,'rich':np.arange(12)})
    kept,dropped,support=temporal_feature_group(frame,list(frame),['category'],'when',limit=3)
    assert kept==['category','rich'] and set(dropped)=={'when','low'}
    assert support['rich']==6
    with pytest.raises(ValueError):temporal_feature_group(frame,list(frame),['category'],None)


@pytest.mark.parametrize('encoding',['standard','piecewise','quantile'])
def test_category_only_task_and_empty_prediction(encoding):
    pytest.importorskip('torch')
    X=pd.DataFrame({'kind':['a','b']*20})
    model=TorchMLP(['kind'],['kind'],['N','P'],encoding=encoding,epochs=1)
    model.fit(X,np.array(['N','P']*20))
    assert model.transformer.transform(X)[0].shape==(40,0)
    p=model.predict(pd.DataFrame({'kind':['new',None]}))
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1),1,rtol=1e-6)
    assert model.predict(X.iloc[:0]).shape==(0,2)


@pytest.mark.parametrize('parameter',[{'epochs':0},{'seconds':0},{'heads':0}])
def test_invalid_training_budget_is_rejected_before_fit(parameter):
    with pytest.raises(ValueError,match='Positive neural'):
        TorchMLP(['x'],[],**parameter)


def test_empty_fit_rejected_without_fitting_statistics():
    with pytest.raises(ValueError,match='nonempty fit'):
        InputTransform(['x'],[]).fit(pd.DataFrame({'x':[]}))


def test_first_pilot_checkpoint_keeps_inference_compatibility(tmp_path):
    pytest.importorskip('torch')
    X=pd.DataFrame({'x':np.linspace(-1,1,20)})
    model=TorchMLP(['x'],[],epochs=1).fit(X,X.x.to_numpy())
    expected=model.predict(X)
    for name in ['width','depth','heads','learning_rate','dual_numeric']:
        delattr(model,name)
    for name in ['dual_numeric','encoded_categorical','quantile']:
        delattr(model.transformer,name)
    joblib.dump(model,tmp_path/'old.joblib')
    np.testing.assert_allclose(joblib.load(tmp_path/'old.joblib').predict(X),expected,rtol=1e-6)


@pytest.mark.parametrize('timeout',[False,True])
def test_failed_worker_preserves_reason_and_fails_series(tmp_path,monkeypatch,timeout):
    import json
    import subprocess
    from types import SimpleNamespace
    from tools.torch_tabular_probe import run
    def failed(*args,**kwargs):
        assert 'DEEPSEEK_API_KEY' not in kwargs['env']
        assert 'QWEN_API_KEY' not in kwargs['env']
        if timeout:raise subprocess.TimeoutExpired(args[0],kwargs['timeout'])
        return SimpleNamespace(returncode=9)
    monkeypatch.setenv('DEEPSEEK_API_KEY','synthetic-test-secret')
    monkeypatch.setenv('QWEN_API_KEY','synthetic-test-secret')
    monkeypatch.setattr(subprocess,'run',failed)
    directory=tmp_path/'failed_probe'
    with pytest.raises(RuntimeError,match='Offline probe failed'):
        run(directory,dual=True)
    for kind in ['regression','classification']:
        reason=json.loads((directory/kind/'failure.json').read_text())
        assert reason==({'reason':'worker_timeout','seconds':420} if timeout else {'returncode':9})
