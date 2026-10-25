"""Research core checks; this is not production capability promotion."""
import os
import subprocess
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from tools.torch_research_core import InputTransform,TorchMLP


def frame(n=50):
    return pd.DataFrame({'x':np.linspace(-2,2,n),'constant':1.,'missing':np.nan,'cat':['a',None]*(n//2)})


def test_fit_only_transform_unknown_missing_constant():
    X=frame();m=InputTransform(list(X),['cat'],'learned_piecewise').fit(X)
    med=m.median.copy();knots=[x.copy() for x in m.knots]
    query=X.iloc[:2].copy();query['cat']=['unknown',None];query['x']=1e10
    a,c=m.transform(query)
    assert np.isfinite(a).all() and c[:,0].tolist()==[0,1]
    np.testing.assert_array_equal(m.median,med)
    for a,b in zip(m.knots,knots):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('encoding,residual,heads',[('standard',False,1),('quantile',True,1),('learned_piecewise',False,1),('periodic',False,1),('quantile',False,4)])
def test_network_roundtrip_and_empty(tmp_path,encoding,residual,heads):
    X=frame();y=X.x**2
    model=TorchMLP(list(X),['cat'],encoding=encoding,residual=residual,heads=heads,epochs=3).fit(X,y)
    path=tmp_path/'model.joblib';joblib.dump(model,path)
    np.testing.assert_allclose(model.predict(X),joblib.load(path).predict(X),rtol=1e-6)
    assert model.predict(X[:0]).shape==(0,)
    np.save(tmp_path/'x.npy',model.predict(X))
    X.to_json(tmp_path/'frame.json')
    subprocess.run([sys.executable,'-c',"import joblib,numpy as np,pandas as pd,sys; from pathlib import Path; p=Path(sys.argv[1]); np.testing.assert_allclose(joblib.load(p/'model.joblib').predict(pd.read_json(p/'frame.json')),np.load(p/'x.npy'),rtol=1e-5,atol=1e-5)",str(tmp_path)],check=True,env={**os.environ,'OMP_NUM_THREADS':'1'})


def test_class_order_and_missing_class():
    X=frame();labels=np.array(['b','a']*25)
    model=TorchMLP(list(X),['cat'],classes=['b','a'],epochs=2).fit(X,labels)
    p=model.predict(X);assert p.shape==(50,2)
    np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
    assert model.predict(X[:0]).shape==(0,2)
    with pytest.raises(ValueError,match='Missing declared fit class'):
        TorchMLP(list(X),['cat'],classes=['b','a'],epochs=2).fit(X,['a']*50)


def test_timeout_does_not_create_publishable_state():
    X=frame();model=TorchMLP(list(X),['cat'],epochs=200,seconds=1e-9)
    with pytest.raises(TimeoutError):model.fit(X,X.x)
    assert not hasattr(model,'state')


def test_categorical_only_and_numeric_only():
    for X,cats in [(frame()[['x','missing']],[]),(frame()[['cat']],['cat'])]:
        model=TorchMLP(list(X),cats,epochs=2).fit(X,np.arange(50.))
        assert np.isfinite(model.predict(X)).all()


def test_label_names_permutation_equivariance_and_repeat():
    X=frame();labels=np.array(['a','b']*25)
    left=TorchMLP(list(X),['cat'],classes=['a','b'],epochs=3,seed=17).fit(X,labels)
    right=TorchMLP(list(X),['cat'],classes=['u','v'],epochs=3,seed=17).fit(X,np.where(labels=='a','u','v'))
    np.testing.assert_array_equal(left.predict(X),right.predict(X))
    repeat=TorchMLP(list(X),['cat'],classes=['a','b'],epochs=3,seed=17).fit(X,labels)
    np.testing.assert_array_equal(left.predict(X),repeat.predict(X))


def test_unsupported_custom_torch_still_rejected():
    from automl_agent.custom import validate_code
    with pytest.raises(ValueError,match='torch'):
        validate_code('import torch\ndef build_pipeline(context):\n    return None\n','pipeline')


def test_nested_time_and_group_disjointness():
    from automl_agent.contracts import TaskSpec
    from automl_agent.validation import inner_split
    X=pd.DataFrame({'x':range(80),'period':np.repeat(range(4),20),'entity':np.tile(range(20),4)})
    y=pd.DataFrame({'y':np.arange(80.)})
    base=dict(kind='regression',targets=['y'],feature_columns=['x'],id_columns=[],metric={'name':'rmse'},output={'columns':['y'],'prediction_type':'values'},source_hash='fixture')
    temporal=TaskSpec(**base,time_column='period')
    a,b=inner_split(X,y,temporal,'temporal',42);na,nb=inner_split(X.iloc[a],y.iloc[a],temporal,'temporal',42)
    assert X.iloc[a].period.max()<X.iloc[b].period.min()
    assert X.iloc[a].iloc[na].period.max()<X.iloc[a].iloc[nb].period.min()
    group=TaskSpec(**base,group_column='entity')
    a,b=inner_split(X,y,group,'group',42);na,nb=inner_split(X.iloc[a],y.iloc[a],group,'group',42)
    assert not set(X.iloc[a].entity)&set(X.iloc[b].entity)
    assert not set(X.iloc[a].iloc[na].entity)&set(X.iloc[a].iloc[nb].entity)


def test_failed_refit_clears_previous_state():
    X=frame();model=TorchMLP(list(X),['cat'],epochs=1).fit(X,X.x)
    assert hasattr(model,'state')
    model.seconds=1e-9
    with pytest.raises(TimeoutError):model.fit(X,X.x)
    assert not hasattr(model,'state')


def test_research_attention_tiny_forward_and_empty():
    X=frame();m=TorchMLP(list(X),['cat'],epochs=1,attention=True).fit(X,X.x)
    assert np.isfinite(m.predict(X)).all()
    assert m.predict(X[:0]).shape==(0,)


def test_target_scaling_is_fit_only_and_affine_consistent():
    X=frame();y=X.x.to_numpy()**2
    a=TorchMLP(list(X),['cat'],epochs=3,seed=42).fit(X,y)
    b=TorchMLP(list(X),['cat'],epochs=3,seed=42).fit(X,100*y+80)
    np.testing.assert_allclose(100*a.predict(X)+80,b.predict(X),atol=1e-3,rtol=1e-5)
    assert a.mean==pytest.approx(y.mean()) and a.scale==pytest.approx(y.std())
