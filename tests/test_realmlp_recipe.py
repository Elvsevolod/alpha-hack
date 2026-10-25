import ast,math
from pathlib import Path
import joblib,numpy as np,pandas as pd,pytest,torch
from tools.realmlp_td_s import RealTransform,RealMLPTDS,coslog4,make_network


def test_schedule_matches_pinned_author_function():
    source=Path(__file__).resolve().parents[1]/'research/pytorch_author_recipe_20261026/pytabkit_sources/pytabkit/models/training/scheduling.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CoslogFunc')
    namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    author=namespace['CoslogFunc'](4)
    np.testing.assert_allclose([coslog4(t) for t in np.linspace(0,1,101)],[author(t) for t in np.linspace(0,1,101)],atol=1e-14)


def test_published_ntp_initialization_and_scaling():
    torch.manual_seed(42);model=make_network(7,3,True)
    assert len(model.layers)==4 and model.layers[0].weight.shape==(256,7)
    assert torch.equal(model.scale,torch.ones(7))
    assert torch.count_nonzero(model.layers[-1].weight)==0 and torch.count_nonzero(model.layers[-1].bias)==0
    x=torch.randn(6,7);layer=model.layers[0]
    torch.testing.assert_close(layer(x),x@layer.weight.T/math.sqrt(7)+layer.bias)
    torch.testing.assert_close(model(x),torch.zeros(6,3))


def test_robust_preprocessing_is_fit_only_with_range_fallback():
    X=pd.DataFrame({'x':[0.,0.,0.,1.,100.],'fixed':1.,'cat':['a','a','b',None,'c']})
    transform=RealTransform(list(X),['cat']).fit(X);before=transform.center.copy();out=transform.transform(X)
    assert np.isfinite(out).all() and (np.abs(out)<3).all()
    assert transform.scale[1]==0
    query=X[:2].copy();query['x']=1e6;query['cat']=['unseen',None]
    assert np.isfinite(transform.transform(query)).all()
    np.testing.assert_array_equal(before,transform.center)
    # Author zero vectors for unknown/missing; preserve separate raw status logic.
    np.testing.assert_array_equal(transform.raw(query)[0,2:],transform.raw(query)[1,2:])


@pytest.mark.parametrize('classification',[False,True])
def test_recipe_replay_serialization_and_epoch_schedule(tmp_path,classification):
    X=pd.DataFrame({'x':np.linspace(-2,2,80),'cat':['a','b']*40});classes=['a','b'] if classification else None
    y=np.where(X.x>0,'b','a') if classification else np.sin(X.x)
    m=RealMLPTDS(list(X),['cat'],classes,epochs=3).fit(X,y)
    path=tmp_path/'model.joblib';joblib.dump(m,path)
    np.testing.assert_array_equal(m.predict(X),joblib.load(path).predict(X))
    assert m.epochs_run==3 and np.isfinite(m.predict(X)).all()
    assert m.predict(X[:0]).shape==((0,2) if classification else (0,))
