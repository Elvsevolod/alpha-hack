import joblib
import numpy as np
import pandas as pd
import pytest

from automl_agent.pair_refinement import PairRefiner


def test_pair_selection_and_correction_preserve_mass_without_mutating_reference(tmp_path):
    X=pd.DataFrame({'x':np.r_[np.linspace(-2,-1,40),np.linspace(1,2,40),np.linspace(3,4,40)]})
    y=np.repeat([0,1,2],40)
    calibration_y=np.repeat([0,1,2],10)
    calibration=np.tile([.4,.5,.1],(30,1));calibration[20:]=[.1,.1,.8]
    model=PairRefiner(max_pairs=1,min_train_support=10,min_calibration_support=5).fit(X,y,calibration_y,calibration)
    assert [(p['a'],p['b']) for p in model.pairs_]==[(0,1)]
    base=np.tile([.4,.5,.1],(len(X),1));snapshot=base.copy()
    result=model.predict_proba(X,base)
    np.testing.assert_allclose(result.sum(axis=1),1)
    np.testing.assert_allclose(result[:,2],base[:,2])
    np.testing.assert_allclose(result[:,:2].sum(axis=1),base[:,:2].sum(axis=1))
    np.testing.assert_array_equal(base,snapshot)
    joblib.dump(model,tmp_path/'refiner.joblib')
    np.testing.assert_allclose(joblib.load(tmp_path/'refiner.joblib').predict_proba(X,base),result)


def test_no_confusion_means_no_extra_fits_or_prediction_change():
    X=np.arange(120).reshape(60,2);y=np.repeat([0,1],30);p=np.eye(2)[y]
    model=PairRefiner().fit(X,y,y,p)
    assert model.pairs_==[]
    np.testing.assert_array_equal(model.predict_proba(X,p),p)
    with pytest.raises(ValueError):model.predict_proba(X,p[:,:1])
