import joblib
import numpy as np
import pandas as pd
import pytest

from automl_agent.residual_calibration import ResidualCalibration


def test_known_shrinkage_unknown_groups_and_query_labels(tmp_path):
    fit=pd.DataFrame({'category':['A','A','B','B']})
    model=ResidualCalibration(['category'],shrinkage=2,strength=.5).fit(fit,[2,4,10,14],np.zeros(4))
    query=pd.DataFrame({'category':['A','B','new'],'target':[-1,999,0]})
    result=model.predict(query,np.full(3,100.))
    np.testing.assert_allclose(result,[102.625,104.875,103.75])
    np.testing.assert_allclose(model.predict(query.assign(target=1000),np.full(3,100.)),result)
    joblib.dump(model,tmp_path/'calibrator.joblib')
    np.testing.assert_allclose(joblib.load(tmp_path/'calibrator.joblib').predict(query,np.full(3,100.)),result)


def test_batch_invariance_and_no_mutation():
    X=pd.DataFrame({'c':['A','A','B','B']});p=np.array([1.,2.,3.,4.]);original=p.copy()
    model=ResidualCalibration(['c']).fit(X,p+1,p)
    point=pd.DataFrame({'c':['A']})
    batch=pd.concat([point,pd.DataFrame({'c':['A']*100})],ignore_index=True)
    np.testing.assert_allclose(model.predict(point,[5.]),model.predict(batch,np.full(101,5.))[:1])
    np.testing.assert_array_equal(p,original)


@pytest.mark.parametrize('actual,prediction', [([1.],[np.nan]),([1.,2.],[1.]),([],[])])
def test_invalid_residual_support_rejected(actual,prediction):
    with pytest.raises(ValueError):
        ResidualCalibration(['c']).fit(pd.DataFrame({'c':['A']*len(actual)}),actual,prediction)
