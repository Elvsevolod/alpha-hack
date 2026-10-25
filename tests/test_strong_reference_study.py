from types import SimpleNamespace

import numpy as np
import pytest

from tools.strong_reference_study import numeric_importance,candidate_estimator
from automl_agent.contracts import PipelineSpec


def test_importance_mapping_keeps_numeric_names_after_one_hot_expansion():
    state=dict(columns=['a','category','b'],categorical=['category'],
        model=SimpleNamespace(feature_importances_=np.array([.1,.8,.9,.95])),
        preprocessor=SimpleNamespace(get_feature_names_out=lambda:np.array([
            'numeric__a','numeric__b','categories__category_X','categories__category_Y'])))
    assert numeric_importance(state)==['b','a']
    state['model'].feature_importances_=np.array([1.])
    with pytest.raises(ValueError,match='mapping'):numeric_importance(state)


@pytest.mark.parametrize('classification',[False,True])
def test_explicit_dart_recipe_uses_finite_fixed_rounds(classification):
    model=candidate_estimator(PipelineSpec(model='lightgbm',hyperparameters=dict(
        boosting_type='dart',drop_rate=.05,skip_drop=.5,n_estimators=5,
        num_leaves=3,min_child_samples=2)),classification)
    X=np.arange(80).reshape(40,2)
    y=(X[:,0]>40).astype(int) if classification else X[:,0]*.5
    model.fit(X,y)
    prediction=model.predict_proba(X) if classification else model.predict(X)
    assert np.isfinite(prediction).all()
    assert model.get_params()['boosting_type']=='dart'
    assert model.get_params()['n_estimators']==5


def test_regression_frozen_forecasts_do_not_reuse_search_stopping_predictions(tmp_path,monkeypatch):
    from tools import strong_reference_study as study
    monkeypatch.setattr(study,'ROOT',tmp_path)
    legacy=tmp_path/'legacy';(legacy/'regularized_lgb_cv0').mkdir(parents=True)
    fixed=tmp_path/'outputs/further_reference_20261025/regression/current_lgb_cv0'
    fixed.mkdir(parents=True)
    np.save(legacy/'regularized_lgb_cv0/prediction.npy',[1.,2.])
    np.save(fixed/'prediction.npy',[3.,4.])
    np.testing.assert_array_equal(study.baseline(legacy,['regularized_lgb'],'cv0',None),[1,2])
    np.testing.assert_array_equal(study.baseline(legacy,['regularized_lgb'],'cv0',None,fixed_rounds=True),[3,4])
    with pytest.raises(ValueError,match='explicit query rows'):
        study.baseline(legacy,['regularized_lgb'],'audit',None)


def test_failed_optional_job_is_not_retried_or_counted_as_complete(tmp_path,monkeypatch):
    import subprocess,time
    from tools.strong_reference_study import jobs
    job=tmp_path/'regression/failed_cv0';job.mkdir(parents=True)
    (job/'failure.json').write_text('{"reason":"resource_limit"}')
    def forbidden(*args,**kwargs):raise AssertionError('No automatic retry')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    assert jobs(tmp_path,'regression',['failed'],['0','1'],time.monotonic()+10)==[]
    assert (job/'failure.json').exists()
