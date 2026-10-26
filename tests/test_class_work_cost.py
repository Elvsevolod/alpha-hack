"""Class work changes model cost, without scaling interpreter/staging twice."""
import copy
import pytest
from automl_agent.cost_model import class_work_factors,phase_forecast
from automl_agent.contracts import PipelineSpec


def measured(observed=2,declared=10):
    return {'rows':100,'seconds':10,'budget':{'selected_fit_rows':100,'targets':{
        'y':{'selected_fit_actual_seconds':10}},'class_work':{'version':1,'engine':'autogluon',
        'scope':'fit_observed_to_declared_class_space','targets':{'y':{
            'observed_classes':observed,'declared_classes':declared}}}},'execution_phases':{
        'fit_seconds':10,'save_load_seconds':1,'predict_and_verify_seconds':2,'prediction_seconds':2,
        'prediction_rows':100,'load_and_construct_seconds':3}}


def test_changed_declared_space_changes_cost_with_identical_rows_and_rounds():
    a=measured(2,2);b=measured(2,10)
    assert phase_forecast(a,20,100,100)==pytest.approx(25.2)
    assert phase_forecast(b,20,100,100)==pytest.approx((100+10+20+3+4)*1.25+.2)
    assert b['budget']['targets']['y']['selected_fit_actual_seconds']==10


def test_fixed_startup_staging_and_initialization_do_not_inherit_class_multiplier():
    a=measured();b=copy.deepcopy(a)
    b['execution_phases']['load_and_construct_seconds']+=3
    b['budget']['worker_initialization_seconds']=2
    assert phase_forecast(b,23,100,100)-phase_forecast(a,20,100,100)==pytest.approx(5*1.25)


def test_multi_target_fit_is_time_weighted_but_model_output_bound_is_maximum():
    s=measured();s['budget']['class_work']['targets']['z']={'observed_classes':10,'declared_classes':10}
    s['budget']['targets']['z']={'selected_fit_actual_seconds':30}
    f=class_work_factors(s)
    assert f['fit']==pytest.approx((10*10+1*30)/40) and f['model']==10
    del s['budget']['targets']['z']
    assert class_work_factors(s)['fit']==10


def test_legacy_or_non_autogluon_metadata_keeps_neutral_behavior():
    s=measured();del s['budget']['class_work']
    assert class_work_factors(s)['fit']==1
    assert class_work_factors(measured(),PipelineSpec(engine='flaml',model='library_search'))['model']==1
    assert class_work_factors(measured(3,3))['fit']==1


@pytest.mark.parametrize('counts',[{'observed_classes':0,'declared_classes':3},
    {'observed_classes':3,'declared_classes':2},{'observed_classes':True,'declared_classes':3}])
def test_invalid_counts_are_not_silently_used_as_a_cost_discount(counts):
    s=measured();s['budget']['class_work']['targets']['y']=counts
    with pytest.raises(ValueError,match='Invalid class counts'):class_work_factors(s)


def test_incomplete_phase_record_still_counts_known_class_work():
    s=measured();del s['execution_phases']
    assert phase_forecast(s,12,100,100)==pytest.approx((100+2)*1.25+.2)


def test_all_constant_automl_uses_actual_rows_without_native_class_work(tabular,tmp_path):
    from automl_agent.search import SearchPipeline
    from automl_agent.contracts import MetricSpec,OutputSpec
    X,y,task=tabular
    y['answer']='A'
    task=task.model_copy(update={'kind':'classification','classes':{'answer':['A','B','rare']},
        'metric':MetricSpec(name='f1',average='macro'),'output':OutputSpec(columns=['answer'],prediction_type='labels')})
    model=SearchPipeline(PipelineSpec(engine='autogluon',model='library_search'),task)
    summary=model.fit(X,y,{'strategy':'random','cpu':1,'memory_mb':512,'budget_seconds':30,'workdir':str(tmp_path)})
    assert summary.budget['class_work']=={}
    assert summary.budget['selected_fit_rows']==len(X)
    assert model.training_keys==list(X.index)
    assert class_work_factors(summary.model_dump())['fit']==1


def test_class_work_reaches_completion_graph_without_target_access(tmp_path):
    from automl_agent.completion import estimate
    from test_completion_recipe_identity import cache_fixture
    c,_,rows,_,data,_=cache_fixture(tmp_path)
    pipeline=PipelineSpec(engine='autogluon',model='library_search',learners=['GBM'])
    spec,result=c.lookup('candidate');spec.pipeline=pipeline
    data['finalization_plan']={'policies':{'candidate':{'frozen_pipeline':pipeline.model_dump(mode='json')}}}
    result.training_summary=measured(2,2);result.seconds=20
    original=estimate(c,['candidate'],rows)
    result.training_summary=measured(2,10)
    expanded=estimate(c,['candidate'],rows)
    assert expanded['nodes'][0]['class_work_factors']['fit']==10
    assert expanded['seconds']>original['seconds']
    # The fake dataset has only row keys; estimate never reads target labels.
    assert set(vars(c.dataset))=={'row_keys'}


def test_measured_fixed_fit_intercept_is_not_class_scaled():
    a=measured();a['cost_curve']={'fixed_fit_seconds':3,'seconds_per_row':.1,'uncertainty_multiplier':1.5}
    b=copy.deepcopy(a);b['cost_curve']['fixed_fit_seconds']+=4
    assert phase_forecast(b,20,100,100)-phase_forecast(a,20,100,100)==pytest.approx(4*1.5)
