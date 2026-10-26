"""Explicit AutoGluon configurations must not inherit sklearn-only semantics."""
import pytest
from automl_agent.contracts import MetricSpec,OutputSpec,PipelineSpec
from automl_agent.search import autogluon_portfolio
from automl_agent.compatibility import incompatibilities


def classified(task):
    return task.model_copy(update={'kind':'classification','classes':{'answer':[2,10]},
        'metric':MetricSpec(name='f1',average='macro',positive_label=2),
        'output':OutputSpec(columns=['answer'],prediction_type='labels')})


@pytest.mark.parametrize('family,parameter,value',[
    ('GBM','class_weight','balanced'),('GBM','class_weight',{'2':8,'10':1}),
    ('CAT','class_weights',{'2':8,'10':1})])
def test_unimplemented_native_weight_semantics_rejected_before_library_fit(tabular,family,parameter,value):
    task=classified(tabular[2])
    spec=PipelineSpec(engine='autogluon',model='library_search',learners=[family],
        autogluon_configurations={family:[{parameter:value}]})
    with pytest.raises(ValueError,match='direct/Optuna'):
        autogluon_portfolio(spec,task,[family])
    assert any('direct/Optuna' in reason for reason in incompatibilities(spec,task,'random'))


def test_native_catboost_auto_weights_and_default_portfolio_are_preserved(tabular):
    task=classified(tabular[2])
    spec=PipelineSpec(engine='autogluon',model='library_search',learners=['CAT'],
        autogluon_configurations={'CAT':[{'auto_class_weights':'Balanced','iterations':5}]})
    assert not incompatibilities(spec,task,'random')
    assert autogluon_portfolio(spec,task,['CAT'])['CAT'][0]['auto_class_weights']=='Balanced'
    assert set(autogluon_portfolio(PipelineSpec(engine='autogluon',model='library_search'),task,['GBM','CAT','XGB']))=={'GBM','CAT','XGB'}
