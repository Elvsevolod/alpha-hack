from types import SimpleNamespace
import json

import numpy as np
import pytest
from pydantic import BaseModel,Field

from automl_agent.contracts import PipelineSpec,public_tool_schema
from automl_agent.pipelines import effective_parameters,make_estimator
from automl_agent.research_evaluation import stability_design


@pytest.mark.parametrize('alias,canonical',[('lambda','reg_lambda'),('alpha','reg_alpha'),('eta','learning_rate'),('min_split_loss','gamma')])
def test_xgboost_official_aliases_normalize_without_mutating_recipe(alias,canonical):
    original=PipelineSpec(model='xgboost',hyperparameters={alias:.2})
    assert effective_parameters(original)=={canonical:.2}
    assert original.hyperparameters=={alias:.2}
    with pytest.raises(ValueError,match='Conflicting XGBoost'):
        effective_parameters(PipelineSpec(model='xgboost',hyperparameters={alias:.2,canonical:.3}))


@pytest.mark.parametrize('classification',[False,True])
def test_actual_xgboost_alias_rejection_is_fixed_with_identical_predictions(classification):
    X=np.random.default_rng(42).normal(size=(80,4));y=(X[:,0]>0).astype(int) if classification else X[:,0]
    common=dict(n_estimators=8,max_depth=2)
    alias=make_estimator(PipelineSpec(model='xgboost',hyperparameters={**common,'lambda':1}),classification,{'cpu':1})
    canonical=make_estimator(PipelineSpec(model='xgboost',hyperparameters={**common,'reg_lambda':1}),classification,{'cpu':1})
    alias.fit(X,y);canonical.fit(X,y)
    np.testing.assert_array_equal(alias.predict(X),canonical.predict(X))


def test_schema_compaction_keeps_real_title_fields_defaults_and_validation():
    class Probe(BaseModel):
        title:str=Field(min_length=3,description='A real user field')
        count:int=Field(ge=1)
        settings:dict=Field(default={'title':'data, not an annotation'})
    schema=public_tool_schema(Probe)
    from context_compressor.transport import _omit_schema_titles
    _omit_schema_titles(schema)
    assert 'title' not in schema
    assert schema['properties']['title']['minLength']==3
    assert schema['properties']['title']['description']=='A real user field'
    assert schema['properties']['count']['minimum']==1
    assert schema['properties']['settings']['default']=={'title':'data, not an annotation'}
    assert set(schema['required'])=={'title','count'}


@pytest.mark.parametrize('strategy,temporal',[('random',False),('group',False),('temporal',True),('temporal_group',True)])
def test_stress_semantics_distinguish_row_and_feature_changes(strategy,temporal):
    design=stability_design(strategy)
    assert design['varied_axis']=='rows_not_features'
    assert design['temporal_order_required']==temporal
    assert design['split_strategy']==strategy


def test_stress_semantics_survive_small_experiment_summary():
    from automl_agent.context_memory import experiment_summary
    c=SimpleNamespace(task=SimpleNamespace(kind='regression',targets=['y'],metric=SimpleNamespace(maximize=False)),
        plan=SimpleNamespace(strategy='random'))
    spec=SimpleNamespace(pipeline=PipelineSpec(model='ridge'),experiment_id='e')
    source=SimpleNamespace(version='v',reference={'id':'source'})
    result={'training_summary':{'temporal_stability':{'status':'complete','scope':'fit_only_stress_windows',
        'scores':{'exact_features_stress_0':1.},'windows_hash':'h'}}}
    change=dict(basis='standalone',declared_component='ridge',effective_fields=[],claimed_field_unchanged=None,mixed_change=False)
    summary=experiment_summary(c,spec,result,source,change)
    assert summary['temporal_stability']['evaluation_design']['temporal_order_required'] is False
    assert len(json.dumps(summary).encode())<=1536


def test_finish_does_not_require_exact_catalog_ids_or_extra_model_fits(monkeypatch):
    from automl_agent import directions
    alternatives=[{'id':'action:linear','model':'logistic'}]
    monkeypatch.setattr(directions,'direction_map',lambda c:dict(affordable_unattempted=alternatives,
        directions=[{'experiment_id':'measured'}],diagnostic_questions=[]))
    c=SimpleNamespace(budget=SimpleNamespace(final_reserve=lambda costs:10),finalist_costs=lambda:[])
    args=SimpleNamespace(skipped_directions={'further_tuning':'The remaining gain is unknown; retain the measured result.'},
        reason='Retain the measured incumbent; generalization of untried approaches remains unknown.')
    result=directions.finish_readiness(c,args)
    assert result['ready'] and result['available_actions']==alternatives
    assert not result['optimality_established'] and not result['quota_required']
    args.reason=' '
    with pytest.raises(ValueError,match='concrete reason'):directions.finish_readiness(c,args)


def test_dispatch_finish_keeps_evidence_and_control_boundaries_without_catalog_quota(tmp_path,monkeypatch):
    from conftest import write_task
    from automl_agent.controller import Controller
    from automl_agent.budget import Limits
    from automl_agent.contracts import ToolCommand
    from automl_agent.tool_results import payload
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    c=Controller(write_task(tmp_path),Limits(seconds=180,cpu=1),run_id='finish_advisory')
    c.no_llm=True;c.qwen.disabled=True
    try:
        c.prepare()
        rejected=payload(c.dispatch(ToolCommand(operation_id='empty_finish',tool='finish',arguments={'reason':'Retain a result'})))
        assert rejected.get('status')=='rejected'
        assert c.state=='searching'
        baseline=c.journal.get('baseline_id')
        result=payload(c.dispatch(ToolCommand(operation_id='valid_finish',tool='finish',arguments={
            'reason':'Retain the measured incumbent; other model families remain untested and their benefit is unknown.',
            'result_ids':[baseline],'skipped_directions':{'further_search':'No useful gain is established by the current evidence.'}})))
        assert result['status']=='frozen' and c.state=='frozen'
        assert c.journal.get('control_status','not_opened')=='not_opened'
    finally:c.close()
