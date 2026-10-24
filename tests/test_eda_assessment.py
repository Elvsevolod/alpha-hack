"""Assess protocol support without closed labels; trace and connection diagnostics."""
import json
from types import SimpleNamespace

import httpx
import pandas as pd

from automl_agent.eda.assessment import protocol_assessment,target_overview
from automl_agent.eda.trace import sanitized
from tools.check_deepseek_connection import check


def test_temporal_support_warning_does_not_change_protocol_or_read_labels():
    dataset=SimpleNamespace(task=SimpleNamespace(time_column='day',group_column=None),
        train=pd.DataFrame({'day':[1,1,2,2,3,3,4,4],'target':[0]*8}),
        test=pd.DataFrame({'day':[5,6,7,8]}))
    plan=SimpleNamespace(hash='fixed',strategy='temporal',fit=[0,1,2,3],validation=[4,5],control=[6,7])
    before=protocol_assessment(dataset,plan)
    dataset.train.loc[plan.validation+plan.control,'target']=999
    assert protocol_assessment(dataset,plan)==before
    assert before['chronological'] and before['test_follows_train']
    assert {w['code'] for w in before['warnings']}=={'low_validation_support','temporal_horizon_mismatch'}
    assert plan.validation==[4,5] and plan.hash=='fixed'


def test_target_overview_does_not_depend_on_historical_risk_flag():
    profile={'targets':{'y':{'semantic_type':'categorical','counts':[{'rows':1200},{'rows':160},{'rows':0}]}},
        'observations':[{'flags':{'class_imbalance':False}}]}
    result=target_overview(profile)['y']
    assert result['observed_count_ratio']==7.5 and result['unobserved_declared_classes']==1
    assert profile['observations'][0]['flags']['class_imbalance'] is False


def test_read_only_auth_check_has_no_generations_or_secret_output(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.setenv('DEEPSEEK_API_KEY','private-test-credential')
    def response(request):
        assert request.method=='GET'
        return httpx.Response(401,json={'error':{'message':'private-test-credential'}})
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        result=check(tmp_path/'check.json',client)
    assert result['diagnosis']=='authentication_failed' and result['generation_requests']==0
    assert 'private-test-credential' not in (tmp_path/'check.json').read_text()


def test_valid_read_only_checks_distinguish_models_and_balance(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.setenv('DEEPSEEK_API_KEY','offline-fixture')
    def response(request):
        assert request.method=='GET'
        return httpx.Response(200,json={'data':[{'id':'deepseek-flash'}]} if request.url.path=='/models' else {'is_available':True})
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        result=check(tmp_path/'check.json',client)
    assert result['diagnosis']=='read_only_checks_ok'
    assert result['requested_model_available'] and result['balance_available']


def test_trace_redaction_preserves_json_types_even_for_short_secret():
    result=sanitized({'secret':'echo e','enabled':True,'count':3},'e')
    assert result['[REDACTED]nabl[REDACTED]d'] is True
    assert json.loads(json.dumps(result))==result


def test_model_boundary_context_does_not_confuse_closed_outer_size_with_scored_inner_support():
    from automl_agent.eda.assessment import research_boundary_context
    raw={'protocol_hash':'p','strategy':'temporal','rows':{'fit':10000,'validation':24,'control':24},
        'periods':{'fit':584,'validation':1,'control':1,'external_test':147},
        'warnings':[{'code':'low_validation_support'}],'chronological':True}
    public=research_boundary_context(raw)
    assert 'rows' not in public and 'periods' not in public and 'warnings' not in public
    assert public['external_test_periods']==147
    assert public['scored_scope']=='fit_inner_holdout_only'
    assert public['outer_validation_and_control']=='closed_and_not_scored'
