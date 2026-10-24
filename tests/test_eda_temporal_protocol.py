from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from automl_agent.eda.temporal import choose_eda_validation, evaluation_contract, period_errors, paired_comparison
from automl_agent.validation import inner_split
from automl_agent.contracts import TaskSpec, MetricSpec, OutputSpec, EvaluationDesign
from automl_agent.eda.reporting import write_reports


def fixture_data(periods=240, external=30):
    task = TaskSpec(kind='regression', targets=['outcome'], feature_columns=['clock','signal'],
        time_column='clock', metric=MetricSpec(name='rmse'), output=OutputSpec(columns=['outcome'], prediction_type='values'), source_hash='fixture')
    train = pd.DataFrame({'clock':np.repeat(np.arange(periods),3), 'signal':np.arange(periods*3), 'outcome':np.arange(periods*3)*.1})
    test = pd.DataFrame({'clock':np.repeat(np.arange(periods,periods+external),3), 'signal':0})
    return SimpleNamespace(task=task, train=train, test=test, source_hash='fixture')


def design(window=34, method='descriptive', block=None, effect=None):
    return EvaluationDesign(rationale='Match the observed future transfer and retain earlier history.',
        assumptions='One historical origin; regime changes may invalidate transfer.', temporal_window_periods=window,
        method=method, block_periods=block, minimum_effect_rel=effect)


def test_windows_are_chosen_and_ignore_heldout_labels():
    d=fixture_data();plan=choose_eda_validation(d,42)
    assert d.train.iloc[plan.validation].clock.nunique()==1  # production defaults unchanged
    assert evaluation_contract(d,plan)['temporal_window'] is None
    contract=evaluation_contract(d,plan,design(34))
    assert contract['temporal_window']==34 and contract['horizon_covered']
    X=d.train.iloc[plan.fit][d.task.feature_columns];y=d.train.iloc[plan.fit][d.task.targets]
    a,b=inner_split(X,y,d.task,plan.strategy,42,temporal_window=34,gap_periods=5)
    assert X.iloc[b].clock.nunique()==34
    assert X.iloc[b].clock.min()-X.iloc[a].clock.max()==6
    d.train.loc[plan.validation+plan.control,'outcome']=-99999
    assert choose_eda_validation(d,42).hash==plan.hash
    assert evaluation_contract(d,plan,design(34))==contract


def test_horizon_mismatch_is_explicit_and_impossible_windows_rejected():
    d=fixture_data(external=400);p=choose_eda_validation(d,1)
    c=evaluation_contract(d,p,design(100));assert not c['horizon_covered']
    with pytest.raises(ValueError,match='no earlier'):
        evaluation_contract(d,p,design(500))


def test_group_inner_window_preserves_group_and_time_boundaries():
    d = fixture_data(); d.task.group_column = 'entity'
    X = d.train[d.task.feature_columns].copy(); X['entity'] = X.clock//4
    a,b = inner_split(X,d.train[d.task.targets],d.task,'temporal_group',42,temporal_window=37)
    assert X.iloc[a].clock.max() < X.iloc[b].clock.min()
    assert not set(X.iloc[a].entity) & set(X.iloc[b].entity)


def make_report(error, n=60):
    time = pd.Series(np.repeat(np.arange(n),3))
    actual = np.zeros(len(time)); predicted = np.repeat(np.asarray(error if np.ndim(error) else [error]*n),3)
    return {'metric':'rmse','score':float(np.sqrt(np.mean(predicted**2))), 'fit_keys_hash':'fit',
        'evaluation_keys_hash':'eval','evaluation_protocol_hash':'protocol',
        'evaluation_contract':{'horizon_covered':True,'method':'moving_block','design':design(n,'moving_block',5,.01).model_dump(mode='json')}, 'period_errors':period_errors(time,actual,predicted)}


@pytest.mark.parametrize('candidate,verdict',[(1.,'supported_on_this_origin'),(1.999,'below_selected_effect_floor'),(3.,'worse_on_this_origin')])
def test_paired_interval_rejects_tiny_gains_and_reports_harm(candidate,verdict):
    baseline = make_report(2.)
    result = paired_comparison(baseline,make_report(candidate))
    assert result['verdict'] == verdict
    assert result['method'] == 'moving_block_percentile'
    assert result == paired_comparison(baseline,make_report(candidate))
    assert result['minimum_absolute_gain'] == .02


def test_comparison_cannot_match_different_rows_protocols_or_insufficient_periods():
    a,b = make_report(2.),make_report(1.)
    for key in ['evaluation_keys_hash','evaluation_protocol_hash','fit_keys_hash','metric']:
        changed=deepcopy(b); changed[key]='other'
        assert paired_comparison(a,changed)['status']=='incompatible'
    small_a,small_b=make_report(2.,6),make_report(1.,6)
    assert paired_comparison(small_a,small_b)['status']=='limited'
    b['evaluation_contract'].update(method='descriptive')
    assert paired_comparison(a,b)['verdict']=='inconclusive'
    b['evaluation_contract'].update(method='moving_block')
    b['evaluation_contract']['design']['minimum_effect_rel']=None
    assert paired_comparison(a,b)['verdict']=='direction_only_on_this_origin'


def test_report_keeps_partial_status_and_does_not_invent_conclusion(tmp_path):
    summary={'state':'partial','stop_reason':'known failure','model_generations':7,'model_fits':1,'finish':None}
    write_reports(tmp_path,summary,[],{'version':2})
    assert 'No agent conclusion was completed' in (tmp_path/'REPORT.md').read_text()
    assert 'unassessed' in (tmp_path/'evaluation_report.json').read_text()


def test_real_temporal_fit_and_candidate_share_window_and_generate_uncertainty():
    from automl_agent.eda.error_analysis import run
    d=fixture_data(); X=d.train[d.task.feature_columns]; y=d.train[d.task.targets]
    evaluation={'temporal_window':30,'horizon_covered':True,'method':'moving_block','design':design(30,'moving_block',5,.01).model_dump(mode='json'),'version':3}
    request={'task':d.task.model_dump(mode='json'),'seed':42,'context':{'strategy':'temporal',
        'protocol_hash':'fixture','cpu':1,'memory_mb':4096,'task':d.task.model_dump(mode='json'),'evaluation_contract':evaluation}}
    baseline=run(request,X,y)['report']
    candidate=run({**request,'baseline_temporal_report':baseline,'pipeline':{
        **baseline['pipeline'],'target_transform':'log1p'}},X,y)['report']
    assert baseline['evaluation_periods']==candidate['evaluation_periods']==30
    assert baseline['evaluation_rows']==candidate['evaluation_rows']==90
    assert baseline['evaluation_keys_hash']==candidate['evaluation_keys_hash']
    assert candidate['temporal_comparison']['status']=='available'
    assert len(candidate['temporal_comparison']['delta_interval'])==2
    assert candidate['outer_validation_used'] is candidate['control_used'] is False


def test_old_evaluation_cannot_silently_resume_with_new_protocol(tmp_path):
    from automl_agent.eda.agent import EDAAgent
    from conftest import write_task
    path=write_task(tmp_path/'task')
    agent=EDAAgent(path,tmp_path/'result',tmp_path/'cost.json')
    agent.journal.put('eda_protocol_version',1);agent.close()
    with pytest.raises(ValueError,match='frozen runtime'):
        EDAAgent(path,tmp_path/'result',tmp_path/'cost.json')


def test_adaptive_design_freezes_before_fit_and_never_spends_on_invalid_revision(tmp_path):
    from automl_agent.eda.agent import EDAAgent
    from conftest import write_task
    from automl_agent.validation import choose_validation
    task=write_task(tmp_path/'task',temporal=True)
    agent=EDAAgent(task,tmp_path/'result',tmp_path/'cost.json')
    try:
        agent.task=agent._task();agent.plan=choose_validation(agent.dataset)
        agent.select_evaluation(design(1))
        before=agent.journal.get('eda_evaluation_contract')
        agent.budget.counter('eda_fits',1)
        agent.select_evaluation(design(1))
        changed=design(1,'moving_block',1)
        with pytest.raises(ValueError,match='frozen'):
            agent.select_evaluation(changed)
        assert agent.journal.get('eda_evaluation_contract')==before
        assert agent.budget.counter('eda_fits')==1
    finally:agent.close()


def test_temporal_fit_without_chosen_window_is_incompatible_and_free(tmp_path):
    from automl_agent.eda.agent import EDAAgent
    from conftest import write_task
    task=write_task(tmp_path/'task',temporal=True)
    agent=EDAAgent(task,tmp_path/'result',tmp_path/'cost.json')
    try:
        agent.task=agent._task();agent.plan=choose_eda_validation(agent.dataset,42)
        result=agent.inner_experiment()
        assert result['status']=='incompatible' and agent.budget.counter('eda_fits')==0
    finally:agent.close()


def test_full_cycle_can_enlarge_temporal_window_without_opening_control(tmp_path):
    from automl_agent.controller import Controller
    from automl_agent.budget import Limits
    from automl_agent.contracts import ReviseValidationArgs
    from conftest import write_task
    c=Controller(write_task(tmp_path/'task',temporal=True),Limits(seconds=300,cpu=1),run_id='adaptive_revision')
    try:
        c.prepare()
        old=c.plan;control=list(old.control)
        result=c.revise_validation(ReviseValidationArgs(proposed_strategy='temporal',temporal_window_periods=2,
            rationale='Feature-only horizon support requires more than the inherited one-period window.'))
        assert result['status']=='revised',result
        assert c.plan.control==control and c.plan.hash!=old.hash
        assert c.dataset.train.iloc[c.plan.validation][c.task.time_column].nunique()==2
        assert c.plan.observations[-1]['source']=='agent_methodological_choice'
    finally:c.close()


def test_default_candidate_order_does_not_impose_universal_practical_effect():
    from automl_agent.validation import selection_for
    policy=selection_for(fixture_data().task)
    assert policy.min_delta_abs==policy.min_delta_rel==0
    assert policy.improvement(1.999999,2.)
    # Ordering is still distinct from inferential/practical support.
    a,b=make_report(2.),make_report(1.999999)
    b['evaluation_contract']['design']['minimum_effect_rel']=None
    result=paired_comparison(a,b)
    assert result['verdict']=='direction_only_on_this_origin'


def test_error_cache_binds_active_design_and_reuses_the_frozen_result(tmp_path,monkeypatch):
    from automl_agent.eda.agent import EDAAgent
    from automl_agent.contracts import DatasetReportArgs
    from conftest import write_task
    agent=EDAAgent(write_task(tmp_path/'task',temporal=True),tmp_path/'result',tmp_path/'cost.json')
    try:
        agent.task=agent._task();agent.plan=choose_eda_validation(agent.dataset,42)
        agent.context_memory.recover()
        args=DatasetReportArgs(report='errors',include_target=True)
        first,old_identity=agent.eda.report(args,agent.plan)
        assert first['status']=='incompatible'
        agent.select_evaluation(design(1))
        calls=[]
        agent.eda.error_runner=lambda:calls.append(True) or {'status':'available','score':1.}
        second,new_identity=agent.eda.report(args,agent.plan)
        assert second['status']=='available' and new_identity!=old_identity
        assert agent.eda.report(args,agent.plan)[0]==second and len(calls)==1
    finally:agent.close()
