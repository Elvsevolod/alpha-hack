"""Reports follow declared roles on arbitrary schemas, not benchmark identities."""
import json
import time

import pandas as pd
import pytest

from automl_agent.contracts import DatasetReportArgs, TrainArgs
from automl_agent.eda.agent import EDAAgent
from automl_agent.profiling import descriptive_report
from automl_agent.validation import choose_validation
from conftest import write_task


@pytest.fixture(params=[False,True],ids=['original_schema','renamed_schema'])
def session(tmp_path,request):
    task=write_task(tmp_path/'data')
    fields={'row':'sample_id','value':'reading,[unit]','empty':'unrecorded','code':'label code','answer':'outcome_score'} if request.param else {}
    paths={'train.csv':'observed_signals.csv','test.csv':'future_signals.csv','sample_submission.csv':'answer_format.csv'} if request.param else {}
    if fields:
        for name,new in paths.items():
            path=task.parent/name
            pd.read_csv(path,keep_default_na=False,dtype=str).rename(columns=fields).to_csv(task.parent/new,index=False)
            path.unlink()
        spec=json.loads((task.parent/'task_spec.json').read_text())
        for key in ['targets','id_columns','feature_columns','categorical_columns']:
            spec[key]=[fields.get(c,c) for c in spec[key]]
        spec['output']['columns']=[fields.get(c,c) for c in spec['output']['columns']]
        spec['file_roles']=paths
        (task.parent/'task_spec.json').write_text(json.dumps(spec))
    agent=EDAAgent(task,tmp_path/'report',tmp_path/'unused_ledger.json')
    agent.task=agent.dataset.activate(agent.dataset.explicit())
    agent.plan=choose_validation(agent.dataset)
    agent.journal.transition('searching','offline declared-role fixture')
    yield agent,{name:fields.get(name,name) for name in ['row','value','empty','code','answer']}
    agent.close()


def test_missing_and_category_facts_use_only_declared_rows_and_names(session):
    agent,names=session
    missing,_=agent.eda.report(DatasetReportArgs(report='missing',columns=[names['value'],names['empty']]),agent.plan)
    fit=agent.dataset.train.iloc[agent.plan.fit]
    assert missing['status']=='available'
    assert missing['missing_counts'][names['empty']]==len(fit)
    assert missing['joint_missing'][names['value']][names['empty']]==int(fit[names['value']].isna().sum())
    categories,_=agent.eda.report(DatasetReportArgs(report='categories',columns=[names['code']]),agent.plan)
    assert categories['columns'][names['code']]['top']=={str(k):int(v) for k,v in fit[names['code']].value_counts().items()}
    assert 'mean' not in categories['columns'][names['code']]
    staged=list((agent.workdir/'analyses').glob('*/analysis_X.json'))
    assert len(staged)==2
    for path in staged:
        assert names['answer'] not in json.loads(path.read_text())['columns']
        assert (path.parent/'request.json').exists()
        assert json.loads((path.parent/'request.json').read_text())['mode']=='descriptive_report'


def test_grouped_profile_does_not_build_global_profile_and_bounds_output(session,monkeypatch):
    agent,names=session
    monkeypatch.setattr(agent.eda,'profile',lambda *a,**k:pytest.fail('Unrequested global profile'))
    report,_=agent.eda.report(DatasetReportArgs(report='profile',columns=[names['value']],group_by=names['value']),agent.plan)
    assert report['status']=='available'
    assert report['groups_truncated'] and len(report['groups'])==50
    assert 'targets' not in report


def test_report_worker_timeout_cached_and_expired_budget_never_starts_work(session,monkeypatch):
    agent,names=session;calls=[]
    def timeout(*args):
        calls.append(args[-1]);return {'status':'timeout'}
    monkeypatch.setattr(agent.executor,'run',timeout)
    args=DatasetReportArgs(report='missing',columns=[names['value']])
    first,_=agent.eda.report(args,agent.plan)
    assert first['status']=='technical_failure' and first['diagnostic_status']=='timeout'
    assert agent.eda.report(args,agent.plan)[0]==first and len(calls)==1
    assert 0<calls[0]<=20
    agent.budget.deadline=time.time()-1
    assert agent.eda.report(DatasetReportArgs(report='categories',columns=[names['code']]),agent.plan)[0]['status']=='budget_too_small'
    assert len(calls)==1


def test_closed_scope_and_false_evidence_cannot_trigger_work(session,monkeypatch):
    agent,names=session
    monkeypatch.setattr(agent.executor,'run',lambda *a:pytest.fail('Closed target reached worker'))
    with pytest.raises(ValueError,match='restricted to fit'):
        agent.eda.report(DatasetReportArgs(report='missing',scope='test_features',columns=[names['answer']],include_target=True),agent.plan)
    monkeypatch.setattr(agent,'inner_experiment',lambda *a:pytest.fail('False evidence triggered a fit'))
    args=TrainArgs(pipeline={'model':'ridge'},hypothesis={'observation':'claim','expected_effect':'effect',
        'changed_component':'model','acceptance':'score','evidence_refs':[{'kind':'observation','id':'invented','version':'invented'}]})
    with pytest.raises(ValueError,match='unavailable run record'):
        agent.train(args,None)
    assert agent.budget.counter('eda_fits')==0


def test_missing_support_does_not_overflow_on_large_counts():
    frame=pd.DataFrame({'sensor':[None]*70000,'other':[None]*70000})
    result=descriptive_report(frame,list(frame),'missing')
    assert result['joint_missing']['sensor']['other']==70000


def test_invalid_estimator_parameter_is_rejected_before_reserving_fit(session):
    from automl_agent.contracts import PipelineSpec
    agent,_=session
    before=agent.budget.counter('eda_fits')
    with pytest.raises(ValueError,match='Unsupported parameters.*n_estimators'):
        agent.validate_pipeline(PipelineSpec(model='hist_gradient_boosting',hyperparameters={'n_estimators':300}))
    assert agent.budget.counter('eda_fits')==before
    assert 'max_iter' in __import__('automl_agent.eda.policy',fromlist=['parameter_capabilities']).parameter_capabilities()['hist_gradient_boosting']
