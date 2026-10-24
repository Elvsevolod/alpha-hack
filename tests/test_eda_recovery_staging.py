"""Recovery owns only its worker; staging retains old artifacts when interrupted."""
import json
import subprocess
import sys
import time

import pandas as pd
import psutil
import pytest

from automl_agent.eda.agent import EDAAgent
from automl_agent.eda.error_analysis import classification_errors
from automl_agent.executor import stage_frame
from conftest import write_task


def test_staging_roundtrip_and_no_full_frame_object_conversion(tmp_path, monkeypatch):
    frame = pd.DataFrame({'x': [1., float('nan'), 3.], 's': ['a', None, 'б']}, index=['01','02','03'])
    original = pd.DataFrame.astype
    def bounded(self, *args, **kwargs):
        assert len(self) <= 2
        return original(self, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, 'astype', bounded)
    target = tmp_path/'rows.json'
    stage_frame(target, frame, deadline=time.time()+5, memory_mb=4096, chunk_rows=2)
    assert json.loads(target.read_text()) == {'columns':['x','s'], 'numeric':['x'],
        'index':['01','02','03'], 'data':[[1.,'a'],[None,None],[3.,'б']]}


def test_expired_or_excess_memory_staging_preserves_old_file(tmp_path):
    target = tmp_path/'rows.json'; target.write_text('previous result')
    for limits, error in [({'deadline':time.time()-1}, TimeoutError), ({'memory_mb':0}, ValueError)]:
        with pytest.raises(error):
            stage_frame(target, pd.DataFrame({'x':[1]}), **limits)
        assert target.read_text() == 'previous result'
        assert not list(tmp_path.glob('.pending-*'))


def test_midstream_expiry_preserves_old_file_and_removes_temporary(tmp_path, monkeypatch):
    target = tmp_path/'rows.json'; target.write_text('previous result')
    ticks = iter([0,0,0,0,2])
    monkeypatch.setattr('automl_agent.executor.time.time',lambda:next(ticks,2))
    with pytest.raises(TimeoutError):
        stage_frame(target,pd.DataFrame({'x':range(10)}),deadline=1,chunk_rows=2)
    assert target.read_text()=='previous result'
    assert not list(tmp_path.glob('.pending-*'))


def test_recovery_stops_owned_process_before_loading(tmp_path, monkeypatch):
    task = write_task(tmp_path/'task')
    agent = EDAAgent(task,tmp_path/'result',tmp_path/'unused_ledger.json')
    attempt = agent.workdir/'analyses'/'interrupted'; attempt.mkdir(parents=True)
    process = subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)',str(attempt/'request.json')],start_new_session=True)
    agent.journal.put('active_worker',{'pid':process.pid,'create_time':psutil.Process(process.pid).create_time(),'attempt':str(attempt)})
    def load():
        assert process.poll() is not None
        assert agent.journal.get('active_worker') is None
        raise ValueError('intentional stop before loading')
    monkeypatch.setattr(agent,'_task',load)
    try:
        result=agent.autonomous()
        assert 'intentional stop' in result['stop_reason']
        assert result['api_calls']==0 and result['model_fits']==0
    finally:
        if process.poll() is None:process.kill()
        process.wait();agent.close()


def test_missed_class_has_supported_recall_and_no_fabricated_precision():
    rows=classification_errors(['a']*60+['b']*40,['a']*100,['a','b'])
    missed=rows[1]
    assert missed['status']=='partial' and missed['recall']==0 and missed['no_predictions']
    assert missed['actual_rows']==40 and 'precision' not in missed and 'f1' not in missed
    assert missed['metric_status']['precision']=='insufficient_predicted_support'
    rare=classification_errors(['a']*99+['b'],['b']*100,['a','b'])[1]
    assert rare['precision']==.01 and 'recall' not in rare


def test_resume_keeps_saved_split_and_confirmed_completion_intent(tmp_path, monkeypatch):
    from automl_agent.validation import choose_validation
    task=write_task(tmp_path/'task')
    agent=EDAAgent(task,tmp_path/'result',tmp_path/'unused_ledger.json')
    try:
        agent.task=agent.dataset.activate(agent.dataset.explicit())
        agent.plan=choose_validation(agent.dataset,seed=123)
        expected=agent.plan.hash
        agent.journal.put('validation',agent.plan.model_dump(mode='json'))
        agent.journal.transition('searching','fixture')
        agent.journal.put('eda_finish',{'reason':'already finished','result_ids':[]})
        agent.journal.transition('frozen','fixture finished')
        operation,_=agent.journal.begin_operation(agent.run_id,'confirmed','get_dataset_report',{'report':'profile'})
        operation.status='succeeded';operation.outcome='confirmed';operation.ended_at=time.time()
        operation.result={'status':'available','report':{'rows':100},'analysis_status':'available'}
        agent.journal.put('completion_intent:confirmed',operation.model_dump(mode='json'))
        monkeypatch.setattr(agent,'_task',lambda:agent.task)
        monkeypatch.setattr(agent.qwen,'next_command',lambda *a:pytest.fail('Finished run requested new generation'))
        result=agent.autonomous()
        assert agent.plan.hash==expected
        assert result['state']=='completed' and result['api_calls']==0
        recovered=agent.journal.operation('confirmed')
        assert recovered.status=='succeeded' and recovered.outcome=='confirmed'
        assert recovered.result['data']['report']=={'rows':100}
        assert agent.history[0]['recovered_from_journal']
        assert agent.history[0]['result']==recovered.result
        assert agent.history[0]['command']['arguments']=={}
        assert agent.journal.get('completion_intent:confirmed') is None
    finally:agent.close()
    with pytest.raises(ValueError,match='policy changed'):
        EDAAgent(task,tmp_path/'result',tmp_path/'unused_ledger.json',policy='alternatives')


@pytest.mark.parametrize('committed',[False,True])
def test_decision_delivery_recovery_never_replays_an_action(tmp_path,committed):
    task=write_task(tmp_path/'task')
    agent=EDAAgent(task,tmp_path/'result',tmp_path/'unused_ledger.json')
    command={'operation_id':'prior-question','tool':'get_dataset_report','arguments':{
        'report':'missing','columns':['value'],'knowledge_gap':'Does missingness affect this feature?'}}
    try:
        agent.journal.put('eda_pending_decision',{'command':command,'comment':'original decision'})
        if committed:
            operation,_=agent.journal.begin_operation(agent.run_id,'prior-question','get_dataset_report',command['arguments'])
            operation.status='succeeded';operation.outcome='confirmed'
            operation.result={'status':'success','data':{'report':{'missing_counts':{'value':7}}}}
            agent.journal.save_operation(operation)
        agent.recover_history()
        assert agent.history[0]['command']==command
        assert agent.history[0]['result']['status']==('success' if committed else 'outcome_unknown')
        assert agent.journal.get('eda_pending_decision') is None
        agent.recover_history()
        assert len(agent.history)==1 and agent.budget.counter('eda_fits')==0
    finally:agent.close()
