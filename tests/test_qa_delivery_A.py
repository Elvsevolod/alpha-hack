"""Delivery A regressions: recovery changes no model or selection policy."""
import hashlib
import json
import os
from pathlib import Path
import time

import psutil
import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import ToolCommand
from automl_agent.controller import Controller, OperationCompletionFailed
from automl_agent.journal import Journal
from automl_agent.tool_results import payload
from conftest import write_task


@pytest.fixture
def controller(tmp_path):
    c=Controller(write_task(tmp_path),Limits(seconds=180,cpu=1),run_id='qa_A')
    c.prepare()
    try:yield c
    finally:c.close()


def command(identifier='step_26', component='component'):
    return ToolCommand(operation_id=identifier,tool='train_model',arguments={
        'pipeline':{'engine':'direct','model':'ridge','hyperparameters':{'alpha':17}},
        'budget_seconds':20,'hypothesis':{'changed_component':component,'observation':'measured fixture',
            'expected_effect':'verify completion','acceptance':'valid prediction'}})


def test_utf8_component_and_full_hypothesis(controller):
    c=controller;text='a'*122+'я'*6
    assert len(text)==128 and len(text.encode())==134
    result=payload(c.dispatch(command(component=text)))
    operation=c.journal.operation('step_26')
    assert operation.status=='succeeded' and len(operation.component.encode())==128
    assert result['experiment']['hypothesis']['changed_component']==text
    assert not c.journal.incomplete_operations()
    assert payload(c.dispatch(command(component=text)))==result
    assert c.budget.counter('candidate_count')==1


def test_transient_completion_error_does_not_retrain(controller,monkeypatch):
    c=controller;original=c.context_memory.complete;calls=[]
    def once(operation):
        calls.append(operation.operation_id)
        if len(calls)==1:raise ValueError('injected completion failure')
        return original(operation)
    monkeypatch.setattr(c.context_memory,'complete',once)
    result=payload(c.dispatch(command()))
    assert calls==['step_26','step_26'] and result['result']['status']=='completed'
    assert c.budget.counter('candidate_count')==1 and c.journal.operation('step_26').status=='succeeded'
    assert c.journal.get('completion_recovery:step_26')['attempted']
    assert not c.journal.get('completion_intent:step_26')


def test_crash_after_experiment_commit_recovers_without_execution(controller,monkeypatch):
    c=controller;cmd=command()
    complete=c.complete_operation
    monkeypatch.setattr(c,'complete_operation',lambda *a:(_ for _ in ()).throw(SystemExit('process crash')))
    with pytest.raises(SystemExit):c.dispatch(cmd)
    monkeypatch.setattr(c,'complete_operation',complete)
    before=c.budget.counter('candidate_count')
    monkeypatch.setattr(c.executor,'run',lambda *a:pytest.fail('repeated training'))
    recovered=payload(c.dispatch(cmd))
    assert recovered['recovered'] and recovered['result']['status']=='completed'
    assert c.journal.operation(cmd.operation_id).status=='succeeded' and c.budget.counter('candidate_count')==before



def test_persistent_completion_fault_has_terminal_outbox_and_stops(controller,monkeypatch):
    c=controller;calls=[];csv=c.publisher.output.read_bytes()
    def fail(operation):calls.append(operation.operation_id);raise ValueError('persistent metadata fault')
    monkeypatch.setattr(c.context_memory,'complete',fail)
    with pytest.raises(OperationCompletionFailed):c.dispatch(command())
    assert calls==['step_26','step_26'] and c.state=='frozen'
    assert c.journal.operation('step_26').status=='failed'
    assert c.context_memory.store.db.execute("SELECT 1 FROM cc_observations WHERE operation_id='step_26'").fetchone()
    assert c.publisher.output.read_bytes()==csv


def test_live_executor_is_not_reported_as_completed(controller):
    c=controller;cmd=command();args=c.context_memory.arguments(cmd)
    op,_=c.journal.begin_operation(c.run_id,cmd.operation_id,cmd.tool,{'arguments':args.model_dump(mode='json'),'data':c.dataset.source_hash,'protocol':c.plan.hash})
    op.status='running';c.journal.save_operation(op)
    c.journal.put('active_worker',{'pid':os.getpid(),'create_time':psutil.Process().create_time()})
    result=c.dispatch(cmd)
    assert result['status']=='in_progress' and result['command_completed'] is False
    assert c.journal.operation('step_26').status=='running'
    c.journal.put('active_worker',None)


def test_duplicate_returns_identity_without_claiming_training(controller):
    c=controller;first=payload(c.dispatch(command()))
    second=payload(c.dispatch(command(identifier='duplicate')))
    assert second['status']=='duplicate_computation'
    assert second['existing_experiment_id']==first['experiment']['experiment_id']
    assert second['computation_hash']==first['experiment']['computation_hash'] if 'computation_hash' in first['experiment'] else len(second['computation_hash'])==64
    assert c.budget.counter('candidate_count')==1


def test_metadata_identity_rejected_before_work(tmp_path):
    j=Journal(tmp_path)
    try:
        with pytest.raises(ValueError):j.begin_operation('run','x','train_model',{'arguments':{'pipeline':{'engine':'x'*129}}})
        assert j.operation('x') is None
    finally:j.close()


def test_full_redacted_traceback_for_caught_command(controller,monkeypatch):
    c=controller;secret='sk-qaTraceSecret0123456789';monkeypatch.setenv('QA_SECRET_TOKEN',secret)
    cmd=ToolCommand(operation_id='bad_ref',tool='train_model',arguments={**command().arguments,'reference_ids':[secret]})
    result=payload(c.dispatch(cmd));path=c.workdir/result['traceback_ref'];text=path.read_text()
    assert 'Traceback' in text and 'Unknown API documentation ID' in text and secret not in text
    assert result['error_category']=='command' and c.budget.counter('candidate_count')==0
