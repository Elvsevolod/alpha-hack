from automl_agent.tool_results import payload as tool_payload
import json
from pathlib import Path
import time

import pytest

from automl_agent.budget import Budget, Limits
from automl_agent.contracts import Hypothesis, PipelineSpec, ToolCommand
from automl_agent.controller import Controller
from automl_agent.custom import validate_code
from automl_agent.journal import Journal
from automl_agent.util import file_hash, read_json
from conftest import write_task


def test_operation_identity_and_transport_replay(tmp_path):
    j=Journal(tmp_path); a={'arguments':{'x':1},'protocol':'a'}
    op,new=j.begin_operation('run','same','train_model',a);assert new
    op.status='succeeded';op.result={'score':1};j.save_operation(op)
    old,new=j.begin_operation('run','same','train_model',{'arguments':{'x':1},'protocol':'b'})
    assert not new and old.result=={'score':1}
    with pytest.raises(ValueError):j.begin_operation('run','same','train_model',{'arguments':{'x':2},'protocol':'b'})
    j.close()


def test_deadline_and_counters_survive_resume(tmp_path):
    j=Journal(tmp_path);b=Budget(j,Limits(seconds=100),time.time());deadline=b.deadline;b.counter('candidate_count',3)
    c=Budget(j,Limits(seconds=10000),time.time())
    assert c.deadline==deadline and c.counter('candidate_count')==3 and c.limits.seconds==100
    j.close()


def test_states_cannot_reopen_search(tmp_path):
    j=Journal(tmp_path)
    for state in ('searching','frozen','control','refitting','completed'):j.transition(state,'test')
    with pytest.raises(ValueError):j.transition('searching','reopen')
    j.close()


@pytest.mark.parametrize('code',["import os\ndef build_pipeline(context): return os.environ", "import socket\ndef build_pipeline(context): return None",
    "def build_pipeline(context):\n return open('x')", "import numpy as np\ndef build_pipeline(context):\n return np.load('x')"])
def test_generated_io_and_import_policy(code):
    with pytest.raises(ValueError):validate_code(code,'pipeline')


def test_full_no_llm_run_and_publication_recovery(tmp_path,monkeypatch):
    path=write_task(tmp_path); c=Controller(path,Limits(seconds=90,cpu=1),run_id='run')
    try:
        report=c.run(no_llm=True)
        assert report['state']=='completed',report
        assert report['control_status']=='completed' and report['refit_status']=='completed'
        assert report['publication']['scope']=='full_train_refit'
        assert report['execution_policy']['mode']=='limited_trust'
        probes=report['execution_policy']['tests']['probes']
        assert probes['source_train']=='blocked_python_io' and probes['credentials_file'] in {'blocked_python_io','unavailable'}
        assert probes['credential_environment']=='absent' and probes['network']=='blocked_python_socket'
        assert probes['installer_subprocess']=='blocked_python_process'
        saved=c.journal.get('publication');c.journal.put('publication_pending',saved)
        c.publisher.output.write_text('interrupted publication')
        c.publisher.recover();assert file_hash(c.publisher.output)==saved['csv_sha256']
    finally:c.close()


def test_unknown_outcome_is_not_retrained(tmp_path):
    path=write_task(tmp_path);c=Controller(path,Limits(seconds=90,cpu=1),run_id='run')
    try:
        c.prepare();op,new=c.journal.begin_operation(c.run_id,'lost','train_model',{'arguments':{'pipeline':'x'},'protocol':c.plan.hash})
        op.status='running';c.journal.save_operation(op);before=len(c.records())
        c.recover_operations();recovered=c.journal.operation('lost')
        assert recovered.outcome=='outcome_unknown' and len(c.records())==before
    finally:c.close()


def test_worker_failure_retains_prior_csv_and_debug_limit(tmp_path):
    path=write_task(tmp_path);c=Controller(path,Limits(seconds=150,cpu=1),run_id='run')
    try:
        c.prepare();before=file_hash(c.publisher.output)
        code="def build_pipeline(context):\n    raise RuntimeError('intentional technical fixture')\n"
        args={'pipeline':PipelineSpec(engine='custom',model='ridge').model_dump(mode='json'),
            'hypothesis':Hypothesis(observation='fixture',expected_effect='fixture',changed_component='fixture',acceptance='valid').model_dump(),
            'budget_seconds':20,'code':code}
        first=tool_payload(c.dispatch(ToolCommand(operation_id='bad',tool='run_custom_experiment',arguments=args)))
        assert first['result']['status']=='error' and file_hash(c.publisher.output)==before
        # The same native/JSON command returns the already completed outcome.
        assert tool_payload(c.dispatch(ToolCommand(operation_id='bad',tool='run_custom_experiment',arguments=args)))==first
        assert c.budget.counter('candidate_count')==1
    finally:c.close()


def test_refit_failure_keeps_control_winner_csv(tmp_path,monkeypatch):
    path=write_task(tmp_path);c=Controller(path,Limits(seconds=90,cpu=1),run_id='run')
    original=c.executor.run
    def fail_refit(directory,request,seconds):
        if request['mode']=='refit' and request.get('predict_splits')==['test']:
            return {'status':'memory_limit','error':'controlled failure'}
        return original(directory,request,seconds)
    monkeypatch.setattr(c.executor,'run',fail_refit)
    try:
        r=c.run(no_llm=True)
        assert r['state']=='completed' and r['refit_status']=='refit_failed'
        assert r['publication']['scope']=='control_prepared'
        assert file_hash(c.publisher.output)==r['publication']['csv_sha256']
    finally:c.close()
