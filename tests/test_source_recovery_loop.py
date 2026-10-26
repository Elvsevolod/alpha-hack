"""Replay the observed bad source through HTTP parsing and real ML completion.

The model is scripted: this verifies recovery delivery, not autonomous quality.
"""
import json

import httpx
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.controller import Controller
from conftest import write_task


@pytest.mark.parametrize('transport', ['native', 'json'])
def test_rejected_source_reaches_next_decision_then_research_and_full_refit(tmp_path, monkeypatch, transport):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    c = Controller(write_task(tmp_path, kind='classification'),
        Limits(seconds=180, cpu=1, max_calls=5), run_id='source_recovery', transport=transport)
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        context = json.loads(body['messages'][1]['content'])
        work = context['core']['current_work']
        if len(requests) == 1:
            command = dict(operation_id='invalid_source', tool='read_run_records', arguments={
                'source':{'source_id':work['research_options']['arguments']['ids'][0],
                    'version':'', 'scope':'reference'}})
        elif len(requests) == 2:
            diagnostic = json.loads(work['diagnostic'])
            assert diagnostic['status'] == 'command_validation'
            assert 'version' in diagnostic['error']
            assert c.journal.operation('invalid_source') is None
            command = dict(operation_id='read_options', **work['research_options'])
        elif len(requests) == 3:
            assert c.journal.operation('read_options').status == 'succeeded'
            command = dict(operation_id='research_after_repair', tool='train_model', arguments={
                'pipeline':{'model':'extra_trees', 'hyperparameters':{'n_estimators':10, 'max_depth':3}},
                'budget_seconds':30, 'hypothesis':{
                    'observation':'Offline protocol replay after a rejected source read',
                    'expected_effect':'Verify the next accepted command reaches a real worker',
                    'changed_component':'extra_trees capacity',
                    'acceptance':'Completed fit and final publication; no quality gain claimed'}})
        else:
            command = dict(operation_id='finish_replay', tool='finish', arguments={
                'result_ids':[c.journal.get('leader_id') or 'baseline'],
                'reason':'Offline recovery replay complete',
                'imputation_skip_reason':'Fixed integration fixture',
                'skipped_directions':{'all':'Scripted technical test, not a search-quality comparison'}})
        message = {'role':'assistant','content':json.dumps(command)}
        if 'tools' in body:
            message = {'role':'assistant','content':None,'tool_calls':[{
                'id':'call_' + str(len(requests)), 'type':'function', 'function':{
                    'name':command['tool'], 'arguments':json.dumps({
                        'operation_id':command['operation_id'], **command['arguments']})}}]}
        return httpx.Response(200, json={'choices':[{'message':message,
            'finish_reason':'tool_calls' if 'tools' in body else 'stop'}],
            'usage':{'prompt_tokens':10,'completion_tokens':5}})

    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(handle))
    c.qwen.key, c.qwen.base_url = 'offline-fixture', 'http://fixture.invalid/v1'
    try:
        report = c.run()
        assert len(requests) == 4, report
        assert all(('tools' in body) == (transport == 'native') for body in requests)
        assert c.journal.operation('invalid_source') is None
        counts = dict(c.journal.db.execute('SELECT kind,COUNT(*) FROM events GROUP BY kind'))
        assert counts.get('command_arguments_rejected', 0) == 1
        assert counts.get('native_fallback', 0) == 0
        assert counts.get('qwen_failure', 0) == 1
        assert counts.get('qwen_decision', 0) == 3
        assert c.journal.operation('research_after_repair').status == 'succeeded'
        candidates = [(s,r) for s,r in c.records() if s.pipeline.model == 'extra_trees']
        assert len(candidates) == 1 and candidates[0][1].status == 'completed'
        assert report['refit_status'] == 'completed' and report['state'] == 'completed'
        assert report['publication']['scope'] == 'full_train_refit'
        output = pd.read_csv(c.publisher.output, dtype={'row':str})
        template = pd.read_csv(tmp_path / 'sample_submission.csv', dtype={'row':str})
        assert list(output) == list(template) and output.row.tolist() == template.row.tolist()
        assert output.answer.notna().all()
    finally:
        c.close()
