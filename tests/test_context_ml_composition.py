"""Combined context repair and typed CatBoost weights through real workers."""
import json
from pathlib import Path

import cloudpickle
import httpx
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.controller import Controller
from conftest import write_task


@pytest.mark.parametrize('transport', ['native', 'json'])
def test_parameter_repair_reaches_weighted_catboost_worker_and_final_csv(tmp_path, monkeypatch, transport):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    task = write_task(tmp_path, kind='classification')
    train = pd.read_csv(tmp_path / 'train.csv', dtype=str, keep_default_na=False)
    train['answer'] = train.answer.map({'negative':0, 'positive':1})
    train.to_csv(tmp_path / 'train.csv', index=False)
    spec = json.loads((tmp_path / 'task_spec.json').read_text())
    spec['classes'] = {'answer':[0, 1]}
    (tmp_path / 'task_spec.json').write_text(json.dumps(spec))
    c = Controller(task, Limits(seconds=180, cpu=1, max_calls=5), run_id='combined', transport=transport)
    requests = []
    hypothesis = dict(observation='Confirmed numeric labels in a fixed offline fixture',
        expected_effect='Verify public parameter repair then retain label identity through JSON',
        changed_component='class_weights', acceptance='Completed candidate and a full-refit CSV')

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        delivered = json.dumps(body['messages'])
        repair = len(requests) == 2 and 'accepted_parameter_names' in delivered and 'class_weights' in delivered
        if len(requests) == 1 or repair:
            command = dict(operation_id='corrected' if repair else 'wrong_name', tool='train_model',
                arguments=dict(pipeline=dict(model='catboost', hyperparameters=dict(iterations=5,
                    **{'class_weights' if repair else 'class_weight':({'0':1, '1':2} if repair else 'PRIVATE_WEIGHT_SENTINEL')})),
                    budget_seconds=40, hypothesis=hypothesis))
        else:
            command = dict(operation_id='finish_' + str(len(requests)), tool='finish',
                arguments=dict(result_ids=[c.journal.get('leader_id') or 'baseline'],
                    reason='Offline combined integration fixture complete; keep the measured winner',
                    imputation_skip_reason='Fixed technical scenario',
                    skipped_directions={'all':'This mock verifies delivery and execution, not autonomous quality.'}))
        message = dict(role='assistant', content=json.dumps(command))
        if 'tools' in body:
            message = dict(role='assistant', content=None, tool_calls=[dict(id='call_' + str(len(requests)),
                type='function', function=dict(name=command['tool'], arguments=json.dumps(dict(
                    operation_id=command['operation_id'], **command['arguments']))))])
        return httpx.Response(200, json=dict(choices=[dict(message=message,
            finish_reason='tool_calls' if 'tools' in body else 'stop')],
            usage=dict(prompt_tokens=10, completion_tokens=5)))

    c.qwen.client.close()
    c.qwen.client = httpx.Client(transport=httpx.MockTransport(handle))
    c.qwen.key, c.qwen.base_url = 'offline-fixture', 'http://fixture.invalid/v1'
    try:
        report = c.run()
        candidates = [(s, r) for s, r in c.records() if s.pipeline.model == 'catboost']
        assert len(candidates) == 1 and candidates[0][1].status == 'completed', report
        result = candidates[0][1]
        with (Path(result.artifacts['directory']) / 'model.pkl').open('rb') as stream:
            fitted = cloudpickle.load(stream)
        assert fitted.models['answer'].get_params()['class_weights'] == {0:1, 1:2}
        assert len(requests) == 3
        assert 'accepted_parameter_names' in json.dumps(requests[1]['messages'])
        assert 'PRIVATE_WEIGHT_SENTINEL' not in json.dumps(requests[1]['messages'])
        assert c.journal.operation('wrong_name') is None
        from automl_agent.prompting import PROMPT_VERSION
        assert c.prompt_contract['version'] == PROMPT_VERSION
        assert report['state'] == 'completed' and report['refit_status'] == 'completed', report
        csv = pd.read_csv(c.publisher.output, dtype={'row':str})
        assert list(csv) == ['row', 'answer'] and len(csv) == 25
        assert set(csv.answer) <= {0, 1}
        assert report['publication']['scope'] == 'full_train_refit'
    finally:
        c.close()


@pytest.mark.parametrize('pressure', [False, True])
def test_public_parameter_help_survives_all_compaction_stages_and_core_pressure(tmp_path, monkeypatch, pressure):
    from automl_agent.decision_context import parameter_repair
    from automl_agent.pipelines import UnsupportedParameters
    from test_layered_prompt_context import read_all

    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1), run_id='repair_projection')
    try:
        c.no_llm = True
        c.prepare()
        error = UnsupportedParameters('catboost', ['x' * 100 + str(i) for i in range(20)])
        help = parameter_repair(c, error)
        c.journal.put('last_tool_result', {'status':'command_validation', 'error':str(error)[:1200], 'repair':help})
        if pressure:
            c.journal.put('active_hypothesis', dict(operation_id='next_hypothesis',
                changed_component='capacity ' * 150, observation='evidence ' * 50,
                expected_effect='effect ' * 50, acceptance='criterion ' * 20))
        snap = c.context_memory.snapshot(c.system_prompt)
        for mode in ['native', 'json']:
            for stage in range(4):
                prepared = c.context_memory.compressor.fit(snap, mode, budget=262144, minimum_stage=stage)
                wire = json.loads(prepared.wire_json)
                work = json.loads(wire['messages'][1]['content'])['core']['current_work']
                assert len(json.dumps(work, separators=(',', ':'), ensure_ascii=False).encode()) <= 4096
                projected = work.get('parameter_repair')
                if projected and 'accepted_parameter_names' in projected:
                    assert not pressure
                    assert projected == help
                else:
                    route = projected['read'] if projected else json.loads(work['diagnostic'])['read_routes'][0]
                    assert json.loads(read_all(c, route))['repair'] == help
        assert c.budget.counter('llm_calls') == 0
    finally:
        c.close()
