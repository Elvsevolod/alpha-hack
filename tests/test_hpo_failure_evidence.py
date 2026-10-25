import json

import pytest

from automl_agent.contracts import PipelineSpec
from automl_agent.pipelines import build_pipeline, DirectPipeline
from test_pipelines import context


def test_empty_hpo_reports_public_joint_constraint_without_training(tabular, tmp_path, monkeypatch):
    X, y, task = tabular
    X = X.copy()
    X.loc[X.index[0], 'category'] = 'PRIVATE_FEATURE_SENTINEL'
    spec = PipelineSpec(engine='optuna', search_owner='optuna', model='lightgbm', n_trials=1,
        hyperparameters={'subsample_freq': 0},
        search_space={'subsample': {'type': 'float', 'low': .5, 'high': .8}})
    calls = []
    monkeypatch.setattr(DirectPipeline, 'fit', lambda *_: calls.append('unexpected fit'))
    c = {**context(task, tmp_path), 'budget_seconds': 30}
    with pytest.raises(ValueError, match='inactive_row_bagging') as error:
        build_pipeline(spec, task, c).fit(X, y, c)
    assert not calls
    history = json.loads((tmp_path / 'hpo_history.json').read_text())
    assert history['trials'][0]['state'] == 'failed'
    assert history['trials'][0]['recipe_issues']
    assert 'PRIVATE_FEATURE_SENTINEL' not in str(error.value) + json.dumps(history)


def test_backend_exception_text_does_not_enter_public_hpo_failure(tabular, tmp_path, monkeypatch):
    X, y, task = tabular
    spec = PipelineSpec(engine='optuna', search_owner='optuna', model='ridge', n_trials=1,
        search_space={'alpha': {'type': 'float', 'low': .1, 'high': 2}})
    def fail(*_):
        raise ValueError('PRIVATE_BACKEND_SENTINEL')
    monkeypatch.setattr(DirectPipeline, 'fit', fail)
    c = {**context(task, tmp_path), 'budget_seconds': 30}
    with pytest.raises(ValueError, match='no valid inner trial') as error:
        build_pipeline(spec, task, c).fit(X, y, c)
    history = json.loads((tmp_path / 'hpo_history.json').read_text())
    assert history['trials'][0]['failure_type'] == 'ValueError'
    assert 'PRIVATE_BACKEND_SENTINEL' not in str(error.value) + json.dumps(history)


def test_public_constraint_survives_worker_and_controller_error_sanitization(tmp_path):
    from automl_agent.budget import Limits
    from automl_agent.contracts import ToolCommand
    from automl_agent.controller import Controller
    from automl_agent.tool_results import payload
    from conftest import write_task
    c=Controller(write_task(tmp_path),Limits(seconds=180,cpu=1),run_id='hpo_reason')
    try:
        c.prepare()
        result=payload(c.dispatch(ToolCommand(operation_id='bad_bagging',tool='tune_model',arguments={
            'pipeline':{'engine':'optuna','model':'lightgbm','n_trials':1,
                'hyperparameters':{'n_estimators':10,'subsample_freq':0},
                'search_space':{'subsample':{'type':'float','low':.5,'high':.8}}},
            'hypothesis':{'observation':'A bounded row-sampling hypothesis','expected_effect':'Measure an independently rebuilt search',
                'changed_component':'row_bagging','acceptance':'A completed fit or an actionable constraint'},'budget_seconds':60})))
        assert result['result']['status']=='error',result
        assert result['result']['score'] is None
        assert 'inactive_row_bagging' in result['result']['error']
        assert 'subsample_freq' in result['result']['error']
        assert c.budget.counter('llm_calls')==0
    finally:c.close()


@pytest.mark.parametrize('transport', ['native', 'json'])
def test_mock_agent_repairs_hpo_from_delivered_constraint_and_publishes(tmp_path, monkeypatch, transport):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    import numpy as np
    import pandas as pd

    from automl_agent.budget import Limits
    from automl_agent.controller import Controller
    from conftest import write_task

    requests = []
    hypothesis = dict(observation='Fixed offline row sampling fixture',
        expected_effect='Deliver an actionable error then execute the corrected search',
        changed_component='row_bagging', acceptance='A completed bounded fit')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(request)
            received = json.dumps(request['messages'])
            # The correction depends on actual feedback, not just the turn number.
            repair = len(requests) == 2 and 'inactive_row_bagging' in received
            if len(requests) == 1 or repair:
                command = dict(operation_id='repaired_bagging' if repair else 'bad_bagging',
                    tool='tune_model', arguments=dict(hypothesis=hypothesis, budget_seconds=80,
                        pipeline=dict(engine='optuna', model='lightgbm', n_trials=2,
                            hyperparameters=dict(n_estimators=10, subsample_freq=1 if repair else 0),
                            search_space={'subsample':dict(type='float', low=.5, high=.8)})))
            else:
                command = dict(operation_id='finish_fixture_' + str(len(requests)), tool='finish',
                    arguments=dict(result_ids=[controller.journal.get('leader_id') or 'baseline'],
                        reason='Offline transport fixture complete; keep the measured winner',
                        imputation_skip_reason='Fixed technical fixture',
                        skipped_directions={'all':'This fixture verifies feedback and publication, not search quality.'}))
            message = dict(role='assistant', content=json.dumps(command))
            if 'tools' in request:
                message = dict(role='assistant', content=None, tool_calls=[dict(
                    id='fixture_' + str(len(requests)), type='function', function=dict(
                        name=command['tool'], arguments=json.dumps(dict(
                            operation_id=command['operation_id'], **command['arguments']))))])
            body = json.dumps(dict(choices=[dict(message=message,
                finish_reason='tool_calls' if 'tools' in request else 'stop')],
                usage=dict(prompt_tokens=10, completion_tokens=5))).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-fixture')
    monkeypatch.setenv('OPENAI_MODEL', 'offline-fixture')
    monkeypatch.setenv('OPENAI_BASE_URL', f'http://127.0.0.1:{server.server_port}')
    monkeypatch.delenv('OPENAI_EXTRA_BODY', raising=False)
    controller = Controller(write_task(tmp_path / 'task'), Limits(seconds=240, cpu=1),
        run_id='hpo_repair', transport=transport)
    try:
        report = controller.run()
        trials = [(s, r) for s, r in controller.records() if s.pipeline.engine == 'optuna']
        assert any(r is not None and r.status == 'error' and r.score is None
            and 'inactive_row_bagging' in r.error for _, r in trials)
        assert any(s.pipeline.hyperparameters.get('subsample_freq') == 1
            and r is not None and r.status == 'completed' for s, r in trials), report
        assert len(requests) >= 3
        assert 'inactive_row_bagging' not in json.dumps(requests[0]['messages'])
        assert 'inactive_row_bagging' in json.dumps(requests[1]['messages'])
        assert 'subsample_freq' in json.dumps(requests[1]['messages'])
        assert report['state'] == 'completed' and report['refit_status'] == 'completed', report
        csv = pd.read_csv(controller.publisher.output)
        assert list(csv) == ['row', 'answer'] and len(csv) == 25
        assert np.isfinite(csv.answer).all()
        assert report['publication']['scope'] == 'full_train_refit'
    finally:
        controller.close()
        server.shutdown()
        server.server_close()
        thread.join()
