from automl_agent.tool_results import payload as tool_payload
import json
from pathlib import Path
import time

import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis, InspectArgs, PipelineSpec, ToolCommand
from automl_agent.controller import Controller
from automl_agent.custom import validate_code
from automl_agent.data import Dataset
from automl_agent.util import canonical
from conftest import write_task
from tools.offline_acceptance import fixture


HYPOTHESIS = Hypothesis(observation="Technical boundary fixture", expected_effect="Verify the saved contract",
    changed_component="boundary", acceptance="Aligned finite predictions and preserved provenance")


def test_documented_roles_survive_an_unresolved_metric(tmp_path):
    path = write_task(tmp_path, temporal=True, groups=True)
    (tmp_path / 'task_spec.json').unlink()
    for name in ('train.csv', 'test.csv'):
        frame = pd.read_csv(tmp_path / name, dtype=str, keep_default_na=False)
        frame['notes'] = 'some free text'
        frame.to_csv(tmp_path / name, index=False)
    path.write_text('Regression task. CSV columns row,answer.\n```csv\nrow,answer\n```\n'
        'when: time column\nquery: group identifier\nnotes: free text\nvalue: categorical feature\n')
    dataset = Dataset(path)
    task, questions = dataset.draft()
    assert task is None and 'metric' in questions
    assert dataset.fixed['time_column'] == 'when' and dataset.fixed['group_column'] == 'query'
    assert dataset.fixed['text_columns'] == ['notes']
    assert 'value' in dataset.fixed['categorical_columns']
    assert 'notes' not in dataset.fixed['categorical_columns']


def test_forecasting_accepts_a_time_identifier_without_exogenous_features(tmp_path):
    pd.DataFrame({'when': range(30), 'target': range(30)}).to_csv(tmp_path / 'train.csv', index=False)
    pd.DataFrame({'when': range(30, 33)}).to_csv(tmp_path / 'test.csv', index=False)
    pd.DataFrame({'when': range(30, 33), 'target': [0] * 3}).to_csv(tmp_path / 'sample_submission.csv', index=False)
    path = tmp_path / 'task.md'
    path.write_text('Technical univariate forecast fixture.')
    (tmp_path / 'task_spec.json').write_text(json.dumps({'kind': 'forecasting', 'targets': ['target'],
        'id_columns': ['when'], 'feature_columns': [], 'time_column': 'when', 'horizon': 3,
        'metric': {'name': 'rmse'}, 'output': {'columns': ['when', 'target'], 'prediction_type': 'values'}}))
    controller = Controller(path, Limits(seconds=60, cpu=1), run_id='univariate')
    try:
        report = controller.run(no_llm=True)
        assert report['state'] == 'completed' and report['control_status'] == 'completed', report
        assert report['refit_status'] == 'completed'
    finally:
        controller.close()


def test_only_no_argument_model_eval_is_permitted():
    validate_code('class Estimator:\n def eval(self): return self\ndef build_pipeline(context):\n model=Estimator()\n model.eval()\n return model', 'pipeline')
    with pytest.raises(ValueError):
        validate_code('import pandas as pd\ndef build_pipeline(context):\n return pd.eval("1+1")', 'pipeline')


def test_custom_analysis_returns_only_declared_large_aggregates(tmp_path):
    # This checks disclosure, not cold-start throughput. Deadline enforcement
    # has separate tests; leave headroom for the isolated worker imports.
    controller = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1), run_id='analysis')
    try:
        controller.prepare()
        code = '''def analyze(X,y,context):
    return {"statistics":[
        {"name":"fit_mean","aggregation":"mean","columns":["answer"],"value":float(y['answer'].mean()),"support_rows":len(y)},
        {"name":"small","aggregation":"mean","columns":["answer"],"value":123.456,"support_rows":1}]}
'''
        arguments = {'pipeline': PipelineSpec(engine='custom', model='analysis', custom_component='analysis').model_dump(mode='json'),
            'hypothesis': HYPOTHESIS.model_dump(), 'budget_seconds': 45, 'code': code}
        result = tool_payload(controller.dispatch(ToolCommand(operation_id='aggregate', tool='run_custom_experiment', arguments=arguments)))
        assert result['result']['status'] == 'completed', result
        analysis = result['result']['training_summary']['analysis']
        assert analysis['statistics'][0]['status'] == 'available'
        assert analysis['statistics'][1]['status'] == 'suppressed'
        assert 'value' not in analysis['statistics'][1] and '123.456' not in canonical(analysis)
        arguments['code'] = "def analyze(X,y,context):\n return {'rows':y.to_numpy().tolist()}\n"
        rejected = tool_payload(controller.dispatch(ToolCommand(operation_id='rows', tool='run_custom_experiment', arguments=arguments)))
        assert rejected['result']['status'] == 'error'
    finally:
        controller.close()


def test_context_summarizes_inner_row_keys_without_mutating_the_saved_result(tmp_path):
    controller = Controller(write_task(tmp_path), Limits(seconds=90, cpu=1), run_id='context')
    try:
        controller.prepare()
        spec, saved = controller.lookup('baseline')
        saved.training_summary['inner_scores'] = {'split': {'fit_keys': ['fit_a', 'fit_b'], 'validation_keys': ['fit_c']},
            'trials': [{'number': i} for i in range(10)]}
        public = controller.public_result(saved)
        split = public['training_summary']['inner_scores']['split']
        assert 'fit_keys' not in split and split['fit_keys_count'] == 2 and len(split['fit_keys_hash']) == 64
        assert public['training_summary']['inner_scores']['trial_count'] == 10
        assert len(public['training_summary']['inner_scores']['trials']) == 4
        assert saved.training_summary['inner_scores']['split']['fit_keys'] == ['fit_a', 'fit_b']
    finally:
        controller.close()


def test_ensemble_refits_complete_graph_and_rejects_a_lost_model(tmp_path, monkeypatch):
    controller = Controller(write_task(tmp_path), Limits(seconds=150, cpu=1), run_id='ensemble')
    try:
        controller.prepare()
        arguments = {'pipeline': PipelineSpec(model='ridge', hyperparameters={'alpha': 2}).model_dump(mode='json'),
            'hypothesis': HYPOTHESIS.model_dump(), 'budget_seconds': 20}
        trained = tool_payload(controller.dispatch(ToolCommand(operation_id='member', tool='train_model', arguments=arguments)))
        member = trained['experiment']['experiment_id']
        ensemble = tool_payload(controller.dispatch(ToolCommand(operation_id='blend', tool='build_ensemble', arguments={
            'members': ['baseline', member], 'hypothesis': HYPOTHESIS.model_dump(), 'budget_seconds': 20})))
        assert ensemble['result']['status'] == 'completed', ensemble
        nested = tool_payload(controller.dispatch(ToolCommand(operation_id='nested', tool='build_ensemble', arguments={
            'members': [ensemble['experiment']['experiment_id'], member], 'hypothesis': HYPOTHESIS.model_dump(), 'budget_seconds': 20})))
        assert nested['result']['status'] == 'completed', nested
        controller.journal.put('leader_id', nested['experiment']['experiment_id'])
        controller.freeze('Exercise the complete dependency graph')
        assert set(controller.journal.get('finalization_plan')['policies']) >= {'baseline', member}
        calls = []
        original = controller.executor.run
        def once(directory, request, seconds):
            calls.append((directory.parent.name, directory.name))
            return original(directory, request, seconds)
        monkeypatch.setattr(controller.executor, 'run', once)
        controller.finalize()
        assert controller.journal.get('control_status') == 'completed'
        assert controller.journal.get('refit_status') == 'completed'
        scores = controller.journal.get('control_scores')
        winner = controller.journal.get('control_winner')
        assert scores[winner] == min(scores.values())
        from automl_agent.completion import dependency_order
        # Blends are local arithmetic; only fitted base nodes launch workers.
        expected = sum(controller.lookup(identifier)[0].pipeline.engine != 'ensemble'
            for identifier in dependency_order(controller, controller.journal.get('finalization_plan')['finalists']))
        expected += sum(controller.lookup(identifier)[0].pipeline.engine != 'ensemble'
            for identifier in dependency_order(controller, [winner]))
        assert len(calls) == len(set(calls)) == expected
        _, result = controller.lookup(member)
        model = Path(result.artifacts['directory']) / 'model.pkl'
        model.unlink()
        with pytest.raises(ValueError, match='missing or corrupt'):
            controller.verify_dependency_artifacts(member)
    finally:
        controller.close()


def test_revision_cannot_remove_declared_groups(tmp_path):
    controller = Controller(write_task(tmp_path, groups=True), Limits(seconds=90, cpu=1), run_id='groups')
    try:
        controller.prepare()
        with pytest.raises(ValueError, match='boundaries'):
            controller.revise_validation(InspectArgs(report='validation_proposal', proposed_strategy='random', rationale='Technical negative fixture'))
    finally:
        controller.close()


def test_panel_forecast_pilot_uses_future_features_without_external_targets(tmp_path, monkeypatch):
    path = fixture(tmp_path, 'forecasting')
    controller = Controller(path, Limits(seconds=120, cpu=1), run_id='panel')
    try:
        controller.prepare()
        probes = []
        original = controller.executor.run
        def inspect_staging(directory, request, seconds):
            if directory.name in {'smoke','performance_1','performance_2'}:
                probes.append(json.loads((directory / 'probe_X.json').read_text()))
                assert not (directory / 'probe_y.json').exists()
            return original(directory, request, seconds)
        monkeypatch.setattr(controller.executor, 'run', inspect_staging)
        arguments = {'pipeline': PipelineSpec(model='ridge', representation='lags').model_dump(mode='json'),
            'hypothesis': HYPOTHESIS.model_dump(), 'budget_seconds': 35}
        result = tool_payload(controller.dispatch(ToolCommand(operation_id='lag_model', tool='train_model', arguments=arguments)))
        assert result['result']['status'] == 'completed', result
        spec, outcome = controller.lookup(result['experiment']['experiment_id'])
        pilot = Path(outcome.artifacts['directory']) / 'smoke'
        probe = probes[0]
        assert set(probe['index']) <= {controller.dataset.row_keys[i] for i in controller.plan.validation}
        assert not (pilot / 'probe_y.json').exists()
        assert not (pilot / 'probe_X.json').exists()  # verified reproducible staging was reclaimed
    finally:
        controller.close()


def test_slower_finalist_receives_its_measured_share_of_preparation_budget(tmp_path, monkeypatch):
    controller = Controller(write_task(tmp_path), Limits(seconds=360, cpu=1), run_id='preparation')
    try:
        controller.prepare()
        arguments = {'pipeline': PipelineSpec(model='ridge', hyperparameters={'alpha': 2}).model_dump(mode='json'),
            'hypothesis': HYPOTHESIS.model_dump(), 'budget_seconds': 20}
        outcome = tool_payload(controller.dispatch(ToolCommand(operation_id='slower', tool='train_model', arguments=arguments)))
        identifier = outcome['experiment']['experiment_id']
        for name, seconds in (('baseline', 5), (identifier, 45)):
            spec, result = controller.lookup(name); result.seconds = seconds
            controller.journal.save_experiment(spec, result)
        controller.journal.put('leader_id', identifier)
        controller.budget.deadline = time.time() + 300
        original = controller._prepare_finalist
        allocations = []
        def measured(identifier, rows, part, seconds):
            if part == 'control':allocations.append((identifier, seconds))
            return original(identifier, rows, part, seconds)
        monkeypatch.setattr(controller, '_prepare_finalist', measured)
        controller.freeze('Reproduce the 45-second versus 5-second allocation case')
        controller.finalize()
        assert dict(allocations)[identifier] > 60, allocations
        assert controller.journal.get('control_status') == 'completed'
    finally:
        controller.close()
