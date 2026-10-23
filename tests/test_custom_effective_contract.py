"""Check the contract actually executed after custom-code engine conversion."""
import pytest
import pandas as pd

from automl_agent.budget import Limits
from automl_agent.contracts import ToolCommand
from automl_agent.controller import Controller
from automl_agent.tool_results import payload
from conftest import write_task
from test_custom_execution import VALID_CODE


@pytest.mark.parametrize('pipeline,classes,error', [
    ({'prediction_policy': 'threshold', 'postprocess_fit_policy': 'inner_holdout'}, 2,
        'custom_postprocessing_has_no_verified_inner_holdout'),
    ({'engine': 'direct', 'prediction_policy': 'calibrated', 'postprocess_fit_policy': 'inner_holdout'}, 2,
        'custom_postprocessing_has_no_verified_inner_holdout'),
    ({'prediction_policy': 'class_offsets', 'postprocess_fit_policy': 'inner_holdout'}, 3,
        'class_offsets_require_direct_or_optuna_frozen_fit_path'),
    ({'engine': 'direct', 'fit_diagnostics': 'inner_holdout'}, 2,
        'fit_diagnostics_require_direct_or_optuna_single_classification'),
    ({'engine': 'custom', 'prediction_policy': 'threshold', 'postprocess_fit_policy': 'inner_holdout'}, 2,
        'custom_postprocessing_has_no_verified_inner_holdout'),
    ({'engine': 'optuna', 'search_owner': 'optuna'}, 2, 'Nested optimizers are forbidden'),
])
def test_omitted_custom_engine_cannot_bypass_inner_holdout_restriction(tmp_path, monkeypatch, pipeline, classes, error):
    task = write_task(tmp_path, kind='classification')
    if classes == 3:
        frame = pd.read_csv(tmp_path / 'train.csv', dtype={'row': str})
        frame.loc[frame.index % 3 == 0, 'answer'] = 'third'
        frame.to_csv(tmp_path / 'train.csv', index=False)
    controller = Controller(task, Limits(seconds=180, cpu=1))
    try:
        controller.prepare()
        before = controller.publisher.output.read_bytes()
        count = controller.budget.counter('candidate_count')
        executed = []
        def execute(spec, **_):
            executed.append(spec)
            return controller.failed_result(spec, {'status': 'error', 'error': 'offline execution sentinel'})
        monkeypatch.setattr(controller, 'execute_experiment', execute)
        result = payload(controller.dispatch(ToolCommand(operation_id='effective_custom', tool='run_custom_experiment', arguments={
            'pipeline': {'model': 'lightgbm', 'custom_component': 'transformer', **pipeline},
            'code': VALID_CODE, 'budget_seconds': 30,
            'hypothesis': {'observation': 'Custom features fit before the direct inner split',
                'expected_effect': 'Require the effective custom holdout contract',
                'changed_component': 'custom features', 'acceptance': 'Reject before candidate execution'}})))
        assert not executed, 'Unverified custom postprocessing reached candidate execution'
        assert result['status'] in {'rejected', 'invalid_command_or_api'}
        assert error in result['error'], result
        assert controller.budget.counter('candidate_count') == count
        assert controller.publisher.output.read_bytes() == before and controller.state == 'searching'
    finally:
        controller.close()
