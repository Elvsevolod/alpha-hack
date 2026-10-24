import numpy as np
import pandas as pd
import pytest

from automl_agent.eda.error_analysis import regression_errors
from tools.diagnose_eda_errors import execute
from conftest import write_task


def test_tail_threshold_comes_from_training_and_error_share_is_real():
    actual = np.r_[np.ones(60), np.full(40, 20.)]
    predicted = np.ones(100)
    X = pd.DataFrame({'feature': np.r_[np.full(40, np.nan), np.ones(60)]})
    result = regression_errors(X, actual, predicted, np.arange(100) / 10)
    assert result['rmse'] == pytest.approx(np.sqrt(40 * 19**2 / 100))
    assert result['tails'][0]['threshold'] == pytest.approx(8.91)
    assert result['tails'][0]['squared_error_fraction'] == 1.
    assert result['missingness'][0]['groups'][0]['rmse'] == 0.


def test_small_error_slices_are_suppressed():
    X = pd.DataFrame({'feature': [np.nan] * 10 + [1.] * 90})
    result = regression_errors(X, np.arange(100), np.zeros(100), np.arange(100))
    assert result['tails'][1]['status'] == 'suppressed'
    assert 'rmse' not in result['tails'][1]
    assert result['missingness'][0]['groups'][0] == {'group': 'missing', 'status': 'suppressed'}


@pytest.mark.parametrize('kind', ['regression', 'classification'])
def test_worker_uses_only_staged_fit_and_never_writes_submission(tmp_path, kind):
    task = write_task(tmp_path / 'task', kind=kind)
    result = execute(task, task.parent / 'task_spec.json', tmp_path / 'diagnostic', seconds=60, cpu=1)
    assert result['status'] == 'completed', result
    report = result['report']
    assert report['fit_rows'] + report['evaluation_rows'] == result['staged_fit_rows']
    assert report['model_fits'] == 1 and report['api_calls'] == 0
    assert not report['outer_validation_used'] and not report['control_used']
    assert not list(tmp_path.rglob('submission.csv'))
