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
