"""Recover the actual rejected LightGBM alias without duplicate fits or changing LightGBM precedence."""
import numpy as np
import pytest

from automl_agent.contracts import PipelineSpec, ToolCommand
from automl_agent.pipelines import effective_parameters, make_estimator
from automl_agent.tool_results import payload


@pytest.mark.parametrize('model', ['lightgbm', 'lgbm_ranker'])
def test_official_alias_is_canonical_and_source_is_unchanged(model):
    a = PipelineSpec(model=model, hyperparameters={'feature_fraction': .8, 'n_estimators': 30})
    b = PipelineSpec(model=model, hyperparameters={'colsample_bytree': .8, 'n_estimators': 30})
    assert effective_parameters(a) == effective_parameters(b)
    assert a.hyperparameters == {'feature_fraction': .8, 'n_estimators': 30}
    a.hyperparameters['colsample_bytree'] = .8
    assert effective_parameters(a) == effective_parameters(b)


@pytest.mark.parametrize('classification', [False, True])
def test_alias_and_canonical_spelling_produce_identical_actual_predictions(classification):
    rng = np.random.default_rng(52)
    X = rng.normal(size=(200, 8)); y = (X[:, 0] > 0).astype(int) if classification else X[:, 0] + X[:, 1] * .5
    common = {'n_estimators': 20, 'min_child_samples': 5}
    alias = make_estimator(PipelineSpec(model='lightgbm', hyperparameters={**common, 'feature_fraction': .8}), classification, {'cpu': 1})
    canonical = make_estimator(PipelineSpec(model='lightgbm', hyperparameters={**common, 'colsample_bytree': .8}), classification, {'cpu': 1})
    alias.fit(X, y); canonical.fit(X, y)
    np.testing.assert_array_equal(alias.predict(X), canonical.predict(X))
    assert alias.get_params()['colsample_bytree'] == .8


def test_conflicting_aliases_are_rejected_and_other_namespaces_are_unchanged():
    spec = PipelineSpec(model='lightgbm', hyperparameters={'feature_fraction': .8, 'colsample_bytree': .7})
    with pytest.raises(ValueError, match='Conflicting'):
        effective_parameters(spec)
    other = PipelineSpec(model='xgboost', hyperparameters={'feature_fraction': .8})
    assert effective_parameters(other) == {'feature_fraction': .8}
    with pytest.raises(ValueError, match='Unsupported parameters'):
        make_estimator(other, False, {'cpu': 1})


def test_controller_rejects_alias_duplicate_and_primary_precedence_duplicate_before_any_new_worker(tmp_path, monkeypatch):
    from conftest import write_task
    from automl_agent.budget import Limits
    from automl_agent.controller import Controller
    c = Controller(write_task(tmp_path), Limits(seconds=180, cpu=1))
    hypothesis = {'observation': 'Actual official parameter alias', 'expected_effect': 'Compare feature subsampling',
        'changed_component': 'feature sampling', 'acceptance': 'One complete canonical candidate'}
    try:
        c.prepare()
        pipeline = {'model': 'lightgbm', 'hyperparameters': {'n_estimators': 20, 'feature_fraction': .8}}
        first = payload(c.dispatch(ToolCommand(operation_id='alias_first', tool='train_model', arguments={
            'pipeline': pipeline, 'budget_seconds': 30, 'hypothesis': hypothesis})))
        assert first['result']['status'] == 'completed', first
        spec, _ = c.lookup(first['experiment']['experiment_id'])
        assert spec.pipeline.hyperparameters == {'n_estimators': 20, 'colsample_bytree': .8}
        previous = c.publisher.output.read_bytes(); count = c.budget.counter('candidate_count')
        monkeypatch.setattr(c.executor, 'run', lambda *args: pytest.fail('Alias duplicate or conflict started a worker'))
        second = payload(c.dispatch(ToolCommand(operation_id='canonical_duplicate', tool='train_model', arguments={
            'pipeline': {'model': 'lightgbm', 'hyperparameters': {'n_estimators': 20, 'colsample_bytree': .8}},
            'budget_seconds': 30, 'hypothesis': hypothesis})))
        assert second['status'] == 'duplicate_computation' and 'already' in second['error'], second
        conflict = payload(c.dispatch(ToolCommand(operation_id='alias_conflict', tool='train_model', arguments={
            'pipeline': {'model': 'lightgbm', 'hyperparameters': {'n_estimators': 20, 'feature_fraction': .8, 'colsample_bytree': .7}},
            'budget_seconds': 30, 'hypothesis': hypothesis})))
        assert conflict['status'] == 'rejected' and 'Conflicting' in conflict['error']
        assert c.budget.counter('candidate_count') == count and c.publisher.output.read_bytes() == previous
    finally:
        c.close()


def test_optuna_sample_overrides_normalized_base_and_freezes_the_actual_model(tabular, tmp_path):
    from automl_agent.search import SearchPipeline
    from test_pipelines import context
    X, y, task = tabular
    spec = PipelineSpec(engine='optuna', search_owner='optuna', model='lightgbm', n_trials=2,
        hyperparameters={'n_estimators': 10, 'colsample_bytree': .9},
        search_space={'feature_fraction': {'type': 'categorical', 'choices': [.6]}})
    c = context(task, tmp_path); c['budget_seconds'] = 20
    pipeline = SearchPipeline(spec, task)
    fitted = pipeline.fit(X, y, c)
    assert len(fitted.inner_scores['trials']) == 1  # A one-point space is not fitted twice.
    assert effective_parameters(pipeline.frozen_spec())['colsample_bytree'] == .6
    assert pipeline.pipeline.models['answer'].get_params()['colsample_bytree'] == .6
    before = pipeline.predict(X.iloc[:10], c).values
    pipeline.refit(X, y, c)
    assert pipeline.pipeline.models['answer'].get_params()['colsample_bytree'] == .6
    assert pipeline.predict(X.iloc[:10], c).values == before


@pytest.mark.parametrize('model', ['lightgbm', 'lgbm_ranker'])
def test_recorded_bagging_and_regularization_names_normalize_without_mutating_source(model):
    original = {'bagging_fraction': .8, 'bagging_freq': 1, 'lambda_l2': 2.}
    spec = PipelineSpec(model=model, hyperparameters=original.copy())
    assert effective_parameters(spec) == {'subsample': .8, 'subsample_freq': 1, 'reg_lambda': 2.}
    assert spec.hyperparameters == original
    estimator = make_estimator(spec, False, {'cpu': 1})
    assert estimator.get_params()['subsample'] == .8
    assert estimator.get_params()['subsample_freq'] == 1
    assert estimator.get_params()['reg_lambda'] == 2.
    other = PipelineSpec(model='xgboost', hyperparameters=original.copy())
    assert effective_parameters(other) == original
    with pytest.raises(ValueError, match='Unsupported parameters'):
        make_estimator(other, False, {'cpu': 1})


@pytest.mark.parametrize('classification', [False, True])
def test_new_names_match_installed_lightgbm_primary_name_precedence(classification):
    from lightgbm import LGBMClassifier, LGBMRegressor
    rng = np.random.default_rng(53)
    X = rng.normal(size=(300, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int) if classification else X[:, 0] + .5 * X[:, 1]
    parameters = {'n_estimators': 20, 'min_child_samples': 5, 'bagging_fraction': .8,
        'bagging_freq': 1, 'lambda_l2': 2.}
    actual = make_estimator(PipelineSpec(model='lightgbm', hyperparameters=parameters), classification, {'cpu': 1})
    reference = (LGBMClassifier if classification else LGBMRegressor)(
        n_jobs=1, random_state=42, verbosity=-1, **parameters)
    actual.fit(X, y); reference.fit(X, y)
    predict = 'predict_proba' if classification else 'predict'
    np.testing.assert_array_equal(getattr(actual, predict)(X), getattr(reference, predict)(X))


def test_optuna_new_primary_names_override_canonical_base_through_refit(tabular, tmp_path):
    from automl_agent.search import SearchPipeline
    from test_pipelines import context
    X, y, task = tabular
    spec = PipelineSpec(engine='optuna', search_owner='optuna', model='lightgbm', n_trials=2,
        hyperparameters={'n_estimators': 10, 'subsample': .9, 'subsample_freq': 0, 'reg_lambda': 0.},
        search_space={'bagging_fraction': {'type': 'categorical', 'choices': [.7]},
            'bagging_freq': {'type': 'categorical', 'choices': [1]},
            'lambda_l2': {'type': 'categorical', 'choices': [2.]}})
    c = context(task, tmp_path); c['budget_seconds'] = 20
    pipeline = SearchPipeline(spec, task)
    summary = pipeline.fit(X, y, c)
    assert len(summary.inner_scores['trials']) == 1
    expected = {'subsample': .7, 'subsample_freq': 1, 'reg_lambda': 2.}
    assert all(effective_parameters(pipeline.frozen_spec())[k] == v for k, v in expected.items())
    assert all(pipeline.pipeline.models['answer'].get_params()[k] == v for k, v in expected.items())
    before = pipeline.predict(X.iloc[:10], c).values
    pipeline.refit(X, y, c)
    assert all(pipeline.pipeline.models['answer'].get_params()[k] == v for k, v in expected.items())
    assert pipeline.predict(X.iloc[:10], c).values == before
