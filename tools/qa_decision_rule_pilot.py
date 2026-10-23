"""Prespecified fit-only Macro-F1 experiments; never imported by frozen A/B."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automl_agent.contracts import MetricSpec, OutputSpec, PipelineSpec, SelectionPolicy, TaskSpec, Transformation
from automl_agent.evaluation import score_predictions
from automl_agent.pipelines import DirectPipeline
from automl_agent.util import digest
from automl_agent.validation import inner_split, stratified_parts


PROTOCOL = {
    'dataset_seeds': [17, 42, 73], 'training_seed': 42,
    'families': {'one_cluster_per_class': 1, 'two_clusters_per_class': 2},
    'rows': 1500, 'features': 10, 'class_weights_in_data': [.70, .22, .08],
    'variants': ['baseline', 'balanced', 'random_over', 'multiclass_rule'],
    'rule': 'one coordinate sweep, canonical class order, weights0.5/1/2; fit inner holdout only',
    'ties': {'current': .001, 'stricter': .0001},
    'selection': 'Freeze all four experimental recipes/costs before opening control; exact best then tied set; baseline priority',
    'default_change_gate': 'Synthetic cases alone cannot establish transfer; retain defaults and report exact per-case effects',
}


def context(task, split):
    return {'cpu': 1, 'memory_mb': 512, 'strategy': 'random', 'task': task.model_dump(mode='json'),
            'split_id': split, 'protocol_hash': 'section4_decision_pilot'}


def model_spec(variant):
    return PipelineSpec(model='logistic', seed=42, transformations=[Transformation(name='standard_scale')],
        hyperparameters={'C': 1., 'max_iter': 300, **({'class_weight': 'balanced'} if variant == 'balanced' else {})},
        resampling='random_over' if variant == 'random_over' else 'none')


def fit_pipeline(X, y, task, variant):
    model = DirectPipeline(model_spec(variant), task)
    sampling = []
    original = RandomOverSampler.fit_resample

    def tracked(sampler, matrix, values, **kwargs):
        assert matrix.shape[0] == len(X) and np.array_equal(values, y['y'].to_numpy())
        result = original(sampler, matrix, values, **kwargs)
        positions = sampler.sample_indices_
        assert all(0 <= index < len(X) for index in positions)
        sampling.append({'input_row_hash': digest(list(X.index)), 'input_rows': len(X),
                         'resampled_rows': len(positions), 'only_allowed_training_positions': True})
        return result

    started = time.monotonic()
    with patch.object(RandomOverSampler, 'fit_resample', tracked):
        model.fit(X, y, context(task, 'fit'))
    assert model.training_rows == list(X.index)
    return model, time.monotonic() - started, sampling


def probabilities(model, X, task, split):
    return np.asarray(model.predict(X, context(task, split)).values['y'])


def decide(probability, classes, weights):
    probability = np.asarray(probability, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != len(classes) or weights.shape != (len(classes),):
        raise ValueError('Decision-rule class/probability shape mismatch')
    if not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(axis=1), 1):
        raise ValueError('Invalid probability distribution')
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid decision weights')
    return np.asarray(classes)[np.argmax(probability * weights, axis=1)]


def metric(task, actual, predicted):
    return score_predictions(task, actual, {'y': predicted}, 'labels')


def learn_rule(X_fit, y_fit, task):
    a, b = inner_split(X_fit, y_fit, task, 'random', 43)
    classes = task.classes['y']
    if set(y_fit.iloc[a]['y']) != set(classes) or set(y_fit.iloc[b]['y']) != set(classes):
        raise ValueError('Inner decision-rule split lacks class coverage')
    model, _, _ = fit_pipeline(X_fit.iloc[a], y_fit.iloc[a], task, 'baseline')
    probability = probabilities(model, X_fit.iloc[b], task, 'inner')
    weights = np.ones(len(classes))
    trials = []
    for index in range(len(classes)):
        choices = []
        for value in (.5, 1., 2.):
            candidate = weights.copy()
            candidate[index] = value
            score = metric(task, y_fit.iloc[b], decide(probability, classes, candidate))
            trials.append({'weights': candidate.tolist(), 'inner_macro_f1': score})
            choices.append((-score, float(np.abs(np.log(candidate)).sum()), tuple(candidate)))
        weights = np.asarray(min(choices)[2])
    assert set(X_fit.iloc[a].index).isdisjoint(X_fit.iloc[b].index)
    return tuple(weights), {'scope': 'inner_fit_only', 'inner_model_rows': list(X_fit.iloc[a].index),
        'inner_rule_rows': list(X_fit.iloc[b].index), 'trials': trials}


def choose(records, delta):
    policy = SelectionPolicy(maximize=True, min_delta_abs=delta)
    best = max(row['control_macro_f1'] for row in records)
    tied = [row for row in records if policy.tied(row['control_macro_f1'], best)]
    baseline = next((row for row in tied if row['variant'] == 'baseline'), None)
    return (baseline or min(tied, key=lambda row: (row['frozen_cost_seconds'], row['created_index'], row['variant'])))['variant']


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'PREREGISTRATION.json').write_text(json.dumps(PROTOCOL, indent=2) + '\n')
    cases = []
    for family, clusters in PROTOCOL['families'].items():
        for seed in PROTOCOL['dataset_seeds']:
            matrix, values = make_classification(n_samples=1500, n_features=10, n_informative=6,
                n_redundant=2, n_classes=3, n_clusters_per_class=clusters,
                weights=[.70, .22, .08], class_sep=.8, flip_y=.025, random_state=seed)
            X = pd.DataFrame(matrix, columns=[f'x{i}' for i in range(10)], index=[f'row:{i}' for i in range(1500)])
            y = pd.DataFrame({'y': values}, index=X.index)
            task = TaskSpec(kind='classification', targets=['y'], id_columns=['id'], feature_columns=list(X),
                classes={'y': [0, 1, 2]}, metric=MetricSpec(name='f1', average='macro'),
                output=OutputSpec(columns=['id', 'y'], prediction_type='labels'), source_hash=f'{family}:{seed}')
            fit, validation, control = stratified_parts(y.reset_index(drop=True), 42)
            weights, rule_audit = learn_rule(X.iloc[fit], y.iloc[fit], task)
            assert set(rule_audit['inner_model_rows'] + rule_audit['inner_rule_rows']) <= set(X.iloc[fit].index)
            assert set(X.iloc[fit].index).isdisjoint(X.iloc[control].index)
            frozen, records = [], []
            for index, variant in enumerate(PROTOCOL['variants']):
                model, seconds, sampling = fit_pipeline(X.iloc[fit], y.iloc[fit], task, variant)
                selected_weights = weights if variant == 'multiclass_rule' else (1., 1., 1.)
                predicted = decide(probabilities(model, X.iloc[validation], task, 'validation'), task.classes['y'], selected_weights)
                frozen.append({'variant': variant, 'created_index': index, 'weights': selected_weights,
                    'pipeline': model.spec.model_dump(mode='json'), 'frozen_cost_seconds': seconds,
                    'validation_macro_f1': metric(task, y.iloc[validation], predicted), 'resampling': sampling})
            frozen_hash = digest(frozen)
            allowed = sorted(fit + validation)
            for recipe in frozen:
                model, _, sampling = fit_pipeline(X.iloc[allowed], y.iloc[allowed], task, recipe['variant'])
                def forbidden_fit(*args, **kwargs):
                    raise AssertionError('predict retrained')
                model.models['y'].fit = forbidden_fit
                predicted = decide(probabilities(model, X.iloc[control], task, 'control'), task.classes['y'], recipe['weights'])
                records.append({**recipe, 'control_macro_f1': metric(task, y.iloc[control], predicted),
                                'control_preparation_sampling': sampling})
            assert digest(frozen) == frozen_hash
            policies = {key: choose(records, delta) for key, delta in PROTOCOL['ties'].items()}
            by_name = {row['variant']: row for row in records}
            gap = by_name[policies['stricter']]['control_macro_f1'] - by_name[policies['current']]['control_macro_f1']
            row = {'family': family, 'dataset_seed': seed, 'rule_fit': rule_audit, 'learned_weights': weights,
                'frozen_recipes_hash': frozen_hash, 'variants': records, 'selected': policies,
                'stricter_minus_current_control_macro_f1': gap,
                'split_rows_hash': digest({'fit': fit, 'validation': validation, 'control': control})}
            cases.append(row)
            (output / 'CASES.json').write_text(json.dumps(cases, indent=2) + '\n')
    result = {'passed': True, 'organizer_requests': 0, 'cases': cases,
        'default_policy_decision': 'retain_current_tolerance_and_optional_class_weight_resampling',
        'multiclass_rule_decision': 'staged_only; report per-case gains/losses, no unconditional activation',
        'scope': 'Six synthetic mechanism experiments; four prespecified variants, not production three-finalist acceptance or private-quality evidence',
        'runtime_archives_modified': False}
    (output / 'REPORT.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(json.dumps({'passed': result['passed'], 'cases': len(result['cases']),
                      'policy_changes': sum(row['selected']['current'] != row['selected']['stricter'] for row in result['cases'])}))
