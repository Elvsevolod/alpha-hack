"""Rebuild only a declared label space around a frozen native AG learner."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd


class AutoGluonClassRefit:
    def __init__(self, feature_generator, native_model, cleaner, cpu):
        self.feature_generator = feature_generator
        self.native_model = native_model
        self.cleaner = cleaner
        self.cpu = cpu

    @classmethod
    def build(cls, source, feature_generator, X, labels, task, target, context, path):
        from autogluon.core.data import LabelCleaner
        if not set(labels) <= set(task.classes[target]):
            raise ValueError('Refit labels are outside the confirmed class space; no rows were dropped or relabeled')
        started = time.monotonic()
        problem = 'binary' if labels.nunique() == 2 else 'multiclass'
        cleaner = LabelCleaner.construct(problem_type=problem, y=labels,
            positive_class=task.metric.positive_label if problem == 'binary' else None)
        encoded = cleaner.transform(labels)
        if encoded.isna().any():
            raise ValueError('Refit class encoding is incomplete')
        if type(source).__name__ not in {'LGBModel', 'CatBoostModel', 'XGBoostModel'}:
            raise ValueError('Class-space refit requires a verified native CPU tree learner')
        key = {'LGBModel': 'num_boost_round', 'CatBoostModel': 'iterations',
            'XGBoostModel': 'n_estimators'}[type(source).__name__]
        fixed = source.params_trained.get(key)
        if fixed is None:
            raise ValueError('Class-space refit requires measured frozen iterations')
        parameters = copy.deepcopy(source.get_params())
        parameters['hyperparameters'].update(source.params_trained)
        parameters.update(path=str(Path(path).resolve()), name='native_class_refit',
            problem_type=cleaner.problem_type_transform)
        model = type(source)(**parameters)
        generator = copy.deepcopy(feature_generator)
        features = generator.transform(X)
        if not labels.index.equals(X.index) or not features.index.equals(X.index):
            raise ValueError('Class-space refit changed row identity or order')
        remaining = context['budget_seconds'] - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('No time remains for the frozen class-space refit')
        # No validation data, search, early-stopping decision or feature refit.
        model.fit(X=features, y=encoded, num_cpus=context['cpu'], num_gpus=0,
            time_limit=remaining, verbosity=0)
        if key == 'num_boost_round':
            actual = model.model.current_iteration()
        elif key == 'iterations':
            actual = model.model.tree_count_
        else:
            actual = model.model.get_booster().num_boosted_rounds()
        if actual != fixed:
            raise ValueError('Class-space refit did not complete the frozen iteration schedule')
        # XGBoostModel omits params_trained when fit has no validation data.
        model.params_trained[key] = int(actual)
        return cls(generator, model, cleaner, context['cpu'])

    def predict_proba(self, X, as_pandas=True, as_multiclass=True, num_cpus=None):
        features = self.feature_generator.transform(X)
        if not features.index.equals(X.index):
            raise ValueError('Class-space prediction changed row identity or order')
        cpu = self.cpu if num_cpus is None else min(self.cpu, num_cpus)
        raw = np.asarray(self.native_model.predict_proba(features, num_cpus=cpu))
        if raw.ndim == 1:
            raw = np.column_stack([1 - raw, raw])
        return pd.DataFrame(raw, index=X.index, columns=self.cleaner.ordered_class_labels)
