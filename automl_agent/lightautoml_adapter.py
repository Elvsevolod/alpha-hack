"""CPU-only LightAutoML 0.4.2 with one frozen, replayable GBM family."""
import time
import numpy as np
import pandas as pd

from .api import prediction_bundle
from .contracts import TrainingSummary
from .evaluation import score_predictions
from .validation import inner_split


class LightAutoMLPipeline:
    def __init__(self, spec, task):
        self.spec, self.task, self.models, self.selected = spec, task, {}, {}

    def _fit_target(self, X, y, target, context, fit, valid, parameters=None):
        from lightautoml.automl.presets.tabular_presets import TabularAutoML
        from lightautoml.tasks import Task
        classes = sorted(y[target].unique().tolist()) if self.task.classes else []
        if self.task.classes and len(classes) < 2:
            raise ValueError('LightAutoML requires two observed classes; use a constant pipeline')
        name = 'binary' if len(classes) == 2 else 'multiclass' if classes else 'reg'
        local_task = self.task.model_copy(update={'targets': [target], 'classes': {target: classes} if classes else {},
            'metric': self.task.metric.model_copy(update={'aggregation': 'uniform', 'target_weights': None})})
        def metric(actual, prediction, sample_weight=None):
            if classes:
                prediction = np.asarray(prediction)
                if prediction.ndim == 1 or prediction.shape[1] == 1:
                    p = prediction.reshape(-1)
                    prediction = np.column_stack([1-p, p])
                truth = np.asarray(classes)[np.asarray(actual, dtype=int)]
                return score_predictions(local_task, pd.DataFrame({target: truth}), {target: prediction}, 'probabilities')
            return score_predictions(local_task, pd.DataFrame({target: actual}), {target: np.asarray(prediction).reshape(-1)}, 'values')
        task = Task(name, metric=metric, greater_is_better=self.task.metric.maximize)
        params = parameters or {'num_trees': 300, 'early_stopping_rounds': 30,
            'num_threads': context['cpu'], 'seed': self.spec.seed, 'verbosity': -1}
        # fit() and refit() each call this helper; split the phase budget so the
        # candidate never consumes twice its granted window.
        model = TabularAutoML(task=task, timeout=max(1, context['budget_seconds'] / (2 * len(self.task.targets))),
            cpu_limit=context['cpu'], memory_limit=max(1, context['memory_mb'] / 1024 * .7), gpu_ids='',
            general_params={'use_algos': [['lgb']], 'nested_cv': False},
            reader_params={'cv': 2, 'random_state': self.spec.seed, 'advanced_roles': False},
            selection_params={'mode': 0}, lgb_params={'default_params': params, 'freeze_defaults': True})
        data = X[self.task.feature_columns].copy().reset_index(drop=True)
        data[target] = y[target].map({value: i for i, value in enumerate(classes)}).to_numpy() if classes else y[target].to_numpy()
        roles = {'target': target}
        categorical = [column for column in self.task.categorical_columns if column in data]
        if categorical:
            roles['category'] = categorical
        from joblib import parallel_backend
        with parallel_backend('threading', n_jobs=context['cpu']):
            model.fit_predict(data, roles=roles, cv_iter=[(np.asarray(fit), np.asarray(valid))], verbose=0)
        algorithms = [algo for level in model.levels for pipe in level for algo in pipe.ml_algos]
        if len(algorithms) != 1 or len(algorithms[0].models) != 1:
            raise ValueError('LightAutoML produced a configuration outside the frozen single-family adapter')
        algo = algorithms[0]
        frozen = {**algo.params, 'num_trees': algo.models[0].current_iteration(), 'early_stopping_rounds': 0}
        self.models[target] = (model, classes)
        self.selected[target] = {'parameters': frozen, 'engine': 'lightautoml',
            'selected_fit_rows': len(fit), 'refit_search': False}

    def fit(self, X, y, context):
        if self.spec.learners and self.spec.learners != ['lgb']:
            raise ValueError('Verified LightAutoML adapter uses learners=[lgb]; no tuned/nested learners')
        started = time.monotonic()
        a, b = inner_split(X, y, self.task, context['strategy'], self.spec.seed)
        for target in self.task.targets:
            self._fit_target(X, y, target, context, a, b)
        search_seconds = time.monotonic() - started
        summary = self.refit(X, y, context)
        summary.seconds = time.monotonic() - started
        summary.inner_scores = {'scope': 'fit_inner_holdout', 'fit_keys': X.index[a].tolist(), 'heldout_keys': X.index[b].tolist()}
        summary.budget['search_seconds'] = search_seconds
        self.summary = summary
        return summary

    def refit(self, X, y, context):
        started = time.monotonic()
        parameters = {target: dict(value['parameters']) for target, value in self.selected.items()}
        for target in self.task.targets:
            # A single fixed learner, fixed iteration count, no HPO, no weighted
            # blender, no stopping decision: every permitted row trains the model.
            self._fit_target(X, y, target, context, np.arange(len(X)), np.arange(len(X)), parameters[target])
        summary = TrainingSummary(rows=len(X), seconds=time.monotonic()-started, selected_parameters=self.selected,
            budget={'selected_fit_rows': len(X), 'selected_fit_actual_seconds': time.monotonic()-started,
                'refit_policy': 'frozen_single_family_fixed_iterations_all_permitted_rows'})
        self.summary = summary
        return summary

    def predict(self, X, context):
        output = {}
        for target, (model, classes) in self.models.items():
            from joblib import parallel_backend
            with parallel_backend('threading', n_jobs=context['cpu']):
                raw = model.predict(X[self.task.feature_columns].copy().reset_index(drop=True), n_jobs=context['cpu']).data
            if classes:
                if raw.shape[1] == 1:
                    raw = np.column_stack([1-raw[:, 0], raw[:, 0]])
                aligned = np.zeros((len(X), len(self.task.classes[target])))
                for i, label in enumerate(classes):
                    aligned[:, self.task.classes[target].index(label)] = raw[:, i]
                output[target] = aligned
            else:
                output[target] = raw.reshape(-1)
        return prediction_bundle(output, X, context, 'probabilities' if self.task.classes else 'values', self.task.classes,
            raw_probabilities={key: value.tolist() for key, value in output.items()} if self.task.classes else {})

    def frozen_spec(self):
        return self.spec
