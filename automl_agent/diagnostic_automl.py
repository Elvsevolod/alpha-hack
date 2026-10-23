"""Observation-only callbacks for the exact pinned AutoML releases."""
from contextlib import contextmanager
import time
import traceback

from .diagnostics import trial_event


@contextmanager
def flaml_trials(context):
    from flaml.automl.state import AutoMLState
    original = AutoMLState._compute_with_config_base
    count = 0
    def observe(config_w_resource, state, estimator, is_report=True):
        nonlocal count
        count += 1
        identifier = f'flaml_{count}'
        started = time.monotonic()
        trial_event(context, 'trial_started', trial_id=identifier, engine='flaml', parameters=config_w_resource, estimator=estimator)
        try:
            result = original(config_w_resource, state, estimator, is_report)
            value = result.get('val_loss')
            trial_event(context, 'trial_finished', trial_id=identifier, engine='flaml', metric=value,
                status='completed' if value is not None and value != float('inf') else 'failed', seconds=time.monotonic()-started)
            return result
        except BaseException:
            trial_event(context, 'trial_finished', trial_id=identifier, engine='flaml', status='failed',
                seconds=time.monotonic()-started, traceback=traceback.format_exc())
            raise
    AutoMLState._compute_with_config_base = staticmethod(observe)
    try:
        yield
    finally:
        AutoMLState._compute_with_config_base = staticmethod(original)


@contextmanager
def autogluon_trials(context):
    from autogluon.tabular.trainer import AutoTrainer
    original = AutoTrainer._train_single_full
    inherited = '_train_single_full' not in AutoTrainer.__dict__
    def observe(trainer, *args, **kwargs):
        model = kwargs.get('model') or args[2]
        started = time.monotonic()
        trial_event(context, 'trial_started', trial_id=model.name, engine='autogluon',
                    parameters=model.params, time_limit=kwargs.get('time_limit'))
        try:
            names = original(trainer, *args, **kwargs)
            trial_event(context, 'trial_finished', trial_id=model.name, engine='autogluon',
                status='completed' if names else 'failed', models=names,
                metrics={name:trainer.get_model_attribute(name, 'val_score') for name in names},
                seconds=time.monotonic()-started)
            return names
        except BaseException:
            trial_event(context, 'trial_finished', trial_id=model.name, engine='autogluon',
                status='failed', seconds=time.monotonic()-started, traceback=traceback.format_exc())
            raise
    AutoTrainer._train_single_full = observe
    try:
        yield
    finally:
        if inherited:
            del AutoTrainer._train_single_full
        else:
            AutoTrainer._train_single_full = original
