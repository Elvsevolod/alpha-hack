"""Phase-aware forecasts from uncapped performance samples, never smoke fits."""
import math
from .util import digest


def family(spec):
    identity = (spec.engine, spec.model, spec.representation, spec.imputation.hash,
        tuple((op.name, digest(op.parameters)) for op in spec.transformations),
        spec.prediction_policy, spec.custom_semantic_hash or spec.custom_hash,
        spec.autogluon_profile, tuple(spec.learners), bool(spec.early_stopping_rounds),
        spec.supervised_transform, spec.shift_strategy, tuple(spec.include_features or []), tuple(spec.exclude_features),
        spec.neighbor_scale, spec.postprocess_fit_policy, spec.log_inverse, tuple(spec.checkpoint_counts),
        spec.refit_iteration_policy, digest(spec.autogluon_configurations))
    return identity + (digest(spec.hyperparameters),) if spec.model=="torch_tabular" else identity


def complexity(spec):
    p = spec.hyperparameters
    if spec.engine == 'autogluon':
        return float(spec.bag_folds * (2 if spec.autogluon_profile == 'stacking' else 1)) if spec.autogluon_profile != 'selection' else 1.
    if spec.model == 'torch_tabular':
        return float(p.get('epochs',50)*p.get('depth',2)*p.get('width',128)**2*p.get('heads',1))
    defaults = {'random_forest':100,'extra_trees':100,'hist_gradient_boosting':100,'catboost':200,'mlp':100,'sgd':1000}
    iterations = p.get('n_estimators', p.get('iterations', p.get('max_iter', defaults.get(spec.model,150))))
    if spec.model in {'ridge', 'elastic_net', 'logistic', 'constant'}:
        return 1.
    depth = p.get('max_depth', p.get('depth', 6)) or 12
    if depth <= 0:
        depth = 12
    return max(1., float(iterations)) * max(1., min(32., float(depth)))


def selected_seconds(summary):
    budget = summary.get('budget') or {}
    if budget.get('selected_fit_actual_seconds') is not None:
        return float(budget['selected_fit_actual_seconds'])
    targets = budget.get('targets') or {}
    if targets and all('selected_fit_actual_seconds' in item for item in targets.values()):
        return sum(item['selected_fit_actual_seconds'] for item in targets.values())
    return float((summary.get('execution_phases') or {}).get('fit_seconds', summary.get('seconds', .2)))


def performance_curve(samples):
    points = sorted([{'rows': max(1, int(s['training_summary']['rows'])),
        'fit_seconds': selected_seconds(s['training_summary'])} for s in samples], key=lambda x:x['rows'])
    if len(points) >= 2 and points[-1]['rows'] > points[0]['rows']:
        slope = max(0., (points[-1]['fit_seconds'] - points[0]['fit_seconds']) / (points[-1]['rows'] - points[0]['rows']))
        fixed = max(0., points[0]['fit_seconds'] - slope * points[0]['rows'])
    else:
        fixed, slope = 0., points[0]['fit_seconds'] / points[0]['rows']
    return {'version': 1, 'basis': 'uncapped_performance_samples', 'samples': points,
        'fixed_fit_seconds': fixed, 'seconds_per_row': slope, 'uncertainty_multiplier': 1.5 if len(points) > 1 else 1.75}


def iteration_multiplier(pipeline, rows, observed_rows):
    """The actual frozen tree policy, independent of the uncertainty margin."""
    if pipeline is None:
        return 1.
    from .contracts import PipelineSpec
    from .pipelines import refit_iterations
    spec = PipelineSpec.model_validate(pipeline) if isinstance(pipeline, dict) else pipeline
    schedule = list(spec.iteration_schedule.values()) or [spec.frozen_iterations]
    return max([refit_iterations(spec, count, rows, 'refit') /
        max(1, refit_iterations(spec, count, observed_rows, 'refit')) for count in schedule if count] or [1.])


def phase_forecast(summary, seconds, rows, default_rows, ensemble=False, pipeline=None):
    seconds = max(0., seconds - sum(summary.get(name, 0) for name in
        ('research_stability_seconds', 'research_oof_seconds', 'research_review_seconds', 'research_comparison_seconds')))
    phases = summary.get('execution_phases') or {}
    budget = summary.get('budget') or {}
    training_rows = max(1, budget.get('selected_fit_rows', summary.get('rows', default_rows)))
    curve = summary.get('cost_curve')
    fit = selected_seconds(summary)
    repeatable = float(budget.get('repeatable_auxiliary_fit_seconds', 0))
    initialization=float(budget.get('worker_initialization_seconds',0))
    iteration_recipe = pipeline
    if pipeline is not None and budget.get('iteration_schedule'):
        from .contracts import PipelineSpec
        iteration_recipe = PipelineSpec.model_validate(pipeline) if isinstance(pipeline, dict) else pipeline
        if not iteration_recipe.iteration_scale_rows:
            iteration_recipe = iteration_recipe.model_copy(update={'iteration_scale_rows':training_rows,
                'iteration_schedule':{key:value for key,value in budget['iteration_schedule'].items() if value}})
    iterations = iteration_multiplier(iteration_recipe, rows, training_rows)
    multiplier = 1.25
    selected = budget.get('selected_fit_actual_seconds') is not None or bool(budget.get('targets'))
    if not curve and not all(name in phases for name in ('fit_seconds','load_and_construct_seconds','save_load_seconds','predict_and_verify_seconds')):
        base = max(.1 if ensemble else .2, seconds)
        if selected:
            overhead = max(.2, seconds-summary.get('seconds',0)-summary.get('automatic_pilot_seconds',0))
            base = max(.2, fit * iterations + repeatable + overhead)
        return (base * max(1., rows/training_rows) + (initialization if selected else 0.)) * 1.25 + .2
    if curve:
        fit = curve['fixed_fit_seconds'] + curve['seconds_per_row'] * rows
        multiplier = curve['uncertainty_multiplier']
    else:
        fit *= max(1., rows / training_rows)
    fit *= iterations
    fit += repeatable * max(1., rows / training_rows)
    # Serialization and interpreter startup are fixed; prediction uses output rows,
    # not fit rows. Full test size is recorded independently by the worker.
    save_load = float(phases.get('save_load_seconds', 0))
    predict_total = float(phases.get('predict_and_verify_seconds', 0))
    predict = float(phases.get('prediction_seconds', predict_total))
    prediction_fixed = max(0., predict_total - predict)
    observed_predict = max(1, phases.get('prediction_rows', training_rows))
    future_predict = phases.get('forecast_prediction_rows', observed_predict)
    predict *= max(1., future_predict / observed_predict)
    loading = float(phases.get('load_and_construct_seconds', 0)) + float(phases.get('controller_staging_seconds', 0))
    actual_fit = float(summary.get('seconds', phases.get('fit_seconds', 0)) if selected else phases.get('fit_seconds', summary.get('seconds', 0)))
    startup = max(.2, seconds - actual_fit - save_load - float(phases.get('predict_and_verify_seconds', 0)) - loading - summary.get('automatic_pilot_seconds', 0))
    result = (max(.1 if ensemble else .2, fit) + save_load + predict + prediction_fixed + loading + startup + initialization) * multiplier + .2
    if not math.isfinite(result):
        raise ValueError('Nonfinite completion forecast')
    return result
