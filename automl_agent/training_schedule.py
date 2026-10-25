"""Native learning curves on an explicitly supplied fit-inner holdout."""
import numpy as np
import pandas as pd

from .evaluation import score_predictions


ITERATION_PARAMETER = {"lightgbm": "n_estimators", "xgboost": "n_estimators", "catboost": "iterations", "hist_gradient_boosting": "max_iter"}


def fit_with_schedule(model, spec, task, target, matrix, values, evaluation, context, fit_kwargs, class_order):
    valid, actual = evaluation
    curve = []
    rounds = spec.early_stopping_rounds or 10
    local_task = task.model_copy(update={"targets": [target],
        "classes": {target: task.classes[target]} if task.classes else {},
        "metric": task.metric.model_copy(update={"aggregation": "uniform", "target_weights": None})})
    trial = context.get("optuna_trial")

    def report(value, iteration):
        if iteration % spec.evaluation_interval == 0:
            curve.append({"iteration": int(iteration), "primary_metric": float(value)})
        if trial is not None:
            trial.report(float(value), step=int(iteration))
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    def primary(actual_encoded, prediction):
        if task.classes:
            truth = np.asarray(class_order)[np.asarray(actual_encoded, dtype=int)]
            prediction = np.asarray(prediction)
            if prediction.ndim == 1:
                prediction = np.column_stack([1 - prediction, prediction])
            aligned = np.zeros((len(prediction), len(task.classes[target])))
            for i, label in enumerate(class_order):
                aligned[:, task.classes[target].index(label)] = prediction[:, i]
            data, kind = {target: aligned}, "probabilities"
        else:
            truth = np.asarray(actual_encoded)
            prediction = np.asarray(prediction)
            if spec.target_transform == "log1p":
                truth, prediction = np.expm1(truth), np.expm1(prediction)
            data, kind = {target: prediction}, "values"
        return score_predictions(local_task, pd.DataFrame({target: truth}), data, kind)

    if spec.model == "lightgbm":
        import lightgbm as lgb
        evaluation_record = {}
        model.set_params(metric="None")
        def metric(y, prediction):
            return "primary", primary(y, prediction), task.metric.maximize
        def observe(env):
            report(next(item[2] for item in env.evaluation_result_list if item[1] == "primary"), env.iteration)
        model.fit(matrix, values, eval_set=[(valid, actual)], eval_metric=metric,
            callbacks=[lgb.record_evaluation(evaluation_record), observe,
                lgb.early_stopping(rounds, first_metric_only=True, verbose=False)], **fit_kwargs)
        iterations = max(1, model.best_iteration_)
    elif spec.model == "xgboost":
        from xgboost.callback import EarlyStopping, TrainingCallback
        def primary_metric(y, prediction):
            return primary(y, prediction)
        primary_metric.__name__ = "primary"
        class Observe(TrainingCallback):
            def after_iteration(self, booster, epoch, history):
                report(history["validation_0"]["primary"][-1], epoch)
                return False
        model.set_params(eval_metric=primary_metric, callbacks=[Observe(), EarlyStopping(rounds,
            metric_name="primary", data_name="validation_0", maximize=task.metric.maximize, save_best=True)])
        model.fit(matrix, values, eval_set=[(valid, actual)], verbose=False, **fit_kwargs)
        iterations = max(1, model.best_iteration + 1)
    elif spec.model == "catboost":
        class PrimaryMetric:
            def is_max_optimal(self):
                return task.metric.maximize
            def get_final_error(self, error, weight):
                return error
            def evaluate(self, approxes, target_values, weight):
                prediction = np.asarray(approxes).T
                if task.classes:
                    if prediction.shape[1] == 1:
                        p = 1/(1+np.exp(-np.clip(prediction[:, 0], -700, 700)))
                        prediction = np.column_stack([1-p, p])
                    else:
                        prediction -= prediction.max(axis=1, keepdims=True)
                        prediction = np.exp(prediction)
                        prediction /= prediction.sum(axis=1, keepdims=True)
                else:
                    prediction = prediction.reshape(-1)
                return primary(np.asarray(target_values), prediction), 1.
        native = "PrimaryMetric"
        model.set_params(eval_metric=PrimaryMetric())
        pruned = []
        class Observe:
            def after_iteration(self, info):
                name = next(key for key in info.metrics["validation"] if key.split(":")[0] == native.split(":")[0])
                try:
                    report(info.metrics["validation"][name][-1], info.iteration - 1)
                except Exception as error:
                    import optuna
                    if not isinstance(error, optuna.TrialPruned):
                        raise
                    pruned.append(True)
                    return False
                return True
        model.fit(matrix, values, eval_set=(valid, actual), early_stopping_rounds=rounds,
            use_best_model=True, callbacks=[Observe()], **fit_kwargs)
        if pruned:
            import optuna
            raise optuna.TrialPruned()
        iterations = max(1, model.get_best_iteration() + 1)
    elif spec.model == "hist_gradient_boosting":
        maximum = model.max_iter
        interval = min(spec.evaluation_interval, maximum)
        schedule = sorted(set([*range(interval, maximum + 1, interval), maximum]))
        best, iterations, stale = None, interval, 0
        model.set_params(warm_start=True, early_stopping=False)
        for iteration in schedule:
            model.set_params(max_iter=iteration)
            model.fit(matrix, values, **fit_kwargs)
            prediction = model.predict_proba(valid) if task.classes else model.predict(valid)
            encoded = np.asarray([class_order.index(value) for value in actual]) if task.classes else actual
            value = primary(encoded, prediction)
            report(value, iteration - 1)
            if best is None or (value > best if task.metric.maximize else value < best):
                best, iterations, stale = value, iteration, 0
            else:
                stale += interval
                if stale >= rounds:
                    break
    else:
        raise ValueError("Learning-curve schedules require a supported boosting family")
    return {"iterations": int(iterations), "curve": curve, "scope": "fit_inner_holdout",
        "primary_metric": task.metric.name, "is_quality_guarantee": False,
        "stopping_prediction_policy": "native_argmax" if task.classes else "original_scale_values",
        "final_prediction_policy": spec.prediction_policy,
        "surrogate_differs_from_final_decision": bool(task.classes and spec.prediction_policy != "default")}
