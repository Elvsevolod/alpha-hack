"""A supported AutoGluon probability pipeline followed by fit-only OOF offsets."""
import time

from .api import prediction_bundle
from .decision import decide, fit_decision
from .research_evaluation import oof_predictions


class AutoMLDecisionPipeline:
    def __init__(self, spec, task):
        self.spec, self.task = spec, task

    def fit(self, X, y, context):
        from .search import SearchPipeline
        started = time.monotonic()
        base = self.spec.model_copy(update={"prediction_policy": "default", "postprocess_fit_policy": "none",
            "class_offsets": None, "learned_decision": {}})
        self.state = dict(self.spec.learned_decision)
        if not self.state:
            report = oof_predictions(base, self.task, X, y, {**context, "budget_seconds": context["budget_seconds"] * .45})
            positions = {str(key): i for i, key in enumerate(X.index)}
            actual = y.iloc[[positions[key] for key in report["row_keys"]]]
            target = self.task.targets[0]
            self.state[target] = fit_decision(self.task, actual, report["values"][target], report["row_keys"], report["folds_hash"], self.spec.decision_method)
        auxiliary = time.monotonic() - started
        self.pipeline = SearchPipeline(base, self.task)
        summary = self.pipeline.fit(X, y, {**context, "budget_seconds": context["budget_seconds"] - auxiliary})
        summary.budget["auxiliary_fit_actual_seconds"] = summary.budget.get("auxiliary_fit_actual_seconds", 0) + auxiliary
        summary.seconds = time.monotonic() - started
        self.summary = summary
        return summary

    def predict(self, X, context):
        bundle = self.pipeline.predict(X, context)
        probabilities = bundle.calibrated_probabilities or bundle.raw_probabilities or bundle.values
        if bundle.classes != self.task.classes:
            raise ValueError("AutoML probability class order changed")
        labels = {target: decide(probabilities[target], self.task.classes[target], self.state.get(target)).tolist() for target in self.task.targets}
        return prediction_bundle(labels, X, context, "labels", self.task.classes,
            raw_probabilities=probabilities, decision_values=labels)

    def refit(self, X, y, context):
        summary = self.pipeline.refit(X, y, context)
        self.summary = summary
        return summary

    def frozen_spec(self):
        return self.pipeline.frozen_spec().model_copy(update={"prediction_policy": "class_offsets", "postprocess_fit_policy": "pooled_oof",
            "learned_decision": self.state, "class_offsets": self.state[self.task.targets[0]]["offsets"]})
