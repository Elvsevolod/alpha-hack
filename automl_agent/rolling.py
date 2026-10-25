"""Search feedback on chronological transitions, followed by a development fit."""
import time

import numpy as np

from .api import prediction_bundle
from .evaluation import align_predictions


class RollingPipeline:
    def __init__(self, spec, task):
        self.spec, self.task = spec, task

    def fit(self, X, y, context):
        from .pipelines import build_pipeline
        started = time.monotonic()
        self.predictions, auxiliary, boundaries = {}, 0, []
        parts = context["rolling_folds"]
        if self.spec.engine == "optuna":
            from .search import SearchPipeline
            self.pipeline = SearchPipeline(self.spec, self.task)
            summary = self.pipeline.fit(X, y, context)
            report = self.pipeline.search_predictions
            self.kind = report["prediction_type"]
            for target in self.task.targets:
                self.predictions[target] = dict(zip(map(str, report["row_keys"]), report["values"][target]))
        else:
            if self.spec.engine not in {"direct"}:
                raise ValueError("Rolling adapter supports fixed direct recipes and bounded Optuna only")
            for index, part in enumerate(parts):
                a, b = part["fit"], part["validation"]
                remaining = context["budget_seconds"] - (time.monotonic() - started)
                local = {**context, "rolling_folds": None, "split_id": f"rolling_{index}",
                    "budget_seconds": remaining / (len(parts) - index + 1), "fit_row_keys": X.index[a].tolist()}
                model = build_pipeline(self.spec, self.task, local)
                model.fit(X.iloc[a], y.iloc[a], local)
                bundle = model.predict(X.iloc[b], local)
                self.kind = bundle.prediction_type
                values = align_predictions(bundle, X.index[b].tolist(), self.task, context["protocol_hash"], local["split_id"])
                for target in self.task.targets:
                    self.predictions.setdefault(target, {}).update(zip(map(str, X.index[b]), np.asarray(values[target]).tolist()))
                auxiliary += 1 + model.summary.auxiliary_fits
                boundaries.append({"fit_keys": X.index[a].tolist(), "evaluation_keys": X.index[b].tolist()})
            self.pipeline = build_pipeline(self.spec, self.task, {**context, "rolling_folds": None})
            summary = self.pipeline.fit(X, y, {**context, "rolling_folds": None,
                "budget_seconds": context["budget_seconds"] - (time.monotonic() - started)})
        summary.auxiliary_fits += auxiliary
        summary.inner_scores = {**summary.inner_scores, "rolling": {"folds": boundaries or parts,
            "prediction_role": "search_validation", "strict_oof_for_meta": False, "control_labels_available": False}}
        summary.seconds = time.monotonic() - started
        self.summary = summary
        return summary

    def predict(self, X, context):
        if context["split_id"] == "validation":
            values = {target: [self.predictions[target][str(key)] for key in X.index] for target in self.task.targets}
            return prediction_bundle(values, X, context, self.kind, self.task.classes)
        return self.pipeline.predict(X, context)

    def frozen_spec(self):
        return self.pipeline.frozen_spec()
