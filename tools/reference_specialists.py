"""Development pattern specialists, reusing the project's estimator factory."""
from __future__ import annotations

import numpy as np


class PatternLGBM:
    """Hard routing by an observable missingness indicator, with global fallback.

    This small pattern-submodel experiment is not a full reproduction of SPSM.
    Ensemble shrinkage is selected externally on forward development periods.
    """
    def __init__(self, spec, indicator='ref_rating_missing', minimum_rows=100):
        self.spec, self.indicator, self.minimum_rows = spec, indicator, minimum_rows

    def fit(self, x, y, **kwargs):
        from automl_agent.contracts import PipelineSpec
        from automl_agent.pipelines import make_estimator
        if kwargs:
            raise ValueError('Pattern experiment requires fixed iterations and no evaluation labels')
        spec = PipelineSpec.model_validate(self.spec)
        self.models_, self.support_ = {}, {}
        self.global_ = make_estimator(spec, False, {'cpu': 4})
        self.global_.fit(x, y)
        for value in [0., 1.]:
            mask = x[self.indicator].to_numpy() == value
            self.support_[str(value)] = int(mask.sum())
            if mask.sum() < self.minimum_rows:
                continue
            model = make_estimator(spec, False, {'cpu': 4})
            model.fit(x.loc[mask], np.asarray(y)[mask])
            self.models_[value] = model
        self.fit_count_ = 1 + len(self.models_)
        self.best_iteration_ = int(spec.hyperparameters['n_estimators'])
        return self

    def predict(self, x):
        result = self.global_.predict(x)
        for value, model in self.models_.items():
            mask = x[self.indicator].to_numpy() == value
            if mask.any():
                result[mask] = model.predict(x.loc[mask])
        return result
