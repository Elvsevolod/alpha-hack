"""Development-only rating factorization, an EPA-motivated hypothesis.

Site/source conversion and unobserved operating characteristics prevent exact
inversion of ENERGY STAR. Gamma shape and clipping are engineering choices.
"""
from __future__ import annotations

import numpy as np


def rating_factor(rating, shape):
    from scipy.stats import gamma
    if not np.isfinite(shape) or shape <= 0:
        raise ValueError('Gamma shape must be positive and finite')
    values = np.asarray(rating, dtype=float)
    if np.isinf(values).any():
        raise ValueError('Infinite rating is not supported')
    # Midpoints keep score endpoints away from zero/infinite gamma quantiles.
    percentile = np.clip((100.5 - values) / 100, .005, .995)
    result = gamma.ppf(percentile, a=shape, scale=1 / shape)
    return np.where(np.isnan(values), 1., result)


class RatingLGBM:
    def __init__(self, spec, shape):
        self.spec, self.shape = spec, shape

    def fit(self, x, y, **kwargs):
        from automl_agent.contracts import PipelineSpec
        from automl_agent.pipelines import make_estimator
        if kwargs:
            raise ValueError('Rating factorization uses fixed iterations without evaluation labels')
        spec = PipelineSpec.model_validate(self.spec)
        self.model_ = make_estimator(spec, False, {'cpu': 4})
        factor = rating_factor(x.energy_star_rating, self.shape)
        # w * (y/f - base_prediction)^2 = (y - f*base_prediction)^2.
        # This keeps the squared-error target in original units, although the
        # model parameterization and effects of regularization are different.
        self.model_.fit(x, np.asarray(y) / factor, sample_weight=factor ** 2)
        self.fit_count_ = 1
        self.best_iteration_ = int(spec.hyperparameters['n_estimators'])
        return self

    def predict(self, x):
        return self.model_.predict(x) * rating_factor(x.energy_star_rating, self.shape)
