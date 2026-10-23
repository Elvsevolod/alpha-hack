"""Real Optuna with a controlled clock: no long fits or sleep-based assertions."""
from types import SimpleNamespace

import pytest

from automl_agent.contracts import PipelineSpec, TrainingSummary
from automl_agent.search import SearchPipeline


@pytest.mark.parametrize('fit_seconds,budget,expect_success', [(.4, 2., True), (1.2, 1.5, False)])
def test_next_trial_cannot_evict_selected_fit(tabular, tmp_path, monkeypatch, fit_seconds, budget, expect_success):
    import automl_agent.search as search
    clock, calls = [0.], []
    class TimedPipeline:
        def __init__(self, *args):
            self.spec = args[0]
        def fit(self, X, y, context):
            clock[0] += fit_seconds
            calls.append(len(X))
            return TrainingSummary(rows=len(X), seconds=fit_seconds)
        def predict(self, *args):
            return SimpleNamespace(prediction_type='values')
        def frozen_spec(self):
            return self.spec
    monkeypatch.setattr(search.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(search, 'DirectPipeline', TimedPipeline)
    monkeypatch.setattr(search, 'align_predictions', lambda *a: {})
    monkeypatch.setattr(search, 'score_predictions', lambda *a: 1.)
    monkeypatch.setattr(search, 'trial_event', lambda *a, **kw: None)
    X, y, task = tabular
    spec = PipelineSpec(engine='optuna', model='ridge', n_trials=8,
        search_space={'alpha': {'type': 'float', 'low': .1, 'high': 1.}})
    pipeline = SearchPipeline(spec, task)
    context = {'strategy': 'random', 'cpu': 1, 'budget_seconds': budget, 'workdir': str(tmp_path),
        'protocol_hash': 'p', 'phase_estimates': {'selected_fit_seconds': .4, 'prediction_seconds': .1, 'publication_seconds': .1}}
    if expect_success:
        summary = pipeline.fit(X, y, context)
        assert len(summary.inner_scores['trials']) == 2 and len(calls) == 3
        assert calls[-1] == len(X) and clock[0] <= budget - .2
        assert summary.budget['stop_reason'] == 'selected_fit_reserve'
    else:
        with pytest.raises(ValueError, match='selected fit.*reserve'):
            pipeline.fit(X, y, context)
        assert len(calls) == 1  # An underestimated first fit cannot trigger more work.
