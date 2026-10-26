"""A cost pilot must support the stopping splits inside its OOF fits."""
import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import MetricSpec, OutputSpec, PipelineSpec, TaskSpec
from automl_agent.research_evaluation import oof_plan
from automl_agent.validation import inner_split, pilot_indices, pilot_population, InsufficientPilotSupport


def fixture(rare=7):
    X = pd.DataFrame({'x': np.arange(600)})
    y = pd.DataFrame({'target': ['rare'] * rare + ['common'] * (600 - rare)})
    task = TaskSpec(kind='classification', targets=['target'], feature_columns=['x'],
        classes={'target': ['common', 'rare']}, metric=MetricSpec(name='f1', average='macro'),
        output=OutputSpec(columns=['target'], prediction_type='labels'), source_hash='fixture')
    spec = PipelineSpec(model='lightgbm', early_stopping_rounds=3,
        prediction_policy='class_offsets', postprocess_fit_policy='pooled_oof')
    return X, y, task, spec


def test_rare_class_missing_only_inside_pilot_oof_stopping_expands_population():
    X, y, task, spec = fixture()
    pilot = pilot_indices(X, y, task, 'exact_group', 100, 42, min_class_rows=12, min_groups=6)
    parts, _ = oof_plan(spec, task, X.iloc[pilot], y.iloc[pilot], 'exact_group', 42)
    with pytest.raises(ValueError, match='training class coverage'):
        for fit, _, fresh, _ in parts:
            inner_split(X.iloc[pilot].iloc[fit], y.iloc[pilot].iloc[fit], task, 'exact_group', fresh.seed)
    rows, support = pilot_population(X, y, task, 'exact_group', 100, 42, spec)
    assert rows.tolist() == list(range(600))
    assert support['expanded_for_structural_support']
    assert support['failure_on_small_sample_is_recipe_failure'] is False
    for fit, _, fresh, _ in oof_plan(spec, task, X.iloc[rows], y.iloc[rows], 'exact_group', 42)[0]:
        inner_split(X.iloc[rows].iloc[fit], y.iloc[rows].iloc[fit], task, 'exact_group', fresh.seed)


def test_full_population_missing_nested_class_support_remains_rejected():
    X, y, task, spec = fixture(rare=1)
    spec = spec.model_copy(update={'seed': 1})
    with pytest.raises(InsufficientPilotSupport, match='full permitted fit'):
        pilot_population(X, y, task, 'exact_group', 600, 1, spec)
