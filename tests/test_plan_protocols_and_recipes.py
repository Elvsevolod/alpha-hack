import numpy as np
import pandas as pd
import pytest

from automl_agent.contracts import PipelineSpec, RecipeChange
from automl_agent.identity import feature_groups
from automl_agent.protocols import configure, paired_comparison
from automl_agent.recipe_patch import patch
from automl_agent.util import digest
from automl_agent.validation import choose_validation
from conftest import write_task


def test_canonical_exact_features_preserve_missing_types_and_named_columns():
    a = pd.DataFrame({"a": [1, 0, np.nan], "b": [None, "missing", "x"]})
    b = a[["b", "a"]].copy()
    b["a"] = b["a"].astype("Float64")
    assert np.array_equal(feature_groups(a), feature_groups(b))
    assert len(set(feature_groups(a))) == 3
    assert not np.array_equal(feature_groups(a), feature_groups(a.rename(columns={"a": "another"})))


def test_patch_preserves_other_fields_and_never_inherits_learned_state():
    base = PipelineSpec(model="lightgbm", training_objective="huber", target_transform="log1p",
        hyperparameters={"n_estimators": 40, "alpha": .8}, learned_inverse={"answer": {"factor": 1.2}})
    changed, preview = patch(base, digest(base.research_recipe()), [RecipeChange(path="hyperparameters.n_estimators", value=60)])
    assert changed.training_objective == "huber" and changed.target_transform == "log1p"
    assert changed.hyperparameters == {"n_estimators": 60, "alpha": .8}
    assert not changed.learned_inverse
    assert set(preview["diff"]) == {"hyperparameters"}
    with pytest.raises(ValueError, match="Stale"):
        patch(base, "a" * 64, [])
    for name in ("unknown", "learned_inverse.answer", "imputation.unknown"):
        with pytest.raises(ValueError):
            patch(base, digest(base.research_recipe()), [RecipeChange(path=name, value=4)])
    removed, preview = patch(base, digest(base.research_recipe()), [RecipeChange(path="hyperparameters.alpha", operation="remove")])
    assert "alpha" not in removed.hyperparameters and "alpha" in preview["diff"]["hyperparameters"]["before"]


def test_transfer_modes_preserve_closed_control_and_unseen_group_boundaries(tmp_path):
    from automl_agent.data import Dataset
    dataset = Dataset(write_task(tmp_path, kind="classification"))
    task = dataset.activate(dataset.explicit())
    dataset.train.loc[:, "value"] = np.arange(len(dataset.train)) // 3
    dataset.train.loc[:, "code"] = "category"
    dataset.train.loc[:, "empty"] = np.nan
    original = choose_validation(dataset)
    with pytest.raises(ValueError, match="Closed control transfer"):
        configure(dataset, original, "exact_group")
    revised = configure(dataset, original, "exact_group", initial=True)
    groups = feature_groups(dataset.train[task.feature_columns])
    assert not set(groups[revised.fit]) & set(groups[revised.validation])
    assert not set(groups[revised.fit + revised.validation]) & set(groups[revised.control])
    low = configure(dataset, original, "low_overlap", initial=True)
    assert low.hash != revised.hash
    assert not low.observations[-1]["selection_uses_candidate_scores"]


def test_rolling_keeps_disjoint_outer_partitions_and_serializes(tmp_path):
    from automl_agent.data import Dataset
    from automl_agent.contracts import ValidationPlan
    dataset = Dataset(write_task(tmp_path, temporal=True))
    dataset.activate(dataset.explicit())
    dataset.train[dataset.task.time_column] = np.repeat(np.arange(1, 6), len(dataset.train) // 5)
    dataset.test[dataset.task.time_column] = 6
    original = choose_validation(dataset)
    revised = configure(dataset, original, "rolling")
    assert revised.control == original.control
    assert not set(revised.fit) & set(revised.validation)
    assert not set(revised.control) & set(revised.fit + revised.validation)
    for part in revised.development_folds:
        assert max(dataset.train.iloc[part["fit"]][dataset.task.time_column]) < min(dataset.train.iloc[part["validation"]][dataset.task.time_column])
    assert ValidationPlan.model_validate_json(revised.model_dump_json()).hash == revised.hash


def test_rare_label_change_is_uncertain_and_full_class_order_remains_fixed(tmp_path):
    from automl_agent.contracts import TaskSpec, OutputSpec, MetricSpec
    from automl_agent.evaluation import score_predictions
    task = TaskSpec(kind="classification", targets=["y"], feature_columns=["x"], classes={"y": ["a", "b", "rare"]},
        metric=MetricSpec(name="f1", average="macro"), output=OutputSpec(columns=["y"], prediction_type="labels"), source_hash="fixture")
    y = pd.DataFrame({"y": ["a"] * 20 + ["b"] * 20 + ["rare"]})
    a = np.array(["a"] * 20 + ["b"] * 21)
    b = y.y.to_numpy()
    report = paired_comparison(task, y, a, b, np.arange(len(y)))
    assert report["status"] == "uncertain" and report["interval"] is None
    assert score_predictions(task, y.iloc[:20], {"y": np.array(["a"] * 20)}, "labels") == pytest.approx(1 / 3)
