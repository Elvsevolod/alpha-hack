from types import SimpleNamespace

import pytest

from automl_agent.contracts import DatasetReportArgs, InspectArgs
from automl_agent.data import Dataset
from automl_agent.run_tools import RunTools
from conftest import write_task


@pytest.fixture
def dataset(tmp_path, tabular):
    _, _, task = tabular
    data = Dataset(write_task(tmp_path))
    data.task = task.model_copy(update={
        "service_columns": ["period", "entity", "series", "audit_flag"],
        "time_column": "period", "group_column": "entity", "series_columns": ["series"]})
    data.sources["dictionary"] = {"sha256": "dictionary_version", "text":
        "column,role,description\n"
        "period,service,Observation time for validation\n"
        "entity,service,Keep each entity together\n"
        "series,service,Series membership\n"
        "audit_flag,service,Excluded annotation\n"
        "answer,target,Response description without row values\n"}
    return data


@pytest.mark.parametrize("column", ["period", "entity", "series", "audit_flag"])
@pytest.mark.parametrize("scope", ["fit", "validation_features", "test_features"])
def test_dictionary_reads_confirmed_metadata_without_promoting_to_predictors(dataset, column, scope):
    before = dataset.task.model_dump(mode="json")
    result = dataset.inspect(InspectArgs(report="dictionary", columns=[column], scope=scope), None)
    assert result["status"] == "available" and result["scope"] == "schema"
    assert column + ",service," in result["content"]
    assert result["sha256"] == "dictionary_version"
    assert dataset.task.model_dump(mode="json") == before
    assert column not in dataset.task.feature_columns


def test_dictionary_unknown_columns_still_rejected(dataset):
    with pytest.raises(ValueError, match="Unknown dictionary column"):
        dataset.inspect(InspectArgs(report="dictionary", columns=["not_a_confirmed_role"]), None)


def test_public_dataset_report_preserves_named_structural_description(dataset):
    tools = object.__new__(RunTools)
    tools.c = SimpleNamespace(dataset=dataset, plan=None)
    response = tools.dataset(DatasetReportArgs(report="dictionary", columns=["period", "entity"]))
    assert response["status"] == "success" and response["complete"]
    assert response["report"]["scope"] == "schema"
    assert "Observation time" in response["report"]["content"]
    assert "Keep each entity together" in response["report"]["content"]
    assert "audit_flag" not in response["report"]["content"]


@pytest.mark.parametrize("scope", ["validation_features", "test_features"])
def test_dictionary_does_not_bypass_target_scope_flag(dataset, scope):
    with pytest.raises(ValueError, match="Target-based analysis is restricted to fit"):
        dataset.inspect(InspectArgs(report="dictionary", columns=["answer"], scope=scope, include_target=True), None)


@pytest.mark.parametrize("scope", ["fit", "validation_features", "test_features"])
def test_metadata_dictionary_permission_does_not_expand_row_analysis(dataset, scope):
    plan = SimpleNamespace(fit=[0, 1], validation=[2, 3])
    with pytest.raises(ValueError, match="Column is unavailable in this analysis scope"):
        dataset.inspect(InspectArgs(report="missing", columns=["audit_flag"], scope=scope), plan)
