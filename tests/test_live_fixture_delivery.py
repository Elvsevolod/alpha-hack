import pytest

from conftest import write_task as original_fixture
from tools.acceptance_fixtures import write_task


@pytest.mark.parametrize("options", [{}, {"kind": "classification"}, {"kind": "classification", "probability": True},
    {"temporal": True}, {"groups": True}, {"multilabel": True}])
def test_exact_dependency_live_fixture_preserves_existing_case_data(tmp_path, options):
    before, after = tmp_path / "before", tmp_path / "after"
    original_fixture(before, **options)
    write_task(after, **options)
    assert {p.name: p.read_bytes() for p in before.iterdir()} == {p.name: p.read_bytes() for p in after.iterdir()}
