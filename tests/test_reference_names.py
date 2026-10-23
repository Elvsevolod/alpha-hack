from pathlib import Path

from automl_agent.contracts import ReferenceArgs
from automl_agent.references import ReferenceStore


ROOT = Path(__file__).resolve().parents[1]


def test_unique_short_api_name_returns_the_same_pinned_record_as_qualified_name():
    store = ReferenceStore(ROOT / "knowledge_runtime")
    for short, qualified in [("Ridge", "sklearn.linear_model.Ridge"), ("IterativeImputer", "sklearn.impute.IterativeImputer")]:
        alias = store.query(ReferenceArgs(mode="api", query=short, version="1.5.2"))
        exact = store.query(ReferenceArgs(mode="api", query=qualified, version="1.5.2"))
        assert alias["status"] == exact["status"] == "available"
        assert alias["entries"] == exact["entries"]
        assert alias["query_resolution"]["resolved_symbol"] == qualified
        assert all(entry["record"]["required_version"] == "1.5.2" for entry in alias["entries"])


def test_ambiguous_method_is_not_resolved_to_an_arbitrary_class():
    store = ReferenceStore(ROOT / "knowledge_runtime")
    result = store.query(ReferenceArgs(mode="api", query="fit"))
    assert result["status"] == "not_found" and "entries" not in result
    assert 1 < len(result["suggestions"]) <= 5
    assert all(s["symbol"].endswith(".fit") for s in result["suggestions"])


def test_short_name_never_bypasses_an_explicit_version_filter():
    store = ReferenceStore(ROOT / "knowledge_runtime")
    result = store.query(ReferenceArgs(mode="api", query="Ridge", library="scikit-learn", version="9.9.9"))
    assert result["status"] == "missing_version" and "entries" not in result


def test_natural_language_api_query_provides_bounded_canonical_suggestions():
    store = ReferenceStore(ROOT / "knowledge_runtime")
    result = store.query(ReferenceArgs(mode="api", query="ridge regression scikit-learn"))
    assert result["status"] == "not_found"
    assert result["suggestions"][0]["symbol"] == "sklearn.linear_model.Ridge"
    assert all("content" not in s for s in result["suggestions"])
