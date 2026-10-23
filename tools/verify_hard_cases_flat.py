"""Read/routing verification from the exact delivered flat module, without Qwen."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import zipfile


def verify(archive, output):
    cases = [("classification", "f1_macro", "random", {"rare_classes": True}, "25"),
        ("regression", "rmse", "random", {"heavy_tail": True}, "33"),
        ("regression", "rmse", "temporal_group", {}, "04"),
        ("classification", "f1_macro", "random", {"label_conflicts": True}, "26"),
        ("regression", "rmse", "random", {"duplicate_target_variation": True}, "26"),
        ("regression", "rmse", "temporal", {"missingness_shift": True}, "19")]
    with tempfile.TemporaryDirectory(prefix="hard-cases-flat-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(archive) as package:
            assert all(Path(n).name == n and (n.endswith(".py") or n == "requirements.txt") for n in package.namelist())
            package.extractall(root)
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("ml_knowledge_bundle", root / "ml_knowledge_bundle.py")
        bundle = importlib.util.module_from_spec(spec); spec.loader.exec_module(bundle)
        corpus = bundle.restore(root / ".ml_knowledge_runtime")
        from mlkb_runtime import ReferenceStore
        from mlkb_context import SelectionContext, select
        from mlkb_core import canonical
        with ReferenceStore(corpus) as store:
            counts = dict(store.db.execute("SELECT kind,count(*) FROM docs GROUP BY kind"))
            assert counts["method"] == 92
            ml26 = store.document("method:hard_case_ml26")
            assert "duplicate_target_variation" in ml26["payload"].get("routing_aliases", [])
            assert "числов" in ml26["text"].casefold()
            assert store.document("api_sklearn_metrics_mean_squared_error")
            results = []
            for task, metric, strategy, flags, suffix in cases:
                context = SelectionContext(task, {"rows": 1000, "columns": 10, **flags}, strategy, "direct extra_trees", {}, metric=metric)
                packet = select(store, context)
                method = packet["materials"][0]
                passed = method["id"] == "method:hard_case_ml" + suffix and method["answer_usable"] and len(canonical(packet)) <= 6000
                assert "Diagnostic:" in method["content"] and "Acceptance:" in method["content"] and method["warnings"]
                results.append({"metric": metric, "strategy": strategy, "risk_flags": flags,
                    "method": method["id"], "passed": passed})
            report = {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "corpus": store.version,
                "counts": counts, "cases": results, "all_passed": all(r["passed"] for r in results),
                "ml26_alias_present": True, "required_api_source_present": True,
                "organizer_requests": 0, "scope": "Exact restored flat module, no source-checkout imports"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = verify(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["all_passed"] else 1)
