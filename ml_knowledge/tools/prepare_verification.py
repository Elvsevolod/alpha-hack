"""Prepare a clean flat-release check; the container receives no project checkout."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile


RUNTIME = ("core", "lexical", "runtime", "memory", "context")
CHECKS = ("build", "normalize", "evaluate", "replay", "benchmark")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


RUNNER = '''import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import unittest
import psutil

sys.path.insert(0, "/case")
output = Path("/results")
identity = json.loads(Path("/harness/identity.json").read_text())
process = psutil.Process()
baseline = process.memory_info().rss
began = time.perf_counter()
import ml_knowledge_bundle
store = ml_knowledge_bundle.open_store()
first = store.query(mode="api", query="sklearn.neural_network.MLPClassifier.fit", max_chars=6000)
cold = {"cold_import_restore_first_search_seconds": time.perf_counter()-began,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "rss_increase_bytes": process.memory_info().rss-baseline,
        "corpus": store.version, "response_status": first["status"],
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "network": "none", "cpu_quota": 1}
corpus = store.root
store.close()
from mlkb_core import file_hash, write_json
for name, expected in identity["flat_runtime_sha256"].items():
    assert file_hash(Path("/case")/name) == expected, name
write_json(output/"cold.json", {**identity, **cold})
with (output/"unit.log").open("w") as stream:
    unit = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.discover("/harness", pattern="test_module.py"))
write_json(output/"unit.json", {**identity, "tests": unit.testsRun, "failures": len(unit.failures), "errors": len(unit.errors), "passed": unit.wasSuccessful(), "python": platform.python_version()})
from mlkb_check_evaluate import evaluate
inputs = Path("/harness/inputs")
hashes = json.loads((inputs/"implementation_hashes.json").read_text())
for split in ("development", "heldout", "legacy_development"):
    path = output/(split+".json")
    evaluate(inputs, corpus, split, path, hashes)
    write_json(path, {**json.loads(path.read_text()), **identity})
from mlkb_check_replay import replay
from mlkb_runtime import ReferenceStore
with ReferenceStore(corpus) as store:
    replay(store, output/"replay.json")
from mlkb_check_benchmark import benchmark
queries = [item[0] for item in json.loads((inputs/"evaluation/legacy40.json").read_text())]
benchmark(corpus, queries, output/"resources.json", Path("/case/requirements.txt"))
for name in ("replay", "resources"):
    path = output/(name+".json")
    write_json(path, {**json.loads(path.read_text()), **identity})
assert unit.wasSuccessful(), "Technical tests failed"
assert json.loads((output/"heldout.json").read_text())["threshold_passed"], "Heldout regression failed"
assert json.loads((output/"replay.json").read_text())["all_passed"], "Replay failed"
print("Flat release checks completed")
'''


def prepare(root, release, destination):
    destination.mkdir(parents=True, exist_ok=False)
    flat = destination / "flat"
    flat.mkdir()
    harness = destination / "harness"
    harness.mkdir()
    inputs = harness / "inputs"
    (inputs / "evaluation").mkdir(parents=True)
    with zipfile.ZipFile(release) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert all("/" not in name and "\\" not in name and (name.endswith(".py") or name == "requirements.txt") for name in names)
        archive.extractall(flat)
    assert (flat/"requirements.txt").read_bytes() == (root/"requirements.txt").read_bytes()
    imports = set()
    for path in flat.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    violations = imports - sys.stdlib_module_names - {path.stem for path in flat.glob("*.py")}
    assert not violations, sorted(violations)
    mapping = {name: "mlkb_"+name for name in RUNTIME}
    mapping.update({name: "mlkb_check_"+name for name in CHECKS})
    for module in RUNTIME:
        expected = (root/(module+".py")).read_text()
        for dependency in RUNTIME:
            expected = expected.replace("from ."+dependency+" import", "from mlkb_"+dependency+" import")
        assert (flat/("mlkb_"+module+".py")).read_text() == expected, module
    for module in CHECKS:
        text = (root/(module+".py")).read_text()
        for original, target in mapping.items():
            text = text.replace("from ."+original+" import", "from "+target+" import")
        (harness/(mapping[module]+".py")).write_text(text)
    text = (root/"tests/test_module.py").read_text()
    for original, target in mapping.items():
        text = text.replace("from ml_knowledge."+original+" import", "from "+target+" import")
    (harness/"test_module.py").write_text(text)
    for name in ("annotations.json", "annotations.lock.json", "heldout_exposed.json", "legacy40.json"):
        shutil.copyfile(root/"evaluation"/name, inputs/"evaluation"/name)
    (inputs/"implementation_hashes.json").write_text(json.dumps({name+".py": sha(root/(name+".py")) for name in ("runtime", "context")}))
    identity = {"archive_sha256": sha(release), "archive_name": release.name, "archive_bytes": release.stat().st_size,
                "files": len(names), "flat": True, "requirements_unchanged": True, "source_identity": True,
                "flat_runtime_sha256": {"mlkb_"+name+".py": sha(flat/("mlkb_"+name+".py")) for name in RUNTIME},
                "runtime_dependencies": {"stdlib_and_delivered_modules_only": True, "imports": sorted(imports)}}
    (harness/"identity.json").write_text(json.dumps(identity, indent=2)+"\n")
    (harness/"run_checks.py").write_text(RUNNER)
    print(json.dumps(identity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.root, args.release, args.destination)
