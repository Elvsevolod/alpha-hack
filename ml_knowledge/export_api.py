"""Development-only exact installed docstring export, inside an offline container.

Import failures remain explicit gaps. API examples are never executed.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import sys
import warnings

from .core import canonical, digest, read_json, write_json


def export(entry, output):
    actual = importlib.metadata.version(entry["package"])
    if actual.split("+")[0] != entry["version"]:
        raise ValueError("Exact installed version required: " + entry["package"] + "=" + actual)
    roots = sorted(set(entry["modules"].values()))
    if entry["package"] == "flaml":
        roots.append("flaml")
    modules, errors, records = set(roots), [], {}
    for root in roots:
        module = importlib.import_module(root)
        for directory in getattr(module, "__path__", []):
            for source in sorted(Path(directory).rglob("*.py")):
                parts = source.relative_to(directory).with_suffix("").parts
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                if any(part.startswith("_") or part in {"test", "tests", "testing", "conftest", "setup", "examples", "sandbox", "benchmarks", "benchmark"} for part in parts):
                    continue
                # FLAML is scoped to AutoML, never agents/tune/LLM.
                if root == "flaml" and parts and parts[0] != "automl":
                    continue
                modules.add(".".join((root, *parts)))
    for name in sorted(modules):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module = importlib.import_module(name)
            names = getattr(module, "__all__", None) or [x for x in vars(module) if not x.startswith("_")]
            for member in sorted(set(names)):
                if not isinstance(member, str):
                    continue
                try:
                    obj = getattr(module, member)
                except Exception:
                    continue
                if not (inspect.isclass(obj) or callable(obj)):
                    continue
                owner = getattr(obj, "__module__", "") or ""
                if not any(owner.startswith(root.split(".")[0]) for root in roots):
                    continue
                emit(records, name + "." + member, obj)
                if inspect.isclass(obj):
                    for method, value in inspect.getmembers_static(obj):
                        if not method.startswith("_") and (callable(value) or isinstance(value, (property, classmethod, staticmethod))):
                            emit(records, name + "." + member + "." + method, value)
        except Exception as exc:
            errors.append({"module": name, "status": "unavailable", "reason": type(exc).__name__ + ": " + str(exc)[:500]})
    output.mkdir(parents=True, exist_ok=True)
    with (output / (entry["package"] + ".jsonl")).open("w", encoding="utf-8") as stream:
        for symbol, record in sorted(records.items()):
            stream.write(canonical(record) + "\n")
    write_json(output / (entry["package"] + ".report.json"), {"package": entry["package"], "version": actual,
               "python": sys.version, "modules_attempted": sorted(modules), "records": len(records), "errors": errors})
    print(canonical({"package": entry["package"], "records": len(records), "import_gaps": len(errors)}), flush=True)


def emit(records, symbol, obj):
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return
    try:
        signature = str(inspect.signature(obj))
    except (ValueError, TypeError):
        signature = None
    if signature:
        signature = re.sub(r" at 0x[0-9a-fA-F]+", "", signature)
    text = symbol + (signature or "") + "\n\n" + doc
    # Pointer addresses in descriptions are unstable, have no API meaning.
    text = re.sub(r" at 0x[0-9a-fA-F]+(?=>)", "", text)
    records[symbol] = {"symbol": symbol, "signature": signature, "text": text, "sha256": digest(text)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--package", required=True)
    args = parser.parse_args()
    entry = next(e for e in read_json(args.root / "manifest.json")["sources"] if e["package"] == args.package)
    export(entry, args.root / "sources" / "installed")

