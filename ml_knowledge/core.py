"""Shared deterministic encoding. Runtime uses only the Python standard library."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

FORMAT_VERSION = "mlkb-1"
NORMALIZER_VERSION = "text-1"
CHUNKER_VERSION = "paragraph-1"
RANKING_VERSION = "lexical-1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    if not isinstance(value, bytes):
        value = (value if isinstance(value, str) else canonical(value)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def requirements(path):
    import re
    return {name.lower().replace("_", "-"): version for name, version in
            re.findall(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==([^\s#]+)",
                       Path(path).read_text(), flags=re.M)}


def validate_manifest(manifest, requirement_file):
    pinned = requirements(requirement_file)
    packages = [entry["package"] for entry in manifest["sources"]]
    if len(packages) != 16 or len(set(packages)) != 16:
        raise ValueError("The manifest must enumerate the 16 distinct agreed ML libraries")
    for entry in manifest["sources"]:
        if pinned.get(entry["package"].lower().replace("_", "-")) != entry["version"]:
            raise ValueError("Version mismatch: " + entry["package"])
        if not entry["repository"].startswith("https://github.com/") or not entry["tag"]:
            raise ValueError("An official repository and exact tag are required")

