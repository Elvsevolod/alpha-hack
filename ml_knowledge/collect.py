"""Development-only acquisition of official, exact-tag repository snapshots.

No source code is executed. Archives are streamed and only text sources are kept.
The manifest, pinned commits, every path and byte hash form a frozen input set.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import re
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time
import urllib.request
import zipfile

from .core import canonical, digest, file_hash, read_json, validate_manifest, write_json

TEXT_SUFFIXES = {".md", ".mdx", ".rst", ".txt", ".py", ".pyi", ".pyx", ".ipynb", ".yaml", ".yml", ".json", ".rstinc", ".inc", ".h", ".cpp", ".pxd", ".in"}


def relevant(path, entry):
    p = PurePosixPath(path)
    if path in entry.get("supplemental_include_paths", []):
        return True, "referenced_text_include_from_exact_snapshot"
    root_notice = len(p.parts) == 1 and any(x in p.name.upper() for x in ("LICENSE", "COPYING", "NOTICE", "CHANGELOG", "RELEASE", "README"))
    # Retain all documentation inputs so includes outside published pages resolve.
    under = any(path.startswith(prefix) for prefix in entry["docs"] + list(entry["modules"]))
    if not (root_notice or under):
        return False, "outside_manifest_routes"
    if not root_notice and p.suffix.lower() not in TEXT_SUFFIXES:
        return False, "binary_or_site_asset"
    if any(part in {"test", "tests", "_static", "_templates", "__pycache__"} for part in p.parts):
        return False, "tests_or_site_decoration"
    if entry["package"] == "catboost" and path.startswith("catboost/docs/") and len(p.parts) > 2 and p.parts[2] in {"ru", "zh"}:
        return False, "duplicate_translation"
    return True, "manifest_text_source"


def acquire(entry, destination):
    target = destination / entry["package"]
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "snapshot.json"
    if marker.exists():
        frozen = read_json(marker)
        if frozen["manifest_entry_sha256"] != digest(entry):
            raise ValueError("Refuse to reuse snapshot from a different manifest")
        for item in frozen["files"]:
            if file_hash(target / "tree" / item["path"]) != item["sha256"]:
                raise ValueError("Corrupt source snapshot: " + item["path"])
        return frozen
    if entry.get("strategy") == "sdist_and_htmlzip":
        return acquire_archives(entry, target)
    if entry.get("strategy") == "sparse_git":
        return acquire_sparse(entry, target)
    ref = "refs/tags/" + entry["tag"]
    response = subprocess.run(["git", "ls-remote", entry["repository"] + ".git", ref, ref + "^{}"],
                              check=True, capture_output=True, text=True, timeout=90).stdout
    refs = dict((r.split()[1], r.split()[0]) for r in response.splitlines())
    commit = refs.get(ref + "^{}", refs.get(ref))
    if not commit or len(commit) != 40:
        raise ValueError("Exact tag did not resolve: " + entry["tag"])
    repo = entry["repository"].removeprefix("https://github.com/")
    url = "https://codeload.github.com/" + repo + "/tar.gz/" + commit
    archive = target / "source.tar.gz"
    if not archive.exists():
        # Development host curl uses its configured, verified CA trust store.
        # Never disable certificate verification to work around Python's CA setup.
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", "--retry", "2",
                        "--max-time", "600", "--output", str(archive.with_suffix(".part")), url],
                       check=True, timeout=1900)
        archive.with_suffix(".part").replace(archive)
    included, excluded = [], []
    with tarfile.open(archive, "r|gz") as source:
        for item in source:
            if not item.isfile():
                continue
            path = PurePosixPath(item.name)
            relative = PurePosixPath(*path.parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe archive path")
            keep, reason = relevant(str(relative), entry)
            if not keep:
                # All non-scope files are accounted for; no content is silently cut.
                excluded.append({"path": str(relative), "status": "out_of_scope", "reason": reason})
                continue
            raw = source.extractfile(item).read()
            dest = target / "tree" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            included.append({"path": str(relative), "sha256": digest(raw), "bytes": len(raw)})
    frozen = {"package": entry["package"], "version": entry["version"], "tag": entry["tag"],
              "commit": commit, "repository": entry["repository"], "source_url": url,
              "archive_sha256": file_hash(archive), "manifest_entry_sha256": digest(entry),
              "retrieved_at": datetime.now(timezone.utc).isoformat(), "license": entry["license"],
              "files": sorted(included, key=lambda x: x["path"]),
              "excluded": sorted(excluded, key=lambda x: x["path"])}
    write_json(marker, frozen)
    print(json.dumps({"package": entry["package"], "files": len(included), "bytes": sum(f["bytes"] for f in included), "commit": commit}), flush=True)
    return frozen


def acquire_archives(entry, target):
    files, excluded, archives = {}, [], []
    for source in entry["archives"]:
        archive = target / source["filename"]
        if not archive.exists():
            subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", "--retry", "2", "--max-time", "300", "--output", str(archive), source["url"]], check=True)
        sha = file_hash(archive)
        if source.get("sha256") and sha != source["sha256"]:
            raise ValueError("Official source distribution hash mismatch")
        archives.append({"source_url": source["url"], "sha256": sha})
        if source["filename"].endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                entries = [(name, z.read(name)) for name in z.namelist() if name.endswith(".html") and "/_" not in name]
            index = next((body.decode() for name, body in entries if name.endswith("/index.html")), "")
            if not re.search(r"<title>.*?" + re.escape(entry["version"]), index, re.S):
                raise ValueError("Version-family archive lacks exact patch evidence")
            for name, raw in entries:
                relative = "docs/" + name.split("/", 1)[1]
                dest = target / "tree" / relative
                if not dest.resolve().is_relative_to((target / "tree").resolve()):
                    raise ValueError("Unsafe source archive path")
                dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)
                files[relative] = {"path": relative, "sha256": digest(raw), "bytes": len(raw)}
        else:
            with tarfile.open(archive) as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    relative = member.name.split("/", 1)[1]
                    keep, reason = relevant(relative, entry)
                    if not keep:
                        excluded.append({"path": relative, "status": "out_of_scope", "reason": reason}); continue
                    dest = target / "tree" / relative
                    if not dest.resolve().is_relative_to((target / "tree").resolve()):
                        raise ValueError("Unsafe source archive path")
                    raw = tar.extractfile(member).read()
                    dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)
                    files[relative] = {"path": relative, "sha256": digest(raw), "bytes": len(raw)}
    frozen = {"package": entry["package"], "version": entry["version"], "tag": entry["tag"], "commit": None,
              "repository": entry["repository"], "source_url": entry["archives"][0]["url"], "archives": archives,
              "version_evidence": "PyPI sdist SHA256 and official HTML title explicitly naming " + entry["version"],
              "manifest_entry_sha256": digest(entry), "retrieved_at": datetime.now(timezone.utc).isoformat(),
              "license": entry["license"], "files": sorted(files.values(), key=lambda x:x["path"]), "excluded": excluded}
    write_json(target / "snapshot.json", frozen)
    print(canonical({"package": entry["package"], "files": len(files), "strategy": entry["strategy"]}), flush=True)
    return frozen


def acquire_sparse(entry, target):
    import shutil
    checkout = target / "checkout"
    if not checkout.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "--branch", entry["tag"], entry["repository"] + ".git", str(checkout)], check=True)
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    paths = subprocess.check_output(["git", "-C", str(checkout), "ls-tree", "-r", "--name-only", "HEAD"], text=True).splitlines()
    selected, excluded = [], []
    for path in paths:
        keep, reason = relevant(path, entry)
        if keep:
            selected.append(path)
        else:
            excluded.append({"path": path, "status": "out_of_scope", "reason": reason})
    subprocess.run(["git", "-C", str(checkout), "sparse-checkout", "set", "--no-cone", "--stdin"], input="\n".join("/" + name for name in selected) + "\n", text=True, check=True)
    files = []
    for name in selected:
        source = checkout / name
        if source.is_symlink() or not source.is_file():
            excluded.append({"path": name, "status": "out_of_scope", "reason": "non_regular_source_entry"}); continue
        dest = target / "tree" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        files.append({"path": name, "sha256": file_hash(dest), "bytes": dest.stat().st_size})
    frozen = {"package": entry["package"], "version": entry["version"], "tag": entry["tag"], "commit": commit,
              "repository": entry["repository"], "source_url": entry["repository"] + "/tree/" + commit,
              "manifest_entry_sha256": digest(entry), "retrieved_at": datetime.now(timezone.utc).isoformat(),
              "license": entry["license"], "files": files, "excluded": excluded}
    write_json(target / "snapshot.json", frozen)
    print(canonical({"package": entry["package"], "files": len(files), "commit": commit}), flush=True)
    return frozen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).parent)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    manifest = read_json(args.root / "manifest.json")
    validate_manifest(manifest, args.root / "requirements.txt")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [(e["package"], pool.submit(acquire, e, args.root / "sources")) for e in manifest["sources"]]
        results, errors = [], []
        for package, future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"package": package, "status": "unavailable", "error": type(exc).__name__ + ": " + str(exc)})
    write_json(args.root / "sources" / "acquisition.json", {"snapshots": [{"package": r["package"], "snapshot_sha256": digest(r)} for r in results], "errors": errors, "manifest_sha256": digest(manifest)})
    if errors:
        print(json.dumps(errors, ensure_ascii=False))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
