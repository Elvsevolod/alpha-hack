"""Build numbered flat Alpha ZIPs under 100 MB without importing solution code."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import zipfile


MAX_ARCHIVE_BYTES = 100_000_000
REQUIRED_FILES = {"solution.py", "requirements.txt"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check_name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("Invalid archive member name")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name or ":" in name:
        raise ValueError(f"Non-portable archive path: {name}")
    if len(path.parts) != 1 or "/" in name:
        raise ValueError(f"ZIP must be flat, without folders or nested files: {name}")
    if name != "requirements.txt" and (path.suffix != ".py" or name.startswith(".")):
        raise ValueError(f"Only root .py files and requirements.txt are copied by the platform: {name}")


def source_file(root, name):
    check_name(name)
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Source must be a regular file inside root: {name}")
    return path


def inspect_archive(path, root=None, requirements=None):
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Run the packager with Python 3.12 to validate the target Python version")
    members = []
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError(f"ZIP exceeds the 100 MB limit ({MAX_ARCHIVE_BYTES} bytes)")
    if requirements is None:
        raise ValueError("Provide the original organizer requirements.txt for comparison")
    requirements = Path(requirements).resolve()
    expected_requirements = requirements.read_bytes()
    root = Path(root).resolve() if root is not None else None
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member names")
        if "solution.py" not in names or archive.getinfo("solution.py").is_dir():
            raise ValueError("solution.py must be a file at the ZIP root, without a wrapper folder")
        if "requirements.txt" not in names or archive.getinfo("requirements.txt").is_dir():
            raise ValueError("The original requirements.txt must be a file at the ZIP root")
        for entry in entries:
            name = entry.filename
            check_name(name)
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not stat.S_ISREG(mode)):
                raise ValueError(f"Special file, directory or symlink in ZIP: {name}")
            data = archive.read(entry)  # read() verifies the member CRC
            if name.endswith(".py"):
                compile(data, name, "exec", dont_inherit=True)
            if name == "requirements.txt" and data != expected_requirements:
                raise ValueError("Packaged requirements.txt differs from the original organizer file")
            file_hash = sha256(data)
            if root is not None and sha256(source_file(root, name).read_bytes()) != file_hash:
                raise ValueError(f"ZIP differs from source: {name}")
            members.append({"path": name, "bytes": len(data), "sha256": file_hash})
    return {"archive": path.name, "bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()),
            "members": members, "source_match_checked": root is not None, "zip_and_syntax_valid": True,
            "flat_layout_checked": True, "archive_size_limit_bytes": MAX_ARCHIVE_BYTES,
            "requirements": {"filename": requirements.name, "sha256": sha256(expected_requirements)},
            "requirements_match_checked": True, "platform_contract": "alpha-flat-2026-10-20",
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "dependency_policy_checked": False, "embedded_payload_checked": False,
            "runtime_interface_checked": False}


def next_version(directory):
    versions = []
    for path in directory.iterdir():
        match = re.fullmatch(r"solution_v([0-9]+)\.(?:zip|manifest\.json)", path.name)
        if match:
            versions.append(int(match[1]))
    return max(versions, default=0) + 1


def build(root, names, baseline, directory, requirements):
    root, directory, baseline = Path(root).resolve(), Path(directory).resolve(), Path(baseline).resolve()
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
        raise ValueError("File list must be a non-empty JSON array of relative paths")
    if len(names) != len(set(names)) or not REQUIRED_FILES <= set(names):
        raise ValueError("File list must contain root solution.py and requirements.txt exactly once and no duplicates")
    sources = [(name, source_file(root, name)) for name in sorted(names)]
    baseline_hash = sha256(baseline.read_bytes())
    directory.mkdir(parents=True, exist_ok=True)
    version = next_version(directory)
    tag = f"v{version:03d}"
    output = directory / f"solution_{tag}.zip"
    manifest = output.with_suffix(".manifest.json")
    fd, temporary = tempfile.mkstemp(prefix=".solution-candidate-", suffix=".zip", dir=directory)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, source in sources:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, source.read_bytes())
        report = inspect_archive(temporary, root, requirements)
        report.update(archive=output.name, version=version, version_tag=tag,
                      baseline={"filename": baseline.name, "sha256": baseline_hash})
        # Same-filesystem link is exclusive: a concurrent publisher cannot overwrite a version.
        os.link(temporary, output)
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return {**report, "archive_path": str(output), "manifest_path": str(manifest)}
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--files", type=Path, required=True, help="JSON array of explicit source-relative paths")
    create.add_argument("--baseline", type=Path, required=True)
    create.add_argument("--requirements", type=Path, required=True, help="Unchanged organizer requirements.txt")
    create.add_argument("--out-dir", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--root", type=Path)
    verify.add_argument("--requirements", type=Path, required=True, help="Original organizer requirements.txt")
    args = parser.parse_args()
    try:
        result = build(args.root, json.loads(args.files.read_text()), args.baseline, args.out_dir, args.requirements) if args.command == "build" else inspect_archive(args.archive, args.root, args.requirements)
    except (ValueError, OSError, SyntaxError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Archive rejected: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
