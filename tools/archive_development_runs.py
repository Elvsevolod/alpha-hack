"""Losslessly archive completed development fits; retain reports and publications."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automl_agent.util import atomic_json, file_hash, read_json


def archive(directory, min_age=30):
    report = directory / 'run_report.json'
    destination = directory / 'execution_artifacts.zip'
    if destination.exists() or not report.is_file() or time.time() - report.stat().st_mtime < min_age:
        return None
    state = read_json(report)
    if state['state'] != 'completed':
        return None
    with (directory / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        folders = [directory / name for name in ('attempts', 'finalization') if (directory / name).is_dir()]
        files = [p for folder in folders for p in sorted(folder.rglob('*')) if p.is_file()]
        if any(p.is_symlink() or not p.resolve().is_relative_to(directory.resolve()) for p in files):
            raise ValueError('Unexpected development artifact path')
        hashes = {str(p.relative_to(directory)): file_hash(p) for p in files}
        original_bytes = sum(p.stat().st_size for p in files)
        temporary = directory / 'execution_artifacts.pending.zip'
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for path in files:
                output.write(path, str(path.relative_to(directory)))
        with zipfile.ZipFile(temporary) as saved:
            for name, expected in hashes.items():
                with saved.open(name) as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != expected or file_hash(directory / name) != expected:
                    raise ValueError('Archive verification failed; original files retained')
        temporary.rename(destination)
        manifest = {'archive': destination.name, 'sha256': file_hash(destination), 'members': hashes,
            'original_bytes': original_bytes, 'archive_bytes': destination.stat().st_size,
            'all_original_hashes_verified': True,
            'restore': 'python tools/archive_development_runs.py --restore <run-directory>'}
        atomic_json(directory / 'execution_artifacts.manifest.json', manifest)
        for folder in folders:
            shutil.rmtree(folder)
        return {'run': str(directory), 'saved_bytes': original_bytes - destination.stat().st_size}


def restore(directory):
    manifest = read_json(directory / 'execution_artifacts.manifest.json')
    source = directory / manifest['archive']
    if file_hash(source) != manifest['sha256']:
        raise ValueError('Archive checksum mismatch')
    with zipfile.ZipFile(source) as saved:
        for name, expected in manifest['members'].items():
            destination = (directory / name).resolve()
            if not destination.is_relative_to(directory.resolve()):
                raise ValueError('Unsafe archive path')
            body = saved.read(name)
            if hashlib.sha256(body).hexdigest() != expected:
                raise ValueError('Artifact checksum mismatch')
            if destination.exists() and file_hash(destination) != expected:
                raise ValueError('Refusing to replace a changed artifact')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(body)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path, nargs='?')
    parser.add_argument('--restore', type=Path)
    args = parser.parse_args()
    if args.restore:
        restore(args.restore.resolve())
    else:
        if args.root is None:
            parser.error('A development output root is required')
        results = [result for path in sorted(args.root.glob('data/*/.agent_work/*')) if (result := archive(path))]
        print(json.dumps({'archived_runs': len(results), 'saved_bytes': sum(r['saved_bytes'] for r in results), 'runs': results}))
