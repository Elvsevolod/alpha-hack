"""Build the flat platform payload from project-owned sources and locked references."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automl_agent.util import atomic_bytes, atomic_json, file_hash
from tools.runtime_sources import production_agent_sources

ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP = '''"""Restore the immutable, project-owned agent and its offline reference data."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

BUNDLE_SHA256 = {bundle_hash!r}
FILES = {files!r}
PAYLOAD = (
{payload}
)

def materialize(task_directory):
    root = Path(task_directory).resolve() / '.agent_runtime' / BUNDLE_SHA256[:20]
    root.mkdir(parents=True, exist_ok=True)
    complete = all((root / name).is_file() and hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
                   for name, expected in FILES.items())
    if complete:
        return root
    body = base64.b85decode(PAYLOAD)
    if hashlib.sha256(body).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError('Embedded runtime checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(body)) as source:
        if set(source.namelist()) != set(FILES):
            raise RuntimeError('Unexpected embedded runtime members')
        for name, expected in FILES.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise RuntimeError('Invalid embedded runtime path')
            content = source.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError('Embedded runtime member checksum mismatch')
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.pending', delete=False) as stream:
                stream.write(content)
                temporary = Path(stream.name)
            temporary.replace(path)
    return root
'''

LAUNCHER = '''"""Platform entry point: python3 solution.py task.md."""
from pathlib import Path
import runpy
import sys
import time

STARTED_AT = time.time()

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / '.env', override=False)
    from agent_bundle import materialize
    # The platform copies code into the writable task directory.
    runtime = materialize(Path(__file__).resolve().parent)
    sys.path.insert(0, str(runtime))
    runpy.run_path(str(runtime / 'solution.py'), init_globals={'_AGENT_ENTRY_STARTED_AT': STARTED_AT}, run_name='__main__')
'''


def build(destination):
    destination.mkdir(parents=True, exist_ok=False)
    knowledge_release = ROOT / 'ml_knowledge/releases/ml_knowledge_v006.zip'
    release_manifest = json.loads(knowledge_release.with_suffix('.json').read_text())
    if file_hash(knowledge_release) != release_manifest['zip_sha256']:
        raise ValueError('Independent knowledge release checksum mismatch')
    knowledge_modules = ('core', 'lexical', 'runtime', 'memory', 'context')
    paths = [ROOT / 'solution.py', ROOT / 'requirements.txt',  *production_agent_sources(ROOT),
        *[ROOT / 'context_compressor' / name for name in ('__init__.py', 'contracts.py', 'store.py', 'core.py', 'transport.py', 'assembler.py')],
        ROOT / 'ml_knowledge/__init__.py', *[ROOT / 'ml_knowledge' / (name + '.py') for name in knowledge_modules]]
    # The large corpus stays in the standalone streaming shards, once. The
    # small package sources must be exactly those tested with that release.
    with zipfile.ZipFile(knowledge_release) as source:
        for module in knowledge_modules:
            code = (ROOT / 'ml_knowledge' / (module + '.py')).read_text()
            for dependency in knowledge_modules:
                code = code.replace('from .' + dependency + ' import', 'from mlkb_' + dependency + ' import')
            if source.read('mlkb_' + module + '.py').decode() != code:
                raise ValueError('Knowledge sources differ from their immutable release: ' + module)
        for name in source.namelist():
            if name == 'ml_knowledge_bundle.py' or name.startswith('mlkb_') and name.endswith('.py'):
                if Path(name).name != name:
                    raise ValueError('Knowledge release must be flat')
                with source.open(name) as incoming, (destination / name).open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            member = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(2026, 10, 20, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, path.read_bytes())
    body = buffer.getvalue()
    encoded = base64.b85encode(body).decode('ascii')
    payload = '\n'.join('    ' + repr(encoded[i:i+120]) for i in range(0, len(encoded), 120))
    bootstrap = BOOTSTRAP.format(bundle_hash=hashlib.sha256(body).hexdigest(), files=hashes, payload=payload)
    atomic_bytes(destination / 'agent_bundle.py', bootstrap.encode())
    atomic_bytes(destination / 'solution.py', LAUNCHER.encode())
    shutil.copyfile(ROOT / 'data_for_participants/requirements.txt', destination / 'requirements.txt')
    compile(bootstrap, 'agent_bundle.py', 'exec'); compile(LAUNCHER, 'solution.py', 'exec')
    flat_members = sorted(path.name for path in destination.iterdir())
    atomic_json(destination.parent / (destination.name + '_sources.json'), {
        'embedded_bundle_sha256': hashlib.sha256(body).hexdigest(), 'sources': hashes,
        'flat_members': flat_members,
        'knowledge_release_sha256': release_manifest['zip_sha256'],
        'knowledge_corpus': release_manifest['corpus'],
        'knowledge_files': {path.name: file_hash(path) for path in destination.iterdir() if path.name.startswith(('mlkb_', 'ml_knowledge_bundle'))},
        'requirements_sha256': file_hash(destination / 'requirements.txt'),
        'contains_third_party_wheels_or_packages': False, 'runtime_materialization_only': True})
    return {'files': len(flat_members), 'embedded_files': len(hashes), 'python_bundle_bytes': (destination / 'agent_bundle.py').stat().st_size}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output)))
