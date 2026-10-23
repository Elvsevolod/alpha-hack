"""Create an immutable flat *module* release, not a competition solution."""
from __future__ import annotations

import argparse
import base64
from pathlib import Path
import tempfile
import zipfile
import zlib

from .core import canonical, digest, file_hash, read_json, write_json

MODULES = ("core", "lexical", "runtime", "memory", "context")

BOOTSTRAP = '''"""Offline, streaming ML knowledge restoration. No network or agent dependency."""
import base64
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zlib

MANIFEST = __MANIFEST__

def _hash(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()

def verify(root):
    root=Path(root)
    for name,spec in MANIFEST["files"].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or path.stat().st_size!=spec["bytes"] or _hash(path)!=spec["sha256"]:
            raise ValueError("Restored knowledge integrity error: "+name)
    return root

def restore(parent=None):
    parent=Path(parent) if parent is not None else Path.cwd()/".ml_knowledge_runtime"
    parent.mkdir(parents=True,exist_ok=True)
    target=parent/MANIFEST["corpus"]
    if target.exists():return verify(target)
    staging=Path(tempfile.mkdtemp(prefix=".restore-",dir=parent))
    try:
        for name,spec in MANIFEST["files"].items():
            dest=staging/name
            if not dest.resolve().is_relative_to(staging.resolve()):raise ValueError("Unsafe payload path")
            dest.parent.mkdir(parents=True,exist_ok=True)
            decoder=zlib.decompressobj();written=0
            with dest.open("wb") as out:
                for shard in spec["shards"]:
                    module=importlib.import_module(shard)
                    try:
                        for encoded in module.DATA:
                            compressed=base64.b85decode(encoded)
                            while compressed:
                                block=decoder.decompress(compressed,1048576)
                                compressed=decoder.unconsumed_tail
                                written+=len(block)
                                if written>spec["bytes"]:raise ValueError("Payload exceeds declared size")
                                out.write(block)
                    finally:
                        sys.modules.pop(shard,None)
                        del module
                tail=decoder.flush()
                written+=len(tail);out.write(tail)
            if not decoder.eof or decoder.unused_data or written!=spec["bytes"]:raise ValueError("Truncated payload")
        verify(staging)
        try:staging.rename(target)
        except FileExistsError:verify(target)
        return target
    finally:
        if staging.exists():shutil.rmtree(staging)

def open_store(parent=None,**kwargs):
    from mlkb_runtime import ReferenceStore
    return ReferenceStore(restore(parent),**kwargs)

if __name__=="__main__":
    import argparse
    from mlkb_core import canonical
    parser=argparse.ArgumentParser();parser.add_argument("--query",default="sklearn.preprocessing.OneHotEncoder")
    args=parser.parse_args()
    with open_store() as store:print(canonical(store.query(mode="api",query=args.query,max_chars=6000)))
'''


def release(root, corpus, destination):
    if destination.exists():
        raise FileExistsError("Numbered releases are immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = read_json(corpus / "knowledge.lock.json")
    with tempfile.TemporaryDirectory(prefix="mlkb-release-") as tmp:
        stage = Path(tmp)
        for module in MODULES:
            text = (root / (module + ".py")).read_text()
            for dependency in MODULES:
                text = text.replace("from ." + dependency + " import", "from mlkb_" + dependency + " import")
            (stage / ("mlkb_" + module + ".py")).write_text(text)
        manifest = {"corpus": lock["corpus"], "format": "streamed-zlib-shards-1", "files": {}}
        shard_id = 0
        for path in sorted(corpus.iterdir()):
            if not path.is_file():
                raise ValueError("Unexpected corpus subdirectory")
            spec = {"sha256": file_hash(path), "bytes": path.stat().st_size, "shards": []}
            compressor = zlib.compressobj(9)
            encoded_blocks, buffered = [], 0
            def flush_shard():
                nonlocal shard_id, encoded_blocks, buffered
                if not encoded_blocks:
                    return
                name = "mlkb_data_" + str(shard_id).zfill(4)
                (stage / (name + ".py")).write_bytes(b"DATA=(\n" + b"".join(repr(block).encode() + b",\n" for block in encoded_blocks) + b")\n")
                spec["shards"].append(name);shard_id += 1;encoded_blocks=[];buffered=0
            def emit(raw):
                nonlocal buffered
                for pos in range(0,len(raw),65536):
                    block=base64.b85encode(raw[pos:pos+65536]);encoded_blocks.append(block);buffered+=len(block)
                    if buffered>=512000:flush_shard()
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(1048576),b""):
                    emit(compressor.compress(block))
            emit(compressor.flush());flush_shard()
            manifest["files"][path.name]=spec
        (stage / "ml_knowledge_bundle.py").write_text(BOOTSTRAP.replace("__MANIFEST__", repr(manifest)))
        (stage / "requirements.txt").write_bytes((root / "requirements.txt").read_bytes())
        with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for path in sorted(stage.iterdir()):
                info=zipfile.ZipInfo(path.name,date_time=(2020,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
                z.writestr(info,path.read_bytes())
    report={"kind":"independent_knowledge_module_not_platform_solution","zip_sha256":file_hash(destination),
            "zip_bytes":destination.stat().st_size,"corpus":lock["corpus"],"files":len(MODULES)+shard_id+2,
            "requirements_sha256":file_hash(root/"requirements.txt"),"under_100000000_bytes":destination.stat().st_size<=100000000,
            "integration_status":"module_requires_host_validation"}
    write_json(destination.with_suffix(".json"),report)
    if not report["under_100000000_bytes"]:
        raise ValueError("Module alone exceeds total solution ZIP budget")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).parent)
    p.add_argument("--corpus",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();print(canonical(release(a.root,a.corpus,a.output)))
