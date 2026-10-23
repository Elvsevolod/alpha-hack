"""Resolve explicitly documented API objects missed by public-module traversal."""
import argparse
import importlib
import importlib.metadata
import inspect
from pathlib import Path

from .core import canonical, read_json, write_json
from .export_api import emit


def resolve(symbol):
    parts=symbol.split('.')
    for n in range(len(parts),0,-1):
        try:
            obj=importlib.import_module('.'.join(parts[:n]))
        except (ImportError,ModuleNotFoundError):
            continue
        for part in parts[n:]: obj=inspect.getattr_static(obj,part) if inspect.isclass(obj) else getattr(obj,part)
        return obj
    raise ImportError(symbol)


def main(root, package):
    entry=next(e for e in read_json(root/'manifest.json')['sources'] if e['package']==package)
    if importlib.metadata.version(package).split('+')[0]!=entry['version']:raise ValueError('Version mismatch')
    source=root/'sources/installed'/(package+'.jsonl')
    records={obj['symbol']:obj for line in source.read_text().splitlines() if (obj:=__import__('json').loads(line))}
    gaps=read_json(root/'frozen/coverage.json')['unresolved']
    requested=sorted({g['target'] for g in gaps if g['library']==package and g['kind'] in {'autosummary','autofunction','autoclass','automethod','autoattribute','autodata','autoproperty','automodule'}})
    failures=[];before=len(records)
    for symbol in requested:
        try:emit(records,symbol,resolve(symbol))
        except Exception as exc:failures.append({'symbol':symbol,'status':'unavailable','reason':type(exc).__name__+': '+str(exc)[:300]})
    source.write_text(''.join(canonical(obj)+'\n' for key,obj in sorted(records.items())))
    write_json(root/'sources/installed'/(package+'.requested.json'),{'requested':requested,'added':len(records)-before,'failures':failures})
    print(canonical({'package':package,'requested':len(requested),'added':len(records)-before,'unresolved':len(failures)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).parent);p.add_argument('--package',required=True);a=p.parse_args();main(a.root,a.package)
