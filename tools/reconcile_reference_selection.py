"""A separately frozen correction of legacy baseline forecast provenance.

Older regression CV caches used early stopping on the query period. Reuse fixed
round reference forecasts instead; do not rewrite past selections or audits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from automl_agent.util import atomic_json,file_hash


def run(out,sources,seconds=300):
    from tools.strong_reference_study import archive,evaluate,locations
    from tools.plan_development_benchmark import tasks
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);archive(out,create=True)
    d=tasks()['regression'];configs={};imported={};parents=[]
    original,_=locations('regression')
    fixed=ROOT/'outputs/further_reference_20261025/regression'
    a=np.load(original/'splits.npz');b=np.load(fixed/'splits.npz')
    for field in ['fit0','fit1','eval0','eval1','development','control']:
        np.testing.assert_array_equal(a[field],b[field])
    for src in sources:
        src=Path(src).resolve();q=json.loads((src/'protocol.json').read_text())
        if q['data_hashes']['regression']!=d.source_hash:raise ValueError('Changed source data')
        parents.append(dict(protocol=str(src.relative_to(ROOT)),sha256=file_hash(src/'protocol.json')))
        for name,config in q['candidates']['regression'].items():
            if name in configs:raise ValueError('Ambiguous candidate identity')
            complete=all((src/'regression'/(name+'_cv'+i)/'result.json').exists() for i in ['0','1'])
            if not complete:continue
            configs[name]=config
            for i in ['0','1']:
                origin=src/'regression'/(name+'_cv'+i)
                destination=out/'regression'/(name+'_cv'+i);destination.mkdir(parents=True)
                for filename in ['prediction.npy','result.json']:
                    shutil.copyfile(origin/filename,destination/filename)
                    imported[str((origin/filename).relative_to(ROOT))]=file_hash(origin/filename)
    atomic_json(out/'protocol.json',dict(candidates={'regression':configs},data_hashes={'regression':d.source_hash},
        baseline_cv_policy='fixed_rounds',source_protocols=parents,imported_sha256=imported,
        source_hash=file_hash(Path(__file__)),worker_hash=file_hash(ROOT/'tools/strong_reference_study.py'),
        baseline_scores={'regression':39.03614722027966},weights=[0,.2,.4,1],
        selection='two chronological development origins; same fixed-round reference as final recipe',
        reason='Legacy search-early-stopped CV is not a comparable frozen incumbent; preserve earlier cycles as historical',
        source_hashes={p.name:file_hash(p) for p in (out/'reference_sources').glob('*.py')},
        automatic_promotion=False,api_calls=0,audit_previously_used=True))
    shutil.copyfile(Path(__file__),out/'source_snapshot.py')
    evaluate(out,{'regression':d},{'regression':configs},time.monotonic()+seconds)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,action='append',required=True);args=p.parse_args()
    run(args.output,args.source)
