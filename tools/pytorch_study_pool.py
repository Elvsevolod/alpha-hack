"""Pool independently saved screening runs after checking shared data/splits/baselines."""
from pathlib import Path
import argparse,json,os
import numpy as np
from automl_agent.util import atomic_json,file_hash


def pool(screen,extra,output):
    screen,extra,output=map(Path,(screen,extra,output));output.mkdir(parents=True,exist_ok=False)
    for name in ['regression','classification','adult','bank','bike','har']:
        source=(screen/name).resolve();dest=output/name;dest.mkdir()
        for path in source.iterdir():
            if path.is_file():os.link(path,dest/path.name)
        origins=[str(source)];extra_dir=extra/name
        if (extra_dir/'results.json').is_file():
            left=json.loads((source/'protocol.json').read_text());right=json.loads((extra_dir/'protocol.json').read_text())
            if left['data_hash']!=right['data_hash']:raise ValueError('Data mismatch')
            a=np.load(source/'splits.npz');b=np.load(extra_dir/'splits.npz')
            for key in a:np.testing.assert_array_equal(a[key],b[key])
            for part in ['oof','full']:
                np.testing.assert_allclose(np.load(source/f'baseline_{part}.npy'),np.load(extra_dir/f'baseline_{part}.npy'),rtol=0,atol=1e-12)
            report=json.loads((dest/'results.json').read_text());added=json.loads((extra_dir/'results.json').read_text())
            report['results'] += [r for r in added['results'] if r.get('model')=='tiny_ft']
            report['pooled_additional_run']=str(extra_dir.resolve());atomic_json(dest/'results.json',report)
            for path in extra_dir.glob('tiny_ft*'):os.link(path,dest/path.name)
            origins.append(str(extra_dir.resolve()))
        protocol=json.loads((dest/'protocol.json').read_text())
        protocol['screening_origins']=[dict(directory=d,protocol_sha256=file_hash(Path(d)/'protocol.json'),results_sha256=file_hash(Path(d)/'results.json')) for d in origins]
        protocol['pool_policy']='No result overwritten in original run; identical data, split indices and baseline predictions required to combine candidates.'
        atomic_json(dest/'protocol.json',protocol)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--screen',required=True);p.add_argument('--extra',required=True);p.add_argument('--output',required=True);a=p.parse_args();pool(a.screen,a.extra,a.output)
