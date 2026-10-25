"""Freeze two completed recipes per task and replay new seeds sequentially."""
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from automl_agent.util import atomic_json,file_hash
from tools.pytorch_study_series import run
from tools.pytorch_study_analyze import evaluate


def repeat(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    datasets=['regression','classification','adult','bank','bike','har'];selected={};hashes={}
    for name in datasets:
        summary=evaluate(source/name,200)
        selected[name]=summary['selected_two'];hashes[name]=file_hash(source/name/'canonical_analysis.json')
        if len(selected[name])!=2:raise ValueError('Two completed recipes required: '+name)
    freeze=dict(selected=selected,screen_hashes=hashes,seeds=[17,42,2026],screen_seed=42,
        criterion='OOF-selected mixture primary development metric; standalone breaks ties',
        weights=[0,.1,.25,.5,.75,1],control_opened=False,created_unix=time.time(),production_gate='positive mean development gain on any task and positive direction in >=2/3 seeds; engineering criterion only')
    atomic_json(output/'selection_freeze.json',freeze)
    for name in ['adult','bank','regression','classification','bike','har']:
        for seed in [17,2026]:
            run(output/str(seed),[name],seed,selected[name])
            directory=output/str(seed)/name
            if (directory/'results.json').is_file():evaluate(directory,200)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--screen',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();repeat(a.screen,a.output)
