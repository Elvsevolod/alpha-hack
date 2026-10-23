"""Replay a preserved failed control request offline; never alters the live run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(archive,original,output):
    output.mkdir(parents=True,exist_ok=False)
    source=output/'source';source.mkdir()
    with zipfile.ZipFile(archive) as package:package.extractall(source)
    sys.path.insert(0,str(source))
    from agent_bundle import materialize
    sys.path.insert(0,str(materialize(source)))
    from automl_agent.executor import Executor
    from automl_agent.journal import Journal
    from automl_agent.util import atomic_json
    directory=output/'attempt';directory.mkdir()
    inputs={}
    for name in ('fit_X.json','fit_y.json','control_X.json','test_X.json'):
        inputs[name]=sha(original/name);shutil.copyfile(original/name,directory/name)
        assert sha(directory/name)==inputs[name]
    request=json.loads((original/'request.json').read_text())
    assert request['pipeline']['engine']=='direct' and request['mode']=='refit'
    old_budget=request['context']['budget_seconds']
    request['read_dependencies']=[];request['model_path']=None
    # 90 seconds is below the corrected slot for the preserved ~128-second
    # finalization budget, with ~16 seconds full-refit and ~14 seconds other prep.
    request['context']['budget_seconds']=90
    journal=Journal(output/'journal')
    try:
        executor=Executor(5,3699.2,time.time()+92,journal)
        result=executor.run(directory,request,90)
        assert result['status']=='completed',result
        assert result['fit_rows']==45312
        assert not (directory/'control_y.json').exists()
        atomic_json(output/'REPORT.json',{'passed':True,'archive_sha256':sha(archive),
            'original_request_sha256':sha(original/'request.json'),'input_hashes':inputs,
            'original_worker_budget_seconds':old_budget,'replay_slot_seconds':90,
            'actual_seconds':result['wall_seconds'],'fit_rows':result['fit_rows'],
            'cpu':executor.cpu,'memory_mb':executor.memory_mb,'network':'none','organizer_requests':0,
            'scope':'Fresh isolated replay of failed baseline-control preparation, no control labels; original real B remains failed',
            'result':result})
    finally:journal.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.archive.resolve(),args.original.resolve(),args.output.resolve())
