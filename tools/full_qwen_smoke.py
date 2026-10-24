"""Short full-controller Qwen check; limits snapshots are development-only."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dotenv import dotenv_values
from tools.qwen_eda_live import limits
from automl_agent.controller import Controller
from automl_agent.budget import Limits
from automl_agent.util import atomic_json


def run(task, output, seconds=180, calls=3):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    config=dotenv_values(ROOT/'.env')
    for key in ('OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL'):
        if key not in os.environ and config.get(key):os.environ[key]=config[key]
    key=os.environ.get('OPENAI_API_KEY')
    before=limits.snapshot(limits.fetch_info(key),key,'before')
    atomic_json(output/'api_limits_before.json',before)
    if before['budget_exhausted'] or before['limits']['blocked']:raise ValueError('Gateway unavailable')
    c=None
    try:
        c=Controller(Path(task),Limits(seconds=seconds,cpu=2,memory_mb=4096,max_calls=calls,max_candidates=2),
            run_id='full_qwen_smoke',workdir=output/'run')
        result=c.run()
        summary={'mode':'full_controller_live_qwen','model':c.qwen.model,'state':result['state'],
            'control_status':result['control_status'],'refit_status':result['refit_status'],
            'failure':result.get('failure'),'generations':c.budget.counter('llm_calls'),
            'submission_exists':bool(c.publisher and c.publisher.output.exists()),
            'limits':{'seconds':seconds,'generations':calls,'candidates':2,'cpu':2,'memory_mb':4096},
            'quality_claim':'None: a short technical run cannot establish full-budget research quality.'}
        atomic_json(output/'summary.json',summary)
        print(json.dumps(summary),flush=True)
        return summary
    finally:
        if c:c.close()
        try:
            after=limits.snapshot(limits.fetch_info(key),key,'after');after['comparison']=limits.compare(before,after)
        except Exception as error:
            after={'phase':'after','remaining_budget':None,'error':type(error).__name__}
        atomic_json(output/'api_limits_after.json',after)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.task,a.output)
