"""One sequential, explicitly budgeted development series; no automatic reruns."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from automl_agent.budget import Budget, Limits
from automl_agent.eda.agent import run_agent
from automl_agent.eda.deepseek import CostLedger, DeepSeekClient
from automl_agent.journal import Journal
from automl_agent.qwen import QwenError
from automl_agent.util import atomic_json, atomic_bytes


def series(root):
    root=root.resolve();runs=root/'runs'
    if runs.exists():raise ValueError('Series already started; inspect saved results instead of repeating')
    runs.mkdir(parents=True)
    ledger_path=root/'deepseek_costs.json'
    journal=Journal(runs/'connection')
    ledger=CostLedger(ledger_path,'connection',cap=.05)
    client=DeepSeekClient(journal,Budget(journal,Limits(seconds=60,max_calls=1),time.time()),ledger)
    summaries={}
    connection_error=None
    try:
        client._request([{'role':'user','content':'Reply with JSON {"ok":true}. This is a connection test.'}])
        atomic_json(runs/'connection'/'summary.json',ledger.summary())
    except QwenError as error:
        atomic_json(runs/'connection'/'summary.json',{'error':str(error),'cost':ledger.summary()})
        connection_error=str(error)
    finally:
        client.close();journal.close()
    for name in ([] if connection_error else ['adult','bank','bike','har']):
        summaries[name]=run_agent(root/'tasks'/name/'task.md',runs/name,ledger_path,seconds=600)
        atomic_json(root/'series.json',summaries)
        print(json.dumps({'task':name,**summaries[name]},ensure_ascii=False),flush=True)
        if summaries[name]['cost']['outcome_unknown']:
            break
    write_report(root, summaries, connection_error)
    return summaries


def write_report(root, summaries, connection_error=None):
    ledger_path=root/'deepseek_costs.json'
    atomic_json(root/'series.json', summaries)
    lines=['# Autonomous EDA development series','',
        'Cost is a conservative USD estimate at frozen peak cache-miss rates; the $0.15 reserve is unavailable.',
        'Closed external labels are not used. Findings below require separate reviewer assessment.', '',
        '| Task | State | Fits | Generations | USD estimate | Stop |',
        '|---|---|---:|---:|---:|---|']
    for name in ['adult','bank','bike','har']:
        result=summaries.get(name)
        if not result:
            lines.append(f'| {name} | not run | 0 | 0 | 0 | series stopped |');continue
        lines.append(f"| {name} | {result['state']} | {result['model_fits']} | {result['api_calls']} | {result['cost']['usd']:.6f} | {result['stop_reason']} |")
    for name,result in summaries.items():
        lines += ['',f'## {name}', '',(result.get('finish') or {}).get('reason','No agent finish; inspect trajectory and stop reason.'),'']
    if connection_error:
        lines += ['', 'Connection failure: '+connection_error, 'No task generations were started; the unknown reservation is retained.']
    state=json.loads(ledger_path.read_text())
    atomic_json(root/'cost_summary.json',{'estimated_usd':sum(r['usd'] for r in state['requests']),
        'settled_usd':sum(r['usd'] for r in state['requests'] if r['state']=='settled'),
        'reserved_unknown_usd':sum(r['usd'] for r in state['requests'] if r['state']=='reserved'),
        'unknown_reservations':sum(r['state']=='reserved' for r in state['requests']),'spendable_cap_usd':.85,'protected_reserve_usd':.15,'pricing':state['pricing']})
    atomic_bytes(root/'REPORT.md',('\n'.join(lines)+'\n').encode())
    return summaries

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    results=series(args.root)
    raise SystemExit(0 if len(results)==4 and all(r['state']=='completed' for r in results.values()) else 1)
