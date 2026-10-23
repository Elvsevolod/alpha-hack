"""Bind delivery A offline verdicts to the exact archive and source hashes."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import diagnostic_3h as d


def check(root,archive):
    sha=d.digest(archive);checks={};pending=[]
    manifest=d.read(archive.with_suffix('.manifest.json'),{})
    checks['archive']=manifest.get('sha256')==sha and archive.stat().st_size<100_000_000
    with zipfile.ZipFile(archive) as package:
        names=package.namelist()
        checks['flat_layout']=(len(names)==len(set(names)) and all('/' not in n and (n.endswith('.py') or n=='requirements.txt') for n in names)
            and package.read('requirements.txt')==(d.ROOT/'data_for_participants/requirements.txt').read_bytes())
        syntax=ast.parse(package.read('agent_bundle.py'))
        sources=next(ast.literal_eval(n.value) for n in syntax.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FILES' for t in n.targets))
        checks['embedded_sources']=all(d.digest(d.ROOT/name)==expected for name,expected in sources.items())
        checks['archive_members']=all(hashlib.sha256(package.read(row['path'])).hexdigest()==row['sha256'] for row in manifest.get('members',[]))
    for label,relative,count in [('supervisor','supervisor_release',9),('history','logging_v028_final',5),('entrypoint','flat_v028',4)]:
        report=d.read(root/relative/'REPORT.json',{})
        checks[label]=(report.get('archive_sha256')==sha and report.get('passed') is True and len(report.get('cases',[]))==count
            and all(case['passed'] for case in report.get('cases',[])))
        if label=='supervisor':checks['supervisor_source_binding']=report.get('supervisor_sources')==d.supervisor_sources()
    checks['project_tests']=bool(re.search(r'309 passed[^\n]*$',(root/'full_tests_final.log').read_text().strip()))
    checks['knowledge_tests']=bool(re.search(r'46 passed[^\n]*$',(root/'knowledge_tests.log').read_text().strip()))
    context=d.read(root/'context_verification/verification.json',{}).get('tests',{})
    checks['context_tests']=context.get('successful') is True and context.get('run')==67 and context.get('failures')==context.get('errors')==0
    checks['exact_schema']=bool(re.search(r'Ran 1 test[^\n]*\n\nOK\s*$',(root/'exact_schema.log').read_text()))
    exact=d.read(root/'exact_offline/task/offline_checks/REPORT.json',{})
    checks['exact_offline_models']=exact.get('all_passed') is True and exact.get('offline') is True and exact.get('exact_environment') is True and exact.get('passed')==exact.get('total')==12
    emergency=d.read(root/'independent_signal/REPORT.json',{})
    checks['emergency_with_broken_audit']=(emergency.get('passed') is True and emergency.get('live_heartbeats') is True
        and emergency.get('container_and_descendants_stopped') is True and 0<emergency.get('actual_stop_latency_seconds',1e9)<=60)
    rehearsal=d.read(root/'rehearsal_3h_02/REPORT.json',{})
    if not rehearsal:pending.append('three_hour_rehearsal')
    checks['three_hour_rehearsal']=(rehearsal.get('passed') is True and rehearsal.get('archive_sha256')==sha
        and rehearsal.get('supervisor_sha256')==d.digest(Path(d.__file__)) and rehearsal.get('organizer_requests')==0
        and rehearsal.get('elapsed_seconds',0)>=10800 and rehearsal.get('configured_seconds')==10800
        and rehearsal.get('history',{}).get('records',0)>=20910 and rehearsal.get('history',{}).get('artifacts',0)>=16138
        and rehearsal.get('largest_verified_sqlite_copy_bytes',0)>=17690624 and rehearsal.get('two_tasks_completed') is True)
    location=d.read(root/'rehearsal_3h_02/RUN_LOCATION.json',{})
    full=d.read(Path(location.get('directory','/nonexistent'))/'REPORT.json',{})
    checks['rehearsal_control_refit_csv']=(full.get('full_run_passed') is True and len(full.get('results',[]))==2 and all(
        r.get('technical_passed') is True and r['report'].get('control_status')==r['report'].get('refit_status')=='completed'
        and r['report'].get('publication',{}).get('scope')=='full_train_refit' and r.get('csv',{}).get('passed') is True
        for r in full.get('results',[])))
    return {'schema_version':1,'archive_sha256':sha,'checks':checks,'pending':pending,'offline_passed':all(checks.values()),
        'live_A_accepted':False,'B_started':False,'official_quality':'unmeasured','goal_complete':False,
        'next_gate':'Separate budget approval and fresh alpha-api-limits before one live public pair; no automatic paid retries'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args();report=check(args.root,args.archive);d.write(args.root/'OFFLINE_VERDICT.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2));raise SystemExit(0 if report['offline_passed'] else 1)
