"""Record reduced, explicitly authorized offline acceptance; never claim live quality."""
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
    sha=d.digest(archive);checks={};evidence={}
    with zipfile.ZipFile(archive) as package:
        names=package.namelist()
        checks['flat_archive']=(len(names)==len(set(names)) and archive.stat().st_size<100000000
            and all('/' not in n and (n.endswith('.py') or n=='requirements.txt') for n in names)
            and package.read('requirements.txt')==(d.ROOT/'data_for_participants/requirements.txt').read_bytes())
        syntax=ast.parse(package.read('agent_bundle.py'))
        sources=next(ast.literal_eval(n.value) for n in syntax.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FILES' for t in n.targets))
        checks['embedded_sources']=all(d.digest(d.ROOT/name)==expected for name,expected in sources.items())
    for label,path in [('standard_entrypoint','flat_v031/REPORT.json'),('diagnostic_history','logging_v031/REPORT.json'),('exact_pair','B_exact_v031_acceptance/REPORT.json')]:
        report=d.read(root/path,{})
        checks[label]=report.get('passed') is True and report.get('archive_sha256')==sha
        evidence[label]=path
        if label=='diagnostic_history':
            retry=d.read(root/'logging_v031_worker_retry/REPORT.json',{})
            checks[label]=(report.get('archive_sha256')==sha and len(report.get('cases',[]))==5
                and all(case.get('passed') for case in report['cases'][:4]) and retry.get('passed') is True and retry.get('archive_sha256')==sha)
            evidence[label]=[path,'logging_v031_worker_retry/REPORT.json']
        if label=='exact_pair':
            checks['pair_finalization']=len(report.get('results',[]))==2 and all(row.get('passed') and row['report'].get('control_status')==row['report'].get('refit_status')=='completed'
                and row['report'].get('publication',{}).get('scope')=='full_train_refit' and row['csv'].get('passed') for row in report.get('results',[]))
            checks['three_finalists_exercised']=any(len(row.get('finalists',[]))==3 for row in report.get('results',[]))
            checks['host_source_binding']=report.get('supervisor_sources')==d.supervisor_sources()
    report=d.read(root/'B_exact_offline_v031/task/offline_checks/REPORT.json',{})
    checks['exact_offline_routes']=report.get('passed')==report.get('total')==12 and report.get('all_passed') and report.get('offline') and report.get('exact_environment')
    checks['base_regression_tests']=bool(re.search(r'322 passed', (root/'B_full_tests_final.log').read_text()))
    checks['post_freeze_role_and_protocol_tests']=bool(re.search(r'31 passed', (root/'B_post_freeze_targeted.log').read_text()))
    checks['final_temporal_ensemble_tests']=bool(re.search(r'38 passed', (root/'B_final_temporal_integration.log').read_text()))
    replay=d.read(root/'sqlite_guest_final/REPORT.json',{})
    checks['concurrent_sqlite_replay']=replay.get('passed') and replay.get('supervisor_sources')==d.supervisor_sources()
    legacy=d.read(root/'supervisor_sqlite_fix_02/REPORT.json',{})
    checks['supervisor_fault_suite_pre_deadline_delta']=legacy.get('passed') and len(legacy.get('cases',[]))==9
    manifest=d.read(root/'B_blocks_v031/MANIFEST.json',{})
    checks['four_reviewable_blocks']=len(manifest.get('blocks',[]))==4 and all(d.digest(root/'B_blocks_v031'/row['patch'])==row['sha256'] for row in manifest.get('blocks',[]))
    return {'schema_version':2,'candidate':str(archive),'archive_sha256':sha,'checks':checks,'evidence':evidence,
        'short_offline_accepted':all(checks.values()),'scope':'Reduced local acceptance explicitly requested by user; original three-hour/live gates remain unfulfilled',
        'original_long_rehearsal_passed':False,'long_rehearsal_repeat_waived_by_user':True,'B_before_live_A_authorized_by_user':True,
        'A_live_accepted':False,'B_live_accepted':False,'quality_advantage_established':False,'independent_full_plan_audit_complete':False,
        'platform_submitted':False,'full_plan_acceptance':False,'goal_complete':False,
        'limitations':['Base 322-test suite preceded final role/ensemble deltas; targeted 31/38 suites cover those deltas; counts overlap',
            'Nine-case supervisor suite preceded total-deadline tightening; final guest replay and exact pair bind current host sources',
            'All model responses are synthetic; no real-Qwen interpretation or independent hidden quality measurement',
            'Original long rehearsal failed and was not repeated; limited windows are excluded from full-vector comparisons'],
        'next_step':'Separate explicit API budget(s), fresh alpha-api-limits snapshots, then a short real public pilot; no automatic paid retries'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    r=check(a.root,a.archive);d.write(a.root/'ACCEPTANCE_v031.json',r);print(json.dumps(r,ensure_ascii=False,indent=2));raise SystemExit(0 if r['short_offline_accepted'] else 1)
