"""One user-authorized bounded Qwen trial; development supervisor, never delivered."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
LIMITS = ROOT / 'skills/alpha-api-limits/scripts/check_limits.py'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def run(archive, output):
    archive, output = archive.resolve(), output.resolve()
    load_dotenv(ROOT / '.env', override=False)
    environment = os.environ.copy()
    before = json.loads((output / 'api_limits_before.json').read_text())
    fingerprint = hashlib.sha256(environment.get('OPENAI_API_KEY','').encode()).hexdigest()[:16]
    if before.get('phase') != 'before' or before.get('key_fingerprint') != fingerprint or before.get('budget_exhausted') or before.get('limits',{}).get('blocked'):
        raise ValueError('Missing, mismatched or blocked pre-run budget snapshot')
    if (output / 'STARTED.json').exists():
        raise ValueError('This one-run authorization has already been used; no automatic repetition')
    directory=output/'task'
    with zipfile.ZipFile(archive) as package:
        assert all('/' not in name and (name.endswith('.py') or name=='requirements.txt') for name in package.namelist())
        assert package.read('requirements.txt')==(ROOT/'data_for_participants/requirements.txt').read_bytes()
        package.extractall(directory)
    manifest=json.loads((output/'MANIFEST.json').read_text())
    manifest.update(status='running',archive=archive.name,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        runtime_seconds=900,external_watchdog_seconds=900,task_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir() if p.name in {'task.md','task_spec.json','train.csv','test.csv','sample_submission.csv'}})
    write(output/'MANIFEST.json',manifest)
    name='alpha-api2-short-'+str(int(time.time()))
    command=['docker','run','--rm','--name',name,'--cpus','6','--memory','5g',
        '-v',str(directory)+':/task','-w','/task','-e','PYTHONDONTWRITEBYTECODE=1']
    for key in ('OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL','OPENAI_EXTRA_BODY'):
        if key in environment:command+=['-e',key]
    command+=['qwen-ml-agent:exact','python','solution.py','task.md','--time-limit','900','--cpu','6',
        '--memory-mb','5120','--max-calls','12','--max-candidates','8','--seed','42','--run-id','api2_live','--transport','auto',
        '--research-focus','Short API v2 verification. Inspect dataset properties, use relevant versioned knowledge or prior run evidence, and carry out distinct justified experiments. Consider a model-based imputation comparison. Preserve validation and complete control/refit within the allotted budget.']
    started=time.time();write(output/'STARTED.json',{'started_epoch':started,'container':name,'archive_sha256':manifest['archive_sha256']})
    code=None;timed_out=False
    try:
        with (output/'container.log').open('w') as stream:
            try:
                code=subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=900).returncode
            except subprocess.TimeoutExpired:
                timed_out=True
    finally:
        # The named container belongs exclusively to this invocation. Wait for it
        # to stop before the mandatory final API-limit observation.
        state=subprocess.run(['docker','inspect','--format','{{.State.Running}}',name],capture_output=True,text=True)
        if state.returncode==0 and state.stdout.strip()=='true':
            subprocess.run(['docker','stop','-t','2',name],capture_output=True)
            subprocess.run(['docker','wait',name],capture_output=True)
        elapsed=time.time()-started
        after=subprocess.run([sys.executable,str(LIMITS),'after','--before',str(output/'api_limits_before.json'),
            '--output',str(output/'api_limits_after.json')],cwd=ROOT,capture_output=True,text=True)
        (output/'api_limits_after_check.log').write_text(after.stdout+after.stderr)
        report_path=directory/'.agent_work/api2_live/run_report.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        summary={'archive_sha256':manifest['archive_sha256'],'exit_code':code,'watchdog_triggered':timed_out,
            'elapsed_seconds':elapsed,'after_check_exit_code':after.returncode,
            **{key:report.get(key) for key in ('state','control_status','refit_status','model_calls','research_candidates','failure')}}
        summary['passed']=code==0 and not timed_out and report.get('state')=='completed' and report.get('control_status')=='completed' and report.get('refit_status')=='completed'
        write(output/'RESULT.json',summary)
        print(json.dumps(summary),flush=True)
    return 0 if summary['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(run(args.archive,args.output))
