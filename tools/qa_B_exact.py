"""Short exact-archive B acceptance pair. Synthetic provider only; no organizer API."""
import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import diagnostic_3h as d
from tools.diagnostic_store import Store
from tools.qa_A_rehearsal import source_tasks
from tools.test_diagnostic_3h import gateway,dependencies_and_resources


def semantic_response(body):
    request=json.loads(body['messages'][-1]['content']);catalog=json.loads(request['sources']['inventory']['text'])
    classification='classification' in request['sources']['task']['text'];target='attack_cat' if classification else 'site_eui'
    roles={role:next(name for name in catalog if Path(name).name==filename) for role,filename in
        [('train.csv','observations.csv'),('test.csv','future.csv'),('sample_submission.csv','example.csv')]}
    semantic_audit = 'Interpretation audit' in request['sources']['task']['text']
    if semantic_audit:
        description = request['sources']['description_csv:input/roles.csv']
        assert 'Synthetic categorical feature' in description['text']
        roles['data_dictionary.csv'] = 'input/roles.csv'
    task={'contract_version':2,'file_roles':roles,'kind':'classification' if classification else 'regression',
        'targets':[target],'id_columns':['id'],'feature_columns':['x','cat'],'categorical_columns':['cat'],
        'metric':{'name':'f1' if classification else 'mae' if semantic_audit else 'rmse'},'output':{'columns':['id',target],'prediction_type':'labels' if classification else 'values'},
        'source_hash':request['source_hash']}
    proposal={'request_id':request['request_id'],'source_hash':request['source_hash'],'status':'proposed','task_spec':task,
        'evidence':{field:['task','inventory'] for field in task}}
    return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(proposal)}}], 'usage':{'prompt_tokens':10,'completion_tokens':10}}


def run(output,archive,semantic_audit=False):
    output=output.resolve();output.mkdir(parents=True,exist_ok=False);archive=archive.resolve()
    capacity=dependencies_and_resources(output);d.write(output/'CAPACITY.json',capacity)
    source=output/'source';source_tasks(source)
    # Easy, explicitly synthetic class task isolates three-finalist mechanics.
    training=source/'Многоклассовая классификация/train.csv'
    with training.open() as stream: reader=csv.DictReader(stream);columns=reader.fieldnames;records=list(reader)
    for row in records:row['attack_cat']=row['cat']
    with training.open('w') as stream: writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(records)
    jobs={};names=[];counter={};calls=[]
    for kind,folder in [('regression','Регрессия'),('classification','Многоклассовая классификация')]:
        task=output/kind/'task';task.mkdir(parents=True);(task/'input').mkdir()
        with zipfile.ZipFile(archive) as package:package.extractall(task)
        for old,new in [('train.csv','observations.csv'),('test.csv','future.csv'),('sample_submission.csv','example.csv')]:
            shutil.copyfile(source/folder/old,task/'input'/new)
        shutil.copyfile(source/folder/'task.md',task/'task.md')
        if semantic_audit:
            if kind=='regression':
                (task/'task.md').write_text((task/'task.md').read_text().replace('Primary metric: RMSE','Metrics are described below'))
            with (task/'task.md').open('a') as stream:
                stream.write('\nInterpretation audit. '+('Evaluation metric summary: auxiliary RMSE is reported for comparison; the primary metric is MAE, lower is better.' if kind=='regression' else 'Primary metric: Macro F1.'))
            (task/'input/roles.csv').write_text('feature,description\nx,Synthetic numeric feature\ncat,Synthetic categorical feature\n')
        name='alpha-B-'+kind+'-'+uuid.uuid4().hex[:8];names.append(name)
        jobs[kind]={'kind':kind,'directory':str(task),'container':name,'deadline_mono':time.monotonic()+180}
    def provider(handler,body,n):
        request=json.loads(body['messages'][-1]['content'])
        if 'sources' in request and 'request_id' in request:return 200,semantic_response(body)
        text=json.dumps(request)
        kind='classification' if 'attack_cat' in text else 'regression'
        counter[kind]=counter.get(kind,0)+1;step=counter[kind]
        if step==1:name,args='get_dataset_report',{'report':'missing'}
        elif step in (2,3):
            name,args='train_model',{'pipeline':{'model':'logistic' if kind=='classification' else 'ridge','hyperparameters':{'C':100 if step==2 else 1000} if kind=='classification' else {'alpha':.0001 if step==2 else .0002}},'budget_seconds':20,
                'hypothesis':{'observation':'Synthetic baseline exists; compare a fresh linear recipe','expected_effect':'Measure generalization on fixed validation',
                'changed_component':'model','acceptance':'Compare the exact primary metric'}}
        else:name,args='finish',{'reason':'Bounded fixture comparison completed; no justified next hypothesis in this offline test','result_ids':['baseline'],
            'imputation_skip_reason':'This test isolates semantic setup and finalization'}
        args['operation_id']=f'B_{kind}_{step}'
        calls.append({'kind':kind,'step':step,'tool':name})
        message={'tool_calls':[{'id':str(n),'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
        return 200,{'choices':[{'finish_reason':'stop','message':message}],'usage':{'prompt_tokens':10,'completion_tokens':10}}
    started=time.monotonic()
    try:
        with gateway(output/'diagnostics',provider,exact=True,jobs=jobs) as (server,guard,upstream):
            for kind,job in jobs.items():
                signal=output/'diagnostics/signals'/kind;signal.mkdir(parents=True);(signal/'HEALTHY').touch();(signal/'ARMED').touch()
                d.docker('run','-d','--name',job['container'],'--cpus','5','--memory','5g','--memory-swap','5g',
                    '-v',job['directory']+':/task','-v',str(signal)+':/signal','-w','/task',
                    '-e','OPENAI_API_KEY=offline-fixture','-e',f'OPENAI_BASE_URL=http://host.docker.internal:{server.server_port}/{kind}/v1',
                    '-e',f'AGENT_DIAGNOSTICS_URL=http://host.docker.internal:{server.server_port}/{kind}/diagnostics',
                    '-e','AGENT_DIAGNOSTICS_FAILURE_DIR=/signal',d.IMAGE,'python','solution.py','task.md','--cpu','5','--memory-mb','5120','--time-limit','180','--run-id','diagnostic')
            results=[]
            for kind,job in jobs.items():
                d.docker('wait',job['container'],timeout=210)
                state=json.loads(d.docker('inspect','--format','{{json .State}}',job['container']).stdout)
                logs=d.docker('logs',job['container'],check=False);(output/kind/'container.log').write_text(logs.stdout+logs.stderr)
                saved=d.read(Path(job['directory'])/'.agent_work/diagnostic/run_report.json',{})
                meta=d.metadata(job['directory'],job['container'])
                if semantic_audit:
                    assert meta['task']['file_roles']['data_dictionary.csv']=='input/roles.csv'
                    assert meta['task']['metric']['name']==('mae' if kind=='regression' else 'f1')
                finalists=meta.get('finalization_plan',{}).get('finalists',[])
                if kind=='regression':assert len(finalists)==3,meta.get('finalization_plan')
                # The CSV validator receives immutable aliases only after agent exit.
                for source_name,alias in [('observations.csv','train.csv'),('future.csv','test.csv'),('example.csv','sample_submission.csv')]:
                    shutil.copyfile(Path(job['directory'])/'input'/source_name,Path(job['directory'])/alias)
                verified=d.validate_csv(job)
                passed=(state['ExitCode']==0 and saved.get('control_status')==saved.get('refit_status')=='completed'
                    and saved.get('publication',{}).get('scope')=='full_train_refit' and verified['passed'])
                results.append({'kind':kind,'passed':passed,'exit_code':state['ExitCode'],'report':saved,'csv':verified,'finalists':finalists,'control_selection':meta.get('control_selection'),'finalization_plan':meta.get('finalization_plan')})
            end=time.monotonic()+120
            while any(row['status'] in {'queued','running'} for row in server.store.jobs()):
                assert time.monotonic()<end and not guard.closed;time.sleep(.1)
            failed=[row for row in server.store.jobs() if row['status']=='failed']
            history=Store(output/'diagnostics').verify()
            report={'passed':all(row['passed'] for row in results) and not failed and history['passed'],
                'organizer_requests':0,'semantic_audit':semantic_audit,'archive_sha256':d.digest(archive),'supervisor_sources':d.supervisor_sources(),
                'elapsed_seconds':time.monotonic()-started,'results':results,'history':history,'failed_backups':failed,'calls':calls,
                'scope':'Synthetic semantic responses; no proof of real-Qwen interpretation or public/private ML quality',
                'resources':{'cpu_per_task':5,'memory_gib_per_task':5,'capacity':capacity}}
            d.write(output/'REPORT.json',report);print(json.dumps({k:report[k] for k in ('passed','elapsed_seconds','history')}))
            assert report['passed'],results
    finally:
        for name in names:d.docker('rm','-f',name,check=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--semantic-audit',action='store_true')
    a=p.parse_args();run(a.output,a.archive,a.semantic_audit)
