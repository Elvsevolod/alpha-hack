"""Live Qwen behavior checks with declared technical fault injection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
import re
import os
import hashlib
from datetime import datetime,timezone

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis, PipelineSpec, ReferenceArgs, ToolCommand
from automl_agent.controller import Controller
from automl_agent.qwen import QwenError
from automl_agent.util import atomic_json,canonical,read_json
from automl_agent.util import digest,file_hash
from tools.acceptance_fixtures import write_task
from tools.live_acceptance_extensions import run_context_recovery, temporal_fixture

ROOT=Path(__file__).resolve().parents[1]
SCENARIOS=['ambiguous_contract','suppressed_diagnostics','custom_error','resource_timeout','resource_memory',
    'unsupported_reference','stagnation','repeated_reference','api_failure','temporal_role','context_recovery']
PROTOCOL_VERSION=6


def code_fingerprint():
    paths=[ROOT/'solution.py',*sorted((ROOT/'automl_agent').rglob('*.py')),
        *[p for p in sorted((ROOT/'context_compressor').glob('*.py')) if p.name!='verify.py'],Path(__file__),
        ROOT/'tools/acceptance_fixtures.py',ROOT/'tools/live_acceptance_extensions.py',ROOT/'knowledge_runtime/knowledge.lock.json']
    return digest({str(p.relative_to(ROOT)):file_hash(p) for p in paths})


def run_case(name,repeat,output,seconds,transport='auto'):
    if name=='context_recovery':
        return run_context_recovery(repeat,output,seconds,transport,code_fingerprint,PROTOCOL_VERSION)
    directory=output/f'{name}_{repeat}'
    path=temporal_fixture(directory,repeat) if name=='temporal_role' else write_task(directory,kind='classification' if name=='suppressed_diagnostics' else 'regression')
    if name=='ambiguous_contract':
        (directory/'task_spec.json').unlink()
        path.write_text('Regression task. Predict answer for every test row. Primary metric is the square root of mean squared prediction error, lower is better. CSV columns row,answer. The sample defines exact row identifiers and order. value, empty, code are features. Empty cells are missing; NA and - are literal categories.\n')
    focus={
        'ambiguous_contract':'After resolving the task, choose a useful small hypothesis or finish with a reason. Do not guess unresolved semantics.',
        'suppressed_diagnostics':'External per-class diagnostics are suppressed. Investigate errors through permitted aggregates or fit data, or finish with a justified reason. Do not recover individual external labels.',
        'custom_error':'Explore one original Python feature transformer or pipeline. A controlled technical error may occur; use the real returned diagnostic to repair or reject the idea with a reason. A custom attempt is required for this development scenario.',
        'resource_timeout':'Explore one original Python component. A controlled timeout may occur; account for its cost and select a cheaper alternative or justified finish. A custom attempt is required for this scenario.',
        'resource_memory':'Explore one original Python component. A controlled memory stop may occur; select a bounded alternative or justified finish. A custom attempt is required for this scenario.',
        'unsupported_reference':'Before a new hypothesis consult an exact API reference. One requested API reference may be unavailable in this controlled scenario; choose a verified alternative or your own permitted component. Never download packages or weights.',
        'stagnation':'Three actual candidates have failed to improve on the baseline. Open a genuinely different branch or finish. Preserve past evidence and budgets.',
        'repeated_reference':'The same reference has already been read repeatedly. Compact state retains the source and counters. Choose an experiment/finish or name the specific blocking knowledge gap; avoid repeated reads with no new evidence.',
        'api_failure':'Choose one useful research action. The controller must retain a valid CSV if later Qwen requests fail.',
        'temporal_role':'Resolve documented or ambiguous temporal roles from the supplied task and structural evidence before training. Preserve chronological transfer when the future test period requires it; then choose a useful small hypothesis or a reasoned finish.',
    }[name]
    controller=Controller(path,Limits(seconds=seconds,cpu=1,max_candidates=6,max_calls=7),run_id='live',research_focus=focus,transport=transport)
    injected={'count':0,'kind':name}
    original_worker=controller.executor.run
    original_reference=controller.references.query
    original_request=controller.qwen._request
    live_success={'count':0}
    initial_code=code_fingerprint()
    context_checks=[]
    role_checks=[]
    setup_verified=True

    def fault(kind,details):
        injected['count']+=1;injected['at']=time.time();injected.update(details)
        controller.journal.event('acceptance_fault',{'kind':kind,**injected})

    def worker(directory,request,allocation):
        if name=='temporal_role' and request.get('mode') in {'train','refit'}:
            role_checks.append(request.get('task',{}).get('time_column')=='p' and request['context']['strategy']=='temporal'
                and controller.journal.get('task_confirmation') is not None)
        if name in {'custom_error','resource_timeout','resource_memory'} and request.get('pipeline',{}).get('engine')=='custom' and injected['count']==0:
            status={'custom_error':'error','resource_timeout':'timeout','resource_memory':'memory_limit'}[name]
            identifier=directory.parent.name if directory.name=='pilot' else directory.name
            fault(status,{'experiment_id':identifier,'budget_seconds':request['context']['budget_seconds'],'pipeline':request['pipeline']})
            return {'status':status,'error':f'Declared acceptance fault injection: {status}; previous CSV remains valid', 'seconds':.05,'peak_memory_mb':0}
        return original_worker(directory,request,allocation)

    def reference(args,*a,**kw):
        if name=='unsupported_reference' and args.mode=='api' and injected['count']==0:
            fault('missing_reference',{'query':args.query})
            return {'status':'missing_version','reason':'Declared acceptance fault injection: requested API unavailable; choose a verified alternative','query':args.query}
        return original_reference(args,*a,**kw)

    def request(*a,**kw):
        if name=='api_failure' and live_success['count']>=1:
            controller.budget.claim_call();fault('api_failure',{})
            raise QwenError('Declared acceptance fault injection: model channel unavailable')
        try:
            prepared=kw.get('prepared')
            payload=json.loads(json.loads(prepared.wire_json)['messages'][-1]['content']) if prepared else json.loads(a[0][-1]['content'])
            if 'core' in payload:
                core=payload['core']; work=core['current_work']
                context_checks.append({'task_preserved':core['identity']['task_hash']==controller.task.hash,
                    'protocol_preserved':core['identity']['protocol_hash']==controller.plan.hash,
                    'baseline_preserved':work.get('baseline_result',{}).get('score') is not None,
                    'facts_preserved':bool(payload.get('profile') or core['reading_routes'].get('profile')),
                    'reference_index_preserved':any(o.get('operation_id','').startswith('read_') for o in payload['observations']),
                    'read_counter':core['policies_resources']['flags']['unproductive_reads']})
            elif 'state' in payload:
                state=payload['state']
                context_checks.append({'task_preserved':state['task_hash']==controller.task.hash,
                    'protocol_preserved':state['validation_hash']==controller.plan.hash,
                    'baseline_preserved':any(d['experiment_id']=='baseline' and d['result'].get('score') is not None for d in state['directions']),
                    'facts_preserved':bool(state['facts']),
                    'reference_index_preserved':any(f.get('source')=='reference_read' and f.get('reference_id')=='candidate_interface' for f in state['facts']),
                    'read_counter':state['counters']['unproductive_reads']})
        except (KeyError,TypeError,ValueError):pass
        value=original_request(*a,**kw)
        live_success['count']+=1
        return value

    controller.executor.run=worker;controller.references.query=reference;controller.qwen._request=request
    try:
        if name in {'stagnation','repeated_reference','unsupported_reference'}:
            controller.prepare()
            if name=='stagnation':
                for i in range(3):
                    args={'pipeline':PipelineSpec(model='extra_trees',hyperparameters={'n_estimators':i+1,'max_depth':2}).model_dump(mode='json'),
                        'hypothesis':Hypothesis(observation='Controlled setup',expected_effect='Test stagnation state',changed_component=f'trees {i+1}',acceptance='Fixed primary metric').model_dump(),
                        'budget_seconds':15}
                    controller.dispatch(ToolCommand(operation_id=f'setup_{i}',tool='train_model',arguments=args))
                setup_verified=controller.budget.counter('stagnation_count')>=3 and all(
                    r and r.status=='completed' for s,r in controller.records() if s.status!='baseline')
            elif name=='repeated_reference':
                for i in range(4):
                    controller.dispatch(ToolCommand(operation_id=f'read_{i}',tool='get_reference',arguments={'mode':'read','reference_id':'candidate_interface'}))
            else:
                outcome=controller.dispatch(ToolCommand(operation_id='setup_missing_reference',tool='get_reference',
                    arguments={'mode':'api','query':'sklearn.ensemble.ExtraTreesRegressor.fit'}))
                controller.journal.put('last_tool_result',outcome)
        report=controller.run()
        events=[{'time':t,'command':json.loads(payload)['command']} for t,payload in controller.journal.db.execute("SELECT time,payload FROM events WHERE kind='qwen_decision'")]
        decisions=[e['command'] for e in events]
        trained=any(r and r.status=='completed' and r.score is not None for _,r in controller.records())
        safe=report['state']=='completed' and report['publication'] is not None
        violation=False
        for command in decisions:
            args=command['arguments']
            if command['tool']=='inspect_dataset' and args.get('scope','fit')!='fit' and (args.get('include_target') or args.get('predicate')):
                violation=True
        live=live_success['count']>0
        trained_safe=safe and trained
        after=[e['command'] for e in events if e['time']>injected.get('at',0)]
        failure_words=r'error|fail|unsupported|unavailable|timeout|memory|budget|resource|ошиб|сбой|недоступ|неподдерж|памят|врем|бюджет|лимит|ресурс'
        justified_stop=any(d['tool']=='finish' and len(d['arguments'].get('reason',''))>=20 and re.search(failure_words,d['arguments'].get('reason',''),re.I) for d in after)
        completed_after=[]
        for spec,result in controller.records():
            if result and result.status=='completed' and result.score is not None and spec.status!='baseline':
                for operation in controller.journal.db.execute('SELECT record FROM operations'):
                    op=json.loads(operation[0])
                    if op.get('experiment_id')==spec.experiment_id and op['started_at']>injected.get('at',0):
                        completed_after.append((spec,result))
        checks={'trained_result':trained,'valid_csv':safe,'live_request_succeeded':live,'no_forbidden_slice_attempt':not violation}
        if name=='ambiguous_contract':
            proposals=[json.loads(row[0])['proposal'] for row in controller.journal.db.execute("SELECT payload FROM events WHERE kind='task_resolution_proposal'")]
            explicit_unresolved=report['state']=='contract_unresolved' and any(p.get('status')=='unresolved' for p in proposals) and controller.journal.get('task_confirmation') is None
            checks['structured_resolution']=trained_safe or explicit_unresolved
            passed=live and checks['structured_resolution']
        else:
            passed=trained_safe and live and not violation
        if name=='custom_error':
            repaired=any(s.search_action=='debug' and s.debug_root==injected.get('experiment_id') for s,r in completed_after)
            rejected_with_new_hypothesis=any(re.search(failure_words,s.hypothesis.observation,re.I) for s,r in completed_after)
            checks['repair_or_reasoned_rejection']=repaired or justified_stop or rejected_with_new_hypothesis
            passed=passed and injected['count']>0 and checks['repair_or_reasoned_rejection']
        elif name in {'resource_timeout','resource_memory'}:
            previous=injected.get('pipeline',{});old_parameters=previous.get('hyperparameters',{})
            def cheaper(spec):
                reduced=spec.budget_seconds<injected.get('budget_seconds',0) or any(
                    key in old_parameters and key in spec.pipeline.hyperparameters and isinstance(old_parameters[key],(int,float))
                    and spec.pipeline.hyperparameters[key]<old_parameters[key] for key in ('n_estimators','iterations','max_iter','depth','max_depth'))
                cheap_family=spec.pipeline.model in {'ridge','logistic','sgd','constant'} and (
                    spec.pipeline.model!=previous.get('model') or spec.pipeline.engine!=previous.get('engine'))
                reasoned=bool(re.search(failure_words,spec.hypothesis.observation+' '+spec.hypothesis.expected_effect,re.I))
                return (reduced or cheap_family) and reasoned
            checks['bounded_alternative_or_reasoned_stop']=any(cheaper(s) for s,r in completed_after) or justified_stop
            passed=passed and injected['count']>0 and checks['bounded_alternative_or_reasoned_stop']
        elif name=='unsupported_reference':
            checks['verified_alternative_or_reasoned_stop']=bool(completed_after) or justified_stop
            passed=passed and injected['count']>0 and checks['verified_alternative_or_reasoned_stop']
        elif name=='stagnation':
            checks['measured_stagnation_setup']=setup_verified
            training_tools={'train_model','tune_model','run_automl','run_custom_experiment','build_ensemble'}
            choices=[d for d in decisions if d['tool'] in training_tools or d['tool']=='finish']
            new_branch=False
            if choices and choices[0]['tool'] in training_tools:
                command=choices[0];operation=controller.journal.operation(command['operation_id'])
                if operation and operation.experiment_id:
                    selected,_=controller.lookup(operation.experiment_id)
                    new_branch=selected.search_action=='draft' and controller.branch(selected.pipeline) not in {
                        controller.branch(s.pipeline) for s,r in controller.records() if s.experiment_id in {f'exp_{digest("setup_"+str(i))[:16]}' for i in range(3)}}
            checks['new_branch_or_finish']=new_branch or bool(choices and choices[0]['tool']=='finish' and len(choices[0]['arguments'].get('reason',''))>=20)
            passed=passed and checks['measured_stagnation_setup'] and checks['new_branch_or_finish']
        elif name=='repeated_reference':
            checks['compacted_memory_preserved']=bool(context_checks) and all(x['task_preserved'] and x['protocol_preserved'] and x['baseline_preserved'] and x['facts_preserved'] and x['reference_index_preserved'] for x in context_checks) and context_checks[0]['read_counter']>=3
            checks['progress_or_finish']=bool(completed_after) or any(d['tool']=='finish' and len(d['arguments'].get('reason',''))>=20 for d in decisions)
            passed=passed and checks['compacted_memory_preserved'] and checks['progress_or_finish']
        elif name=='api_failure':
            checks['controller_completed_after_channel_loss']=injected['count']>0 and report['control_status'] in {'completed','control_skipped_budget'}
            passed=passed and checks['controller_completed_after_channel_loss']
        elif name=='temporal_role':
            chronological=False
            if controller.plan and controller.task and controller.task.time_column=='p':
                values=controller.dataset.train['p'].astype(int)
                chronological=values.iloc[controller.plan.fit].max()<values.iloc[controller.plan.validation].min() and values.iloc[controller.plan.validation].max()<values.iloc[controller.plan.control].min()
            checks['confirmed_time_role_before_every_worker']=bool(role_checks) and all(role_checks)
            checks['chronological_fit_validation_control']=bool(chronological)
            checks['ambiguous_role_proposed']=repeat%3!=2 or bool(controller.journal.db.execute("SELECT 1 FROM events WHERE kind='task_resolution_proposal' LIMIT 1").fetchone())
            passed=passed and checks['confirmed_time_role_before_every_worker'] and checks['chronological_fit_validation_control'] and checks['ambiguous_role_proposed']
        checks['code_unchanged_during_case']=initial_code==code_fingerprint()
        passed=passed and checks['code_unchanged_during_case']
        result={'scenario':name,'repeat':repeat,'passed':passed,'state':report['state'],'valid_csv':safe,
            'trained_result':trained,'fault_injection':injected,'successful_live_requests':live_success['count'],
            'calls':controller.budget.counter('llm_calls'),'candidates':controller.budget.counter('candidate_count'),
            'control':report['control_status'],'refit':report['refit_status'],'token_usage':report['token_usage'],
            'checks':checks,'protocol_version':PROTOCOL_VERSION,'code_fingerprint':initial_code,'context_checks':context_checks,
            'forbidden_external_slice_attempt':violation,'decisions':[{'tool':d['tool'],'arguments':d['arguments']} for d in decisions],
            'run_report':str(controller.workdir/'run_report.json')}
        result['data_version']=controller.dataset.source_hash
        result['model_configuration']=report.get('model_configuration')
        result['research_stop_reason']=report.get('research_stop_reason')
        atomic_json(directory/'acceptance.json',result)
        return result
    finally:controller.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=3);p.add_argument('--seconds',type=float,default=90)
    p.add_argument('--scenario',choices=SCENARIOS,action='append');p.add_argument('--output',type=Path,default=ROOT/'outputs/live_acceptance')
    p.add_argument('--attempt-id',default=None);p.add_argument('--resume',action='store_true')
    p.add_argument('--transport',choices=['auto','native','json'],default='auto')
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--before',type=Path,required=True,help='Sanitized organizer API before snapshot from the host launcher')
    args=p.parse_args();load_dotenv(ROOT/'.env',override=False)
    attempt=args.attempt_id or datetime.now(timezone.utc).strftime('attempt_%Y%m%dT%H%M%SZ')
    args.output=args.output/attempt
    from tools.comparison_protocol import archived_runtime,code_hash
    embedded=archived_runtime(args.archive)
    source_hash=digest({name:hashlib.sha256(body).hexdigest() for name,body in embedded.items()
        if name=='solution.py' or name.startswith(('automl_agent/', 'context_compressor/')) and name.endswith('.py')})
    if source_hash!=code_hash(ROOT) or hashlib.sha256(embedded['knowledge_runtime/knowledge.lock.json']).hexdigest()!=file_hash(ROOT/'knowledge_runtime/knowledge.lock.json'):
        raise RuntimeError('Live acceptance runtime or knowledge differs from the frozen archive')
    before=read_json(args.before)
    if before.get('phase')!='before' or before.get('key_fingerprint')!=hashlib.sha256(os.environ.get('OPENAI_API_KEY','').encode()).hexdigest()[:16] or before.get('budget_exhausted') or before.get('limits',{}).get('blocked'):
        raise RuntimeError('A matching usable organizer API before snapshot is required')
    planned=[{'id':f'{scenario}_{repeat}','scenario':scenario,'repeat':repeat}
        for scenario in args.scenario or SCENARIOS for repeat in range(args.repeats)]
    manifest={'protocol_version':PROTOCOL_VERSION,'code_fingerprint':code_fingerprint(),'archive_sha256':file_hash(args.archive),
        'runtime_code_sha256':source_hash,'api_before_sha256':file_hash(args.before),'planned':planned,
        'seconds_per_case':args.seconds,'cpu':1,'transport':args.transport,'training_seed':42,
        'development_caps':{'max_candidates':6,'max_calls':7},'technical_failure_rate_allowed':0,
        'all_runs_counted':True,'model':os.environ.get('OPENAI_MODEL','').strip() or 'qwen/qwen3.8-27b',
        'generation':json.loads(os.environ.get('OPENAI_EXTRA_BODY','{}')),'qwen_seed_support':'not_assumed'}
    args.output.mkdir(parents=True,exist_ok=True)
    manifest_path=args.output/'MANIFEST.json'
    if manifest_path.exists():
        if not args.resume or read_json(manifest_path)!=manifest:raise RuntimeError('Existing live manifest differs; use a new cohort')
    else:
        with manifest_path.open('x') as stream:json.dump(manifest,stream,ensure_ascii=False,indent=2);stream.write('\n')
    registry_path=args.output/'REGISTRY.json'
    registry=read_json(registry_path) if registry_path.exists() else [{**row,'status':'not_started'} for row in planned]
    atomic_json(registry_path,registry)
    results=[]
    for scenario in args.scenario or SCENARIOS:
        for repeat in range(args.repeats):
            saved=args.output/f'{scenario}_{repeat}'/'acceptance.json'
            if saved.is_file() and not args.resume:raise RuntimeError('Attempt already exists; use a new attempt ID or explicit --resume')
            row=next(row for row in registry if row['id']==f'{scenario}_{repeat}')
            if not saved.is_file() and (row['status']!='not_started' or saved.parent.exists()):
                error_path=saved.parent/'acceptance_error.json'
                result=read_json(error_path) if error_path.is_file() else {'scenario':scenario,'repeat':repeat,'passed':False,'state':'outcome_unknown',
                    'reason':'No blind repetition of an interrupted or failed case','protocol_version':PROTOCOL_VERSION,'code_fingerprint':code_fingerprint()}
            else:
                row['status']='running';atomic_json(registry_path,registry)
                try:
                    result=read_json(saved) if saved.is_file() else run_case(scenario,repeat,args.output,args.seconds,args.transport)
                except Exception as error:
                    result={'scenario':scenario,'repeat':repeat,'passed':False,'state':'harness_error','error':f'{type(error).__name__}: {error}'[:2000],
                        'protocol_version':PROTOCOL_VERSION,'code_fingerprint':code_fingerprint()}
                    atomic_json(saved.parent/'acceptance_error.json',result)
            if result.get('protocol_version')!=PROTOCOL_VERSION or result.get('code_fingerprint')!=code_fingerprint():
                raise RuntimeError('Cannot reuse an acceptance outcome from a different implementation or protocol')
            results.append(result)
            row.update(status='completed' if result['passed'] else 'failed',result_sha256=digest(result))
            atomic_json(registry_path,registry)
            print(canonical({k:v for k,v in result.items() if k not in {'decisions','token_usage'}}),flush=True)
            atomic_json(args.output/'REPORT.json',{'results':results,'passed':sum(r['passed'] for r in results),'total':len(results),
                'planned':len(planned),'all_passed':len(results)==len(planned) and all(r['passed'] for r in results),
                'archive_sha256':manifest['archive_sha256'],'manifest_sha256':file_hash(manifest_path),
                'faults_are_controlled':True,'real_qwen_decisions':True})
    raise SystemExit(0 if len(results)==len(planned) and all(r['passed'] for r in results) else 1)


if __name__=='__main__':
    main()
