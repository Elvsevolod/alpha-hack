"""One registered, non-executing replay of a failed native training command."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time

import httpx
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import diagnostic_3h as d
from automl_agent.contracts import TOOL_SCHEMAS, TaskSpec, ExperimentSpec
from automl_agent.compatibility import require_compatible
from automl_agent.knowledge import AgentReferences
from automl_agent.pipelines import effective_parameters
from automl_agent.qwen_stream import completion_from_sse, IncompleteModelStream
from context_compressor.transport import Transport


DESCRIPTIONS={
    'pipeline':'A PipelineSpec JSON object. Keep it as an object; never JSON-encode it into a string.',
    'hypothesis':'A Hypothesis JSON object. Keep it as an object; never JSON-encode it into a string.',
    'method_refs':'Method-card IDs only (materials with type=method). Do not put documentation or memory IDs here.',
    'reference_ids':"Documentation IDs only: get_reference api/read results or reference_evidence materials with type=documentation. Method-card IDs belong in method_refs; memory IDs and source hashes never belong here.",
}


def annotate(body):
    body=json.loads(json.dumps(body))
    count=0
    for tool in body['tools']:
        properties=tool['function']['parameters'].get('properties',{})
        for name,description in DESCRIPTIONS.items():
            if name in properties:
                properties[name]['description']=description
                count+=1
    assert count>=4
    return body,count


def validate_action(body, response, task, baseline, journal):
    transport=Transport(system='Execute one allowed action.',tool_schemas={n:s.model_json_schema()for n,s in TOOL_SCHEMAS.items()})
    command=transport.parse_response(response,'native')
    if command.tool!='train_model':
        raise ValueError('Replay requires an immediately admissible train_model; other actions are inconclusive')
    args=TOOL_SCHEMAS[command.tool].model_validate(command.as_dict()['arguments'])
    if args.pipeline.engine!='direct':raise ValueError('train_model must use direct engine')
    require_compatible(args.pipeline,task)
    resources=json.loads(body['messages'][-1]['content'])['core']['policies_resources']
    if args.budget_seconds+resources['reserve_seconds']>resources['remaining_seconds']:
        raise ValueError('New fit exceeds the remaining time after finalization reserve')
    if args.parent_id not in (None,'baseline'):raise ValueError('Unseen parent in the initial research state')
    normalized=args.pipeline.model_copy(update={'hyperparameters':effective_parameters(args.pipeline)})
    if normalized.hash==baseline.pipeline.hash:raise ValueError('Duplicate baseline pipeline')
    references=AgentReferences(ROOT,None)
    try:
        for identifier in args.method_refs:references.catalog_entry('methods',identifier)
        if any(not references.known_document(x)for x in args.reference_ids):raise ValueError('Wrong documentation ID namespace')
        if args.recipe_ref:references.catalog_entry('recipes',args.recipe_ref)
        if args.search_space_ref:raise ValueError('Search-space reference is owned by Optuna')
        wire=body['messages'][-1]['content']
        for ref in args.hypothesis.evidence_refs:
            if ref.get('id','') not in wire or ref.get('version','') not in wire:
                raise ValueError('Evidence identity/version was not present in the replay state')
            if ref.get('kind')=='knowledge':
                from automl_agent.knowledge_tools import KINDS
                doc=references.document(ref.get('id',''))
                if ref.get('version')!=references.version or not doc or doc['type']!=KINDS.get(ref.get('collection')):
                    raise ValueError('Invalid knowledge evidence')
            elif ref.get('kind')in {'observation','record'}:
                row=journal.execute('SELECT payload FROM cc_records WHERE record_id=?',(ref.get('id'),)).fetchone()
                if not row or json.loads(row[0])['source_version']!=ref.get('version'):raise ValueError('Invalid run evidence')
            else:raise ValueError('Invalid evidence kind')
    finally:references.close()
    return {'tool':command.tool,'model':args.pipeline.model,'budget_seconds':args.budget_seconds,
        'method_refs':args.method_refs,'reference_ids':args.reference_ids,
        'scope':'Typed arguments, compatibility, budget, ID namespaces and evidence; no fit or full dispatch executed'}


def run(args):
    load_dotenv(ROOT/'.env',override=False)
    args.output.mkdir(parents=True,exist_ok=False)
    store=d.Store(args.source)
    artifact=next(a for a in store.index()['artifacts'].values()if a.get('name')=='qwen_request.json'and a.get('identity')==args.identity)
    original=json.loads(gzip.decompress((store.root/artifact['path']).read_bytes()))
    assert original.get('tools') and original['stream'] and original['reasoning']['effort']=='xhigh'
    body,count=annotate(original)
    d.write(args.output/'request.json',body)
    d.write(args.output/'REGISTRATION.json',{'A':'Actual saved failed native command; no new generation',
        'A_identity':args.identity,'A_error':'pipeline and hypothesis encoded as strings',
        'B':'Same request with descriptions of four argument fields only','annotated_fields':count,
        'criterion':'Immediately admissible distinct direct train_model; invalid/inconclusive -> do not promote, no retries',
        'ML_quality_gain_claim':False,'source_request_sha256':hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()})
    journal_path=args.source/'regression/task/.agent_work/diagnostic/journal.sqlite'
    assert not Path(str(journal_path)+'-wal').exists() or Path(str(journal_path)+'-wal').stat().st_size==0
    with sqlite3.connect('file:'+str(journal_path.resolve())+'?mode=ro&immutable=1',uri=True)as journal:
        task=TaskSpec.model_validate(json.loads(journal.execute("SELECT value FROM metadata WHERE key='task'").fetchone()[0]))
        baseline=ExperimentSpec.model_validate_json(journal.execute("SELECT spec FROM experiments WHERE id='baseline'").fetchone()[0])
        if args.dry_run:
            print(json.dumps({'dry_run':True,'organizer_requests':0,'annotated_fields':count}));return
        key=d.limits.load_key(ROOT/'.env');d.register_secret(key)
        before=d.limits.snapshot(d.limits.fetch_info(key),key,'before');d.write(args.output/'api_limits_before.json',before)
        anchor=d.read(args.anchor)
        if (before['remaining_budget']or 0)<20 or before['budget_exhausted']or before['limits'].get('blocked'):
            raise RuntimeError('Verified budget unavailable')
        if anchor['limits']['budget_reset_at']!=before['limits']['budget_reset_at']:raise RuntimeError('Period changed; reconcile new400 allowance')
        result={'admissible':False,'upstream_outcome_unknown':False,'retries':0,'training_executed':False}
        started=time.monotonic();previous=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Probe wall limit')))
        try:
            signal.setitimer(signal.ITIMER_REAL,900)
            with httpx.Client(transport=httpx.HTTPTransport(retries=0),timeout=httpx.Timeout(900,connect=15))as client:
                with client.stream('POST',os.environ['OPENAI_BASE_URL'].rstrip('/')+'/chat/completions',
                        headers={'Authorization':'Bearer '+key},json=body)as response:
                    result['http_status']=response.status_code
                    if response.status_code!=200:response.read();response.raise_for_status()
                    def lines():
                        with gzip.open(args.output/'response.sse.gz','wb')as saved:
                            for line in response.iter_lines():saved.write(d.scrub(line+'\n'));yield line
                    value=completion_from_sse(lines())
            d.write(args.output/'response.json',value)
            result['usage']=value.get('usage');result['finish_reason']=value['choices'][0]['finish_reason']
            result['action']=validate_action(body,value,task,baseline,journal);result['admissible']=True
        except (httpx.TransportError,TimeoutError,IncompleteModelStream)as error:
            result.update(error=type(error).__name__,upstream_outcome_unknown=True)
        except Exception as error:result.update(error=type(error).__name__,detail=d.scrub(str(error)).decode()[:2000])
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
            result['elapsed_seconds']=time.monotonic()-started;d.write(args.output/'REPORT.json',result)
            after=d.limits.snapshot(d.limits.fetch_info(key),key,'after');after['comparison']=d.limits.compare(before,after)
            d.write(args.output/'api_limits_after.json',after)
            print(json.dumps({**result,'remaining_before':before['remaining_budget'],'remaining_after':after['remaining_budget'],
                'shared_interval_increase':after['comparison'].get('increase')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--identity',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--anchor',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');run(p.parse_args())
