"""One counterfactual JSON-schema request; no tool execution or training."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import diagnostic_3h as d
from tools import hpo_policy_probe as probe
from tools.citation_interface_probe import controller,close
from automl_agent.contracts import Hypothesis,TOOL_SCHEMAS
from automl_agent.util import canonical
from context_compressor.transport import Transport,_omit_schema_titles


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=False)
    store=d.Store(args.source)
    artifacts=store.index()['artifacts'].values()
    req=next(r for r in artifacts if r.get('identity')==args.identity and r.get('name')=='qwen_request.json')
    original=json.loads(gzip.decompress((store.root/req['path']).read_bytes()))
    assert not original.get('tools') and original['response_format']=={'type':'json_object'}
    body=copy.deepcopy(original);context=json.loads(body['messages'][-1]['content'])
    before=copy.deepcopy(context['$defs']['Hypothesis']['properties']['evidence_refs'])
    field=Hypothesis.model_json_schema()['properties']['evidence_refs']
    _omit_schema_titles(field,True)
    context['$defs']['Hypothesis']['properties']['evidence_refs']=field
    body['messages'][-1]['content']=canonical(context)
    clone=args.output/'journal';clone.mkdir();shutil.copyfile(args.backup,clone/'journal.sqlite')
    c=controller(args.output,body)
    try:
        resources=context['core']['policies_resources']
        epoch=resources['deadline_epoch']-resources['remaining_seconds']
        future=[json.loads(raw) for raw, in c.journal.db.execute('SELECT record FROM operations')
            if json.loads(raw)['started_at']>epoch]
        with c.journal.db:
            c.journal.db.executemany('DELETE FROM operations WHERE id=?',[(r['operation_id'],) for r in future])
            c.journal.db.executemany('DELETE FROM experiments WHERE id=?',[(r['experiment_id'],) for r in future if r.get('experiment_id')])
        work=context['core']['current_work']
        c.journal.put_many({'state':'searching','leader_id':work['leader_id'],'baseline_id':work['baseline_id'],'protocol_revision':None})
        d.write(args.output/'request.json',body)
        d.write(args.output/'REGISTRATION.json',{'source_identity':args.identity,'source_request_sha256':req['sha256'],
            'A':'Actual train_model rejected for fabricated kindexperiment/idbaseline/version1/collectionnone',
            'B':'Exactly same JSON context/settings with only Hypothesis.evidence_refs schema changed',
            'criterion':'One complete admissible direct train_model under existing full controller checks; emptyoptionalrefs allowed; invalid/inconclusive =>DO_NOT_PROMOTE,no retries',
            'scope':'API contract/behavior evidence only, not causal quality gain; no tools/training executed',
            'schema_before':before,'schema_after':field,'request_sha256':d.digest(args.output/'request.json'),
            'production_archive':'v052 remains immutable; workspace schema prototype is not yet archived',
            'source_snapshot':'Consistent Linux SQLite backup; later operations/experiments excluded',
            'sources':{'automl_agent/contracts.py':d.digest(ROOT/'automl_agent/contracts.py')}})
    finally:close(c)


def validate(output,response):
    body=d.read(output/'request.json');c=controller(output,body)
    try:
        t=Transport(system='Validate one action.',tool_schemas={k:v.model_json_schema() for k,v in TOOL_SCHEMAS.items()})
        command=t.parse_response(response,'json')
        assert command.tool=='train_model','One immediate direct train_model is required by registration'
        args=TOOL_SCHEMAS[command.tool].model_validate(command.as_dict()['arguments'])
        assert args.pipeline.engine=='direct'
        resources=json.loads(body['messages'][-1]['content'])['core']['policies_resources']
        assert args.budget_seconds+resources['reserve_seconds']<=resources['remaining_seconds']
        for identifier in args.method_refs:c.references.catalog_entry('methods',identifier,c.task,c.plan)
        assert all(c.references.known_document(x) for x in args.reference_ids)
        if args.recipe_ref:c.references.catalog_entry('recipes',args.recipe_ref,c.task,c.plan)
        before={t:c.journal.db.execute('SELECT COUNT(*) FROM '+t).fetchone()[0] for t in ('operations','experiments')}
        c.context_memory.validate(command)
        assert before=={t:c.journal.db.execute('SELECT COUNT(*) FROM '+t).fetchone()[0] for t in before}
        d.write(output/'validated_command.json',command.as_dict())
        return {'tool':command.tool,'full_controller_admission':True,'evidence_refs':args.hypothesis.evidence_refs,
            'training_or_tool_execution_count':0,'quality_effect_claim':False}
    finally:close(c)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--identity');p.add_argument('--backup',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--anchor',type=Path);p.add_argument('--execute',action='store_true')
    args=p.parse_args();args.output=args.output.resolve()
    if args.execute:
        probe.validate=validate
        probe.execute(args)
    else:prepare(args)
