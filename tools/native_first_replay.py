"""Offline first-operation admission for every recorded multi-call native reply."""
import ast
import copy
import gzip
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import diagnostic_3h as d
from tools.citation_interface_probe import controller,close
from tools.comparison_protocol import archived_runtime
from automl_agent.contracts import TOOL_SCHEMAS
from automl_agent.qwen_stream import completion_from_sse
from context_compressor.transport import Transport
import context_compressor.transport as transport_module


def run(output):
    audit=d.read(ROOT/'outputs/quality_goal_20261023/NATIVE_MULTICALL_AUDIT_01.json')
    cases=[(run['run'],item) for run in audit['runs'] for item in run['multiple_native_calls']]
    d.write(output/'REGISTRATION.json',{'scope':'Every multi-call response in fixed four-run audit; actual responses unchanged',
        'cases':len(cases),'criterion':'Oldv052 rejects all; candidate selects index0 only, unchanged rawresponse, full existing controller admission passes; no operation/note/experiment execution or registration',
        'paid_generation_requests':0,'training_executed':False,'quality_advantage_established':False})
    original=archived_runtime(ROOT/'dist/solution_v052.zip')['context_compressor/transport.py'].decode()
    cls=next(n for n in ast.parse(original).body if isinstance(n,ast.ClassDef) and n.name=='Transport')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='parse_response')
    namespace=vars(transport_module).copy();exec(ast.get_source_segment(original,method),namespace)
    old_parse=namespace['parse_response'];results=[]
    for run,item in cases:
        store=d.Store(ROOT/'outputs/quality_goal_20261023'/run)
        artifacts=store.index()['artifacts'].values()
        raw_request=next(a for a in artifacts if a.get('identity')==item['identity'] and a.get('name')=='qwen_request.json')
        raw_response=next(a for a in artifacts if a.get('identity')==item['identity'] and a.get('name')=='qwen_response.txt')
        body=json.loads(gzip.decompress((store.root/raw_request['path']).read_bytes()))
        response=completion_from_sse(gzip.decompress((store.root/raw_response['path']).read_bytes()).decode().splitlines())
        baseline=copy.deepcopy(response);context=json.loads(body['messages'][-1]['content'])
        backup=('v045class' if run=='stream_v045_full3h' else 'v049class' if run.startswith('cost_phase')
            else 'v052reg' if item['task']=='regression' else 'v052class')
        case=output/item['identity'];(case/'journal').mkdir(parents=True)
        shutil.copyfile(output/'backups'/(backup+'.sqlite'),case/'journal/journal.sqlite')
        c=controller(case,body)
        try:
            resources=context['core']['policies_resources'];epoch=resources['deadline_epoch']-resources['remaining_seconds']
            future=[json.loads(raw) for raw, in c.journal.db.execute('SELECT record FROM operations')
                if json.loads(raw)['started_at']>epoch]
            with c.journal.db:
                c.journal.db.executemany('DELETE FROM operations WHERE id=?',[(v['operation_id'],) for v in future])
                c.journal.db.executemany('DELETE FROM experiments WHERE id=?',[(v['experiment_id'],) for v in future if v.get('experiment_id')])
            work=context['core']['current_work']
            c.journal.put_many({'state':'searching','leader_id':work['leader_id'],'baseline_id':work['baseline_id'],'protocol_revision':None})
            t=Transport(system='Validate one first operation.',tool_schemas={k:v.model_json_schema() for k,v in TOOL_SCHEMAS.items()})
            try:old_parse(t,response,'native');old_rejected=False
            except ValueError:old_rejected=True
            first=t.parse_response(response,'native')
            before={k:c.journal.db.execute('SELECT COUNT(*) FROM '+k).fetchone()[0]
                for k in ['operations','experiments','cc_receipts','cc_commands','cc_notes']}
            c.context_memory.validate(first)
            assert before=={k:c.journal.db.execute('SELECT COUNT(*) FROM '+k).fetchone()[0] for k in before}
            assert response==baseline and old_rejected
            first_wire=response['choices'][0]['message']['tool_calls'][0]
            assert first.tool==first_wire['function']['name']
            result={'run':run,'identity':item['identity'],'native_call_count':item['calls'],'selected_index':0,
                'selected_tool':first.tool,'old_v052_rejected':old_rejected,'candidate_full_controller_admission':True,
                'raw_response_unchanged':True,'no_operation_experiment_receipt_note_writes':True,
                'additional_paid_requests':0,'discarded_calls_not_queued':True,'passed':True}
        except Exception as error:
            result={'run':run,'identity':item['identity'],'passed':False,'error_type':type(error).__name__,'error':str(error)}
        finally:close(c)
        d.write(case/'REPORT.json',result);results.append(result)
    report={'cases':results,'passed':len(results)==4 and all(r['passed'] for r in results),
        'training_executed':False,'paid_generation_requests':0,'quality_advantage_established':False}
    d.write(output/'REPORT.json',report);print(json.dumps(report))


if __name__=='__main__':run(Path(sys.argv[1]).resolve())
