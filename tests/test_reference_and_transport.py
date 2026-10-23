import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from threading import Thread
import time

import pytest

from automl_agent.budget import Budget,Limits
from automl_agent.contracts import ReferenceArgs
from automl_agent.journal import Journal
from automl_agent.qwen import QwenClient, json_tool_schemas, parse_json_object
from automl_agent.qwen import QwenError
from automl_agent.capabilities import CapabilityRegistry
from automl_agent.knowledge import AgentReferences
from automl_agent.util import read_json

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('text',[
    '{"operation_id":"one","tool":"finish","arguments":{"reason":"done"}}',
    'I will preserve the valid baseline.\n\n{"operation_id":"one","tool":"finish","arguments":{"reason":"done"}}',
    '```json\n{"operation_id":"one","tool":"finish","arguments":{"reason":"done"}}\n```',
])
def test_single_command_can_have_a_narrative_preamble(text):
    from automl_agent.contracts import ToolCommand
    command=ToolCommand.model_validate(parse_json_object(text))
    assert command.operation_id=='one' and command.tool=='finish'


@pytest.mark.parametrize('text',['{} {}','[{"tool":"finish"}]','{"tool":"finish"} trailing prose','No command'])
def test_ambiguous_or_incomplete_command_messages_are_rejected(text):
    with pytest.raises(ValueError):parse_json_object(text)


def test_compact_json_schemas_preserve_every_reference():
    from automl_agent.contracts import TOOL_SCHEMAS
    schemas,definitions=json_tool_schemas()
    for name,model in TOOL_SCHEMAS.items():
        original=model.model_json_schema()
        for identifier,value in original.pop('$defs',{}).items():assert definitions[identifier]==value
        assert schemas[name]==original
    expanded={name:model.model_json_schema() for name,model in TOOL_SCHEMAS.items()}
    assert len(json.dumps({'schemas':schemas,'$defs':definitions}))<len(json.dumps(expanded))*.7


def test_reference_integrity_symbols_version_and_cursor(tmp_path):
    with AgentReferences(ROOT,CapabilityRegistry(tmp_path/'caps.json')) as s:
        assert s.verify(ROOT/'data_for_participants/requirements.txt')['status']=='verified'
        r=s.query(ReferenceArgs(mode='api',query='lightgbm.LGBMRanker.fit',max_chars=2000))
        doc=r['entry']['id']
        assert 'lightgbm.LGBMRanker.fit' in s.document(doc)['symbols']
        a=s.query(ReferenceArgs(mode='read',reference_id=doc,max_chars=2000))
        assert isinstance(a['next_cursor'],str)
        b=s.query(ReferenceArgs(mode='read',reference_id=doc,cursor=a['next_cursor'],max_chars=2000))
        assert a['entry']['content']!=b['entry']['content'] and a['entry']['version']=='4.6.0'
        assert s.query(ReferenceArgs(mode='api',query='sklearn.impute.SimpleImputer',library='scikit-learn',version='9.0'))['status']=='missing_version'


def test_fallback_without_fts(tmp_path):
    with AgentReferences(ROOT,CapabilityRegistry(tmp_path/'caps.json')) as s:
        s.verify(ROOT/'data_for_participants/requirements.txt')
        s.use_fts=False
        r=s.query(ReferenceArgs(query='sklearn.impute.IterativeImputer',max_chars=20000))
        assert any(e['id']=='api_sklearn_impute_IterativeImputer' for e in r['entries'])


@pytest.mark.parametrize('native_supported',[True,False])
def test_native_and_json_share_command_schema(tmp_path,monkeypatch,native_supported):
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));requests.append(body)
            if 'tools' in body and not native_supported:
                self.send_response(400);self.end_headers();return
            command={'operation_id':'stable_1','tool':'finish','arguments':{'reason':'fixture','result_ids':['baseline'], 'imputation_skip_reason':'fixture'}}
            message={'role':'assistant','content':json.dumps(command)}
            if 'tools' in body:
                message={'role':'assistant','content':None,'tool_calls':[{'id':'call_1','type':'function','function':{'name':'finish',
                    'arguments':json.dumps({'operation_id':'stable_1',**command['arguments']})}}]}
            payload=json.dumps({'choices':[{'message':message,'finish_reason':'tool_calls' if 'tools' in body else 'stop'}],
                'usage':{'prompt_tokens':10,'completion_tokens':3,'total_tokens':13}}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(payload)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    monkeypatch.setenv('OPENAI_API_KEY','fixture-key-not-a-credential');monkeypatch.setenv('OPENAI_MODEL','fixture-model')
    monkeypatch.setenv('OPENAI_BASE_URL',f'http://127.0.0.1:{server.server_port}');monkeypatch.delenv('OPENAI_EXTRA_BODY',raising=False)
    j=Journal(tmp_path);q=QwenClient(j,Budget(j,Limits(seconds=120),time.time()))
    try:
        command,_=q.next_command('fixture',{'state':'fixture'})
        assert command.operation_id=='stable_1' and command.arguments['reason']=='fixture'
        assert j.get('transport')==('native' if native_supported else 'json')
        assert j.get('llm_calls')==(1 if native_supported else 2)
        assert j.get('token_usage')['total_tokens']==13
        assert requests[0]['tool_choice']=='required' and requests[0]['parallel_tool_calls'] is False
    finally:q.close();j.close();server.shutdown();server.server_close()


@pytest.mark.parametrize('model',[None,'','   '])
def test_official_two_variable_environment_runs_research(tmp_path,monkeypatch,model):
    from automl_agent.controller import Controller
    from conftest import write_task
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            command={'operation_id':'two_variables_finish','tool':'finish',
                'arguments':{'reason':'Measured baseline is sufficient for this transport check',
                    'result_ids':['baseline'], 'imputation_skip_reason':'Fixed transport check; no new imputation hypothesis'}}
            payload=json.dumps({'choices':[{'message':{'role':'assistant','content':json.dumps(command)},
                'finish_reason':'stop'}]}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(payload)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    Thread(target=server.serve_forever,daemon=True).start()
    monkeypatch.setenv('OPENAI_API_KEY','fixture-key-not-a-credential')
    monkeypatch.setenv('OPENAI_BASE_URL',f'http://127.0.0.1:{server.server_port}')
    monkeypatch.delenv('OPENAI_EXTRA_BODY',raising=False)
    if model is None:monkeypatch.delenv('OPENAI_MODEL',raising=False)
    else:monkeypatch.setenv('OPENAI_MODEL',model)
    controller=Controller(write_task(tmp_path/'task'),Limits(seconds=60,cpu=1),transport='json')
    try:
        assert controller.qwen.configured
        report=controller.run()
        assert report['state']=='completed' and report['publication'] is not None
        assert len(requests)==1 and requests[0]['model']=='qwen/qwen3.8-27b'
        decisions=controller.journal.db.execute("SELECT payload FROM events WHERE kind='qwen_decision'").fetchall()
        assert len(decisions)==1 and json.loads(decisions[0][0])['command']['operation_id']=='two_variables_finish'
    finally:controller.close();server.shutdown();server.server_close()


def test_total_model_deadline_interrupts_a_busy_response_and_preserves_accounting(tmp_path,monkeypatch):
    import signal
    monkeypatch.setenv('OPENAI_API_KEY','fixture-key-not-a-credential')
    monkeypatch.setenv('OPENAI_MODEL','fixture-model')
    monkeypatch.setenv('OPENAI_BASE_URL','https://example.invalid')
    monkeypatch.delenv('OPENAI_EXTRA_BODY',raising=False)
    journal=Journal(tmp_path);client=QwenClient(journal,Budget(journal,Limits(seconds=60,max_llm_seconds=.1),time.time()))
    previous=signal.getsignal(signal.SIGALRM)
    import httpx
    class Busy(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"choices":[]}\n\n'
            time.sleep(2)
    client.client.close()
    client.client=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,
        headers={'content-type':'text/event-stream'},stream=Busy())))
    started=time.monotonic()
    try:
        with pytest.raises(QwenError,match='wall-clock'):
            client._request([{'role':'user','content':'fixture'}])
        assert time.monotonic()-started<.8
        assert journal.get('llm_inflight') is None and journal.get('llm_seconds')>=.09
        assert signal.getsignal(signal.SIGALRM)==previous and signal.getitimer(signal.ITIMER_REAL)==(0.,0.)
        with pytest.raises(TimeoutError,match='time budget'):
            journal_budget=client.budget; journal_budget.claim_call()
    finally:client.close();journal.close()
