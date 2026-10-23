import json

from tools.diagnostic_store import Store
from tools.goal_quality_run import decode_response
from tools.test_diagnostic_3h import gateway, post


def test_external_gateway_preserves_sse_bytes_type_and_usage(tmp_path):
    raw = ('data: '+json.dumps({'choices':[{'index':0,'delta':{'content':'{}'},'finish_reason':'stop'}]})
        +'\n\ndata: '+json.dumps({'choices':[],'usage':{'prompt_tokens':4,'completion_tokens':3}})
        +'\n\ndata: [DONE]\n\n').encode()
    def upstream(handler, body, count):
        handler.send_response(200)
        handler.send_header('Content-Type','text/event-stream; charset=utf-8')
        handler.send_header('Content-Length',str(len(raw)))
        handler.end_headers();handler.wfile.write(raw)
    with gateway(tmp_path/'complete',upstream) as (server,guard,calls):
        server.response_decoder=decode_response
        result=post(server,stream=True)
        assert result.status_code==200 and result.content==raw
        assert result.headers['content-type']=='text/event-stream; charset=utf-8'
        saved=[r for r in Store(guard.directory).records() if r['kind']=='response_saved']
        assert saved[0]['payload']['usage']['completion_tokens']==3
        assert not server.unknown and len(calls)==1


def test_external_gateway_closes_both_agents_after_partial_sse(tmp_path):
    raw=b'data: {"choices":[{"index":0,"delta":{"content":"{}"},"finish_reason":"stop"}]}\n\n'
    def upstream(handler,body,count):
        handler.send_response(200);handler.send_header('Content-Type','text/event-stream')
        handler.send_header('Content-Length',str(len(raw)))
        handler.end_headers();handler.wfile.write(raw)
    with gateway(tmp_path/'partial',upstream) as (server,guard,calls):
        server.response_decoder=decode_response
        assert post(server,stream=True).status_code==402
        assert server.unknown and guard.reason=='upstream_outcome_unknown'
        assert post(server,task='classification',stream=True).status_code==402
        assert len(calls)==1


def test_external_gateway_preserves_known_provider_failure_without_retry_or_global_close(tmp_path):
    raw=b'data: {"error":{"code":"502","message":"provider unavailable"}}\n\n'
    def upstream(handler,body,count):
        handler.send_response(200);handler.send_header('Content-Type','text/event-stream')
        handler.send_header('Content-Length',str(len(raw)))
        handler.end_headers();handler.wfile.write(raw)
    with gateway(tmp_path/'known',upstream) as (server,guard,calls):
        server.response_decoder=decode_response
        result=post(server,stream=True)
        assert result.status_code==200 and result.content==raw
        assert not server.unknown and not guard.closed and len(calls)==1
        assert post(server,task='classification',stream=True).content==raw
        assert len(calls)==2
