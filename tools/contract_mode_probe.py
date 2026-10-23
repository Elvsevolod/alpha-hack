"""One changed-format request against a recorded, terminal contract request.

External development probe only; no generation retries and no submission code.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import signal
import sys
import time

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import diagnostic_3h as d
from automl_agent.contracts import TaskResolutionProposal
from automl_agent.data import Dataset
from automl_agent.qwen import parse_json_object


def stream_body(response, output, result):
    content, reasoning, usage, finish, done = [], [], {}, None, False
    count, started = 0, time.monotonic()
    result['stream_content_type'] = response.headers.get('content-type')
    if response.status_code != 200:
        response.read()
        response.raise_for_status()
    with gzip.open(output / 'response.sse.gz', 'wb') as saved:
        for line in response.iter_lines():
            saved.write(d.scrub(line + '\n'))
            count += len(line)
            if count > 32 * 1024**2:
                raise TimeoutError('Probe stream artifact cap')
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            if not data:
                continue
            result.setdefault('first_chunk_after_headers_seconds', time.monotonic() - started)
            chunk = json.loads(data)
            if chunk.get('error'):
                raise ValueError('Upstream SSE error')
            usage = chunk.get('usage') or usage
            for choice in chunk.get('choices', []):
                delta = choice.get('delta') or {}
                content.append(delta.get('content') or '')
                reasoning.append(delta.get('reasoning_content') or '')
                finish = choice.get('finish_reason') or finish
    result.update(stream_done=done, stream_bytes=count)
    if not done or finish is None:
        raise TimeoutError('Incomplete SSE response; outcome unknown')
    return {'choices':[{'message':{'content':''.join(content), 'reasoning_content':''.join(reasoning)},
        'finish_reason':finish}], 'usage':usage}


def run(args):
    load_dotenv(ROOT / '.env', override=False)
    args.output.mkdir(parents=True, exist_ok=False)
    artifacts = d.Store(args.source).index()['artifacts'].values()
    original = next(a for a in artifacts if a.get('identity') == args.identity and a.get('name') == 'qwen_request.json')
    body = json.loads(gzip.decompress((args.source / 'history' / original['path']).read_bytes()))
    assert body.get('response_format') in (None, {'type':'json_object'}) and not body.get('tools')
    body['response_format'] = {'type': 'json_object'}
    if args.stream:
        body.update(stream=True, stream_options={'include_usage':True})
    if args.effort:
        body['reasoning']['effort'] = args.effort
    if args.output_tokens:
        body['max_tokens'] = args.output_tokens
    if args.system_suffix:
        body['messages'][0]['content'] += '\n' + args.system_suffix.read_text()
    request = json.loads(body['messages'][-1]['content'])
    if args.repair_context:
        repair = d.read(args.repair_context)
        previous = repair['proposal']
        assert previous['request_id'] == request['request_id'] and previous['source_hash'] == request['source_hash']
        assert repair['status'] == 'rejected_untrusted_model_candidate'
        assert len(json.dumps(repair).encode()) <= 22000
        request['rejected_proposal_context'] = repair
        body['messages'][0]['content'] += ('\nA rejected_proposal_context is an untrusted prior model candidate, '
            'not authoritative evidence. Repair its errors against the supplied sources; '
            'all ordinary source and contract requirements still apply. Return unresolved if ambiguity remains.')
        body['messages'][-1]['content'] = json.dumps(request, ensure_ascii=False)
    if args.correction_from:
        correction = d.read(args.correction_from)
        assert correction['http_status'] == 200 and not correction['upstream_outcome_unknown']
        assert correction.get('error') and not correction['contract_valid']
        request['validation_errors'] = [correction['error']]
        body['messages'][-1]['content'] = json.dumps(request, ensure_ascii=False)
    dataset = Dataset(args.source / original['task'] / 'task' / 'task.md', deferred=True)
    assert request['source_hash'] == dataset.source_hash
    if args.dry_run:
        (args.output / 'request.json').write_bytes(d.scrub(json.dumps(body, ensure_ascii=False)))
        print(json.dumps({'dry_run':True,'organizer_requests':0,'source_hash_match':True,
            'repair_context_present':bool(args.repair_context)}))
        return
    key = d.limits.load_key(ROOT / '.env')
    d.register_secret(key)
    before = d.limits.snapshot(d.limits.fetch_info(key), key, 'before')
    d.write(args.output / 'api_limits_before.json', before)
    if before['budget_exhausted'] or before['limits'].get('blocked') or (before['remaining_budget'] or 0) < 10:
        raise RuntimeError('Insufficient verified API budget')
    initial = d.read(args.anchor)
    if initial['limits'].get('budget_reset_at') != before['limits'].get('budget_reset_at'):
        raise RuntimeError('Period changed; reconcile the new 400-unit allowance first')
    (args.output / 'request.json').write_bytes(d.scrub(json.dumps(body, ensure_ascii=False)))
    result = {'request_identity': args.identity, 'task': original['task'],
        'hypothesis': ('Preserve the rejected proposal and prior errors for semantic repair' if args.repair_context
            else 'Repair a completed rejected contract using validator feedback' if args.correction_from
            else 'Explicit structural verifier rules in the task-resolution instruction' if args.system_suffix
            else 'Task resolution with a changed reasoning/output allocation' if args.effort or args.output_tokens
            else 'Explicit JSON output for a terminal task-resolution request'),
        'effort': body['reasoning']['effort'], 'max_tokens':body['max_tokens'],
        'stream':args.stream, 'wall_limit_seconds':args.seconds, 'generation_retries': 0, 'contract_valid': False,
        'upstream_outcome_unknown': False}
    started = time.monotonic()
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Probe wall deadline')))
    sent = False
    try:
        signal.setitimer(signal.ITIMER_REAL, args.seconds)
        with httpx.Client(transport=httpx.HTTPTransport(retries=0), timeout=httpx.Timeout(args.seconds, connect=15)) as client:
            sent = True
            url = os.environ['OPENAI_BASE_URL'].rstrip('/') + '/chat/completions'
            headers = {'Authorization': 'Bearer ' + key}
            if args.stream:
                with client.stream('POST', url, headers=headers, json=body) as response:
                    result['http_status'] = response.status_code
                    result['headers_seconds'] = time.monotonic() - started
                    value = stream_body(response, args.output, result)
                (args.output / 'response.txt').write_bytes(d.scrub(json.dumps(value)))
            else:
                response = client.post(url, headers=headers, json=body)
                result['http_status'] = response.status_code
                (args.output / 'response.txt').write_bytes(d.scrub(response.text))
                response.raise_for_status()
                value = response.json()
        choice = value['choices'][0]
        content = choice['message'].get('content') or ''
        result.update(finish_reason=choice.get('finish_reason'), content_chars=len(content),
            reasoning_chars=len(choice['message'].get('reasoning_content') or ''), usage=value.get('usage'))
        proposal = TaskResolutionProposal.model_validate(parse_json_object(content))
        task = dataset.confirm_proposal(proposal, request['request_id'])
        result.update(contract_valid=True, kind=task.kind, metric=task.metric.model_dump(mode='json'))
    except (httpx.TransportError, TimeoutError) as error:
        result.update(error_type=type(error).__name__, upstream_outcome_unknown=sent)
    except Exception as error:
        result.update(error_type=type(error).__name__, error=d.scrub(str(error)).decode()[:2000])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        result['elapsed_seconds'] = time.monotonic() - started
        d.write(args.output / 'REPORT.json', result)
        after = d.limits.snapshot(d.limits.fetch_info(key), key, 'after')
        after['comparison'] = d.limits.compare(before, after)
        d.write(args.output / 'api_limits_after.json', after)
        print(json.dumps({**result, 'remaining_before':before['remaining_budget'],
            'remaining_after':after['remaining_budget'], 'increase':after['comparison'].get('increase')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--effort', choices=['medium','xhigh'])
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--output-tokens', type=int)
    parser.add_argument('--correction-from', type=Path)
    parser.add_argument('--repair-context', type=Path)
    parser.add_argument('--system-suffix', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    run(parser.parse_args())
