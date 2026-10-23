"""Small development-only proxy check. No private task data or credentials in reports."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv
import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_agent.budget import Budget, Limits
from automl_agent.journal import Journal
from automl_agent.qwen import QwenClient, GENERATION


def run(output):
    load_dotenv(ROOT / '.env', override=False)
    before = json.loads((output / 'api_limits_before.json').read_text())
    key = os.environ['OPENAI_API_KEY']
    assert before['key_fingerprint'] == hashlib.sha256(key.encode()).hexdigest()[:16]
    assert not before.get('budget_exhausted') and not before['limits'].get('blocked')
    journal = Journal(output / 'probe')
    client = QwenClient(journal, Budget(journal, Limits(seconds=480), time.time()))
    tools = [{'type': 'function', 'function': {'name': 'report_path', 'description': 'Report the shortest path.',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'array', 'items': {'type': 'string'}},
            'cost': {'type': 'integer'}}, 'required': ['path', 'cost'], 'additionalProperties': False}}}]
    messages = [{'role': 'system', 'content': 'Solve the synthetic problem and call report_path once.'},
        {'role': 'user', 'content': 'Find the shortest path from A to F in this directed acyclic graph: '
            'A->B=4, A->C=2, B->D=3, C->B=1, C->D=7, C->E=6, D->E=1, D->F=5, E->F=2. '
            'Return the path and total cost.'}]
    report = {'cases': [], 'negative_controls': [], 'generation': GENERATION,
        'scope': 'Synthetic proxy check; no ML quality or three-hour run claim.'}
    try:
        for effort in ('medium', 'xhigh'):
            client.select_reasoning(effort, 'live_parameter_verification')
            started = time.monotonic()
            message = client._request(messages, tools)
            calls = message.get('tool_calls') or []
            answer = json.loads(calls[0]['function']['arguments']) if calls else None
            latest = json.loads(journal.db.execute("SELECT payload FROM events WHERE kind='qwen_response' ORDER BY rowid DESC LIMIT 1").fetchone()[0])
            report['cases'].append({'effort': effort, 'elapsed_seconds': time.monotonic() - started,
                'requested_output_tokens': client.output_tokens, 'answer': answer,
                'correct': answer == {'path': ['A', 'C', 'B', 'D', 'E', 'F'], 'cost': 9},
                'has_reasoning': bool(message.get('reasoning') or message.get('reasoning_content') or message.get('reasoning_details')),
                'reasoning_tokens': latest.get('reasoning_tokens'), 'model': latest.get('model')})
            (output / 'proxy_probe.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report['cases'][-1]), flush=True)
        # Rejection is stronger evidence of forwarding/validation than HTTP 200
        # alone; it still cannot expose the provider's applied sampling settings.
        for field, value in [('reasoning', {'enabled': True, 'effort': 'invalid_probe_effort'}), ('temperature', -1)]:
            payload = {'model': client.model, 'messages': [{'role': 'user', 'content': 'Reply OK.'}],
                'max_tokens': 16, 'stream': False, **client.extra, field: value}
            response = client.client.post(client.base_url + '/chat/completions', json=payload,
                headers={'Authorization': 'Bearer ' + key}, timeout=60)
            # Do not persist arbitrary server diagnostics or headers with credentials.
            text = response.text.lower()
            report['negative_controls'].append({'field': field, 'status': response.status_code,
                'mentions_field': field in text, 'mentions_invalid_value': str(value).lower() in text,
                'accepted_invalid_value': response.status_code == 200})
            print(json.dumps(report['negative_controls'][-1]), flush=True)
        report['server_usage'] = journal.get('token_usage')
        report['source_sha256'] = hashlib.sha256((ROOT / 'automl_agent/qwen.py').read_bytes()).hexdigest()
        report['applied_sampling_and_exact_effort_proven'] = False
        report['limitation'] = 'No server echo of effective sampling/effort. Reasoning output proves thinking; accepted effort and validation are recorded separately.'
    finally:
        (output / 'proxy_probe.json').write_text(json.dumps(report, indent=2))
        client.close()
        journal.close()


if __name__ == '__main__':
    run(Path(sys.argv[1]).resolve())
