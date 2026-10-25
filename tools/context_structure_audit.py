"""Read saved wire requests without API calls, mutation or raw context export."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def size(value):
    return len(canonical(value).encode())


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def audit(path):
    path = Path(path).resolve()
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        rows = connection.execute('SELECT wire_json FROM cc_requests ORDER BY rowid').fetchall()
    finally:
        connection.close()
    requests = []
    for index, (raw,) in enumerate(rows):
        wire = json.loads(raw)
        message = wire['messages'][1]['content']
        if isinstance(message, list):
            message = next(part['text'] for part in message if part['type'] == 'text')
        context = json.loads(message)
        system = wire['messages'][0]['content']
        tools = wire.get('tools', [])
        requests.append({'index': index, 'roles': [m['role'] for m in wire['messages']],
            'system_bytes': len(system.encode()), 'system_sha256': sha(system),
            'native_function_count': len(tools), 'logical_tool_count': len(tools) if tools else len(context.get('tool_argument_schemas', {})),
            'transport': 'native' if tools else 'json', 'tools_bytes': size(tools), 'tools_sha256': sha(tools),
            'context_bytes': size(context), 'context_blocks': {k: size(v) for k, v in context.items()},
            'core_blocks': {k: size(v) for k, v in context['core'].items()},
            'history_count': len(context.get('history', [])),
            'observation_count': len(context.get('observations', [])),
            'research_note_present': context['core'].get('research_note') is not None,
            'hypothesis_present': context['core'].get('current_work', {}).get('hypothesis') is not None,
            'compaction_stage': context.get('compaction', {}).get('stage')})
    return {'requests': requests, 'count': len(requests),
        'unique_systems': len({r['system_sha256'] for r in requests}),
        'unique_toolsets': len({r['tools_sha256'] for r in requests}),
        'only_snapshot_messages': all(r['roles'] == ['system', 'user'] for r in requests) if requests else None,
        'interpretation': 'bytes are not tokens; historical prepared requests need not all be accepted or billed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.journal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print({k: v for k, v in result.items() if k != 'requests'})
