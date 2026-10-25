"""Read-only gateway metadata and deterministic registration of the 100-unit series.

No chat/completions call, credential export or external-label access exists here.
Market USD prices and observed charges are not gateway-unit upper bounds.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL = 'qwen/qwen3.8-27b'
PUBLIC_BASES = {'https://changellenge.up.railway.app/v1', 'https://api.alfahack-gateway26.ru/v1'}


def inspect_metadata(value):
    """Allowlisted public pricing facts only; never store arbitrary API JSON."""
    rows = value.get('data', []) if isinstance(value, dict) else []
    if not isinstance(rows, list):
        rows = []
    findings = []
    for row in rows:
        if not isinstance(row, dict) or row.get('id', row.get('model_name')) != MODEL:
            continue
        info = row.get('model_info') if isinstance(row.get('model_info'), dict) else {}
        keys = ('input_cost_per_token', 'output_cost_per_token', 'billing_unit',
            'completion_includes_reasoning', 'fixed_request_cost', 'prompt', 'completion', 'currency')
        pricing = {}
        for container in (row, info, row.get('pricing'), info.get('pricing')):
            if not isinstance(container, dict):
                continue
            for key in keys:
                v = container.get(key)
                if type(v) in (int, float, bool):
                    pricing[key] = v
                elif isinstance(v, str):
                    if key == 'billing_unit':
                        pricing[key] = v if v == 'gateway_unit' else 'unconfirmed'
                    elif key == 'currency':
                        pricing[key] = v if v in {'USD', 'CNY', 'RUB'} else 'unconfirmed'
                    elif key in {'prompt', 'completion'}:
                        try:
                            pricing[key] = float(v)
                        except ValueError:
                            pass
        findings.append(pricing)
    return findings


def proof_issues(pricing):
    import math
    issues = []
    if pricing.get('billing_unit') != 'gateway_unit':
        issues.append('gateway_unit_mapping_unconfirmed')
    for key in ('input_cost_per_token', 'output_cost_per_token', 'fixed_request_cost'):
        rate = pricing.get(key)
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate < 0:
            issues.append(key + '_upper_bound_unknown')
    if pricing.get('completion_includes_reasoning') is not True:
        issues.append('reasoning_charge_bound_unknown')
    return issues


def registration():
    runs = []
    for index, task in enumerate(('adult', 'bank', 'bike', 'har')):
        for arm in (('A', 'B') if index % 2 == 0 else ('B', 'A')):
            runs.append({'task': task, 'arm': arm, 'status': 'pending_preflight',
                'seconds': 600, 'cpu': 2, 'memory_mb': 4096, 'max_calls': 10,
                'max_candidates': 4, 'gateway_unit_cap': 10,
                'outer_score': None, 'outer_labels_opened': False})
    return {'version': 1, 'model': MODEL, 'baseline_commit': '6b0c0cd',
        'series_cap': 100, 'initial_cap': 80, 'correction_reserve': 20,
        'seed': 42, 'sequential': True, 'datasets_known_to_development': True, 'runs': runs}


def run(output):
    import os
    import httpx
    from dotenv import dotenv_values
    from automl_agent.util import atomic_json
    config = {**dotenv_values(ROOT / '.env'), **os.environ}
    base = (config.get('OPENAI_BASE_URL') or 'https://changellenge.up.railway.app/v1').rstrip('/')
    key = config.get('OPENAI_API_KEY')
    if base not in PUBLIC_BASES or not key:
        raise ValueError('Known organizer base and local credential required; no values are printed')
    observations, proofs = [], []
    with httpx.Client(timeout=20, trust_env=False, follow_redirects=False) as client:
        for endpoint in ('/models', '/model/info'):
            try:
                response = client.get(base + endpoint, headers={'Authorization': 'Bearer ' + key})
                findings = inspect_metadata(response.json()) if response.status_code == 200 else []
                observations.append({'endpoint': endpoint, 'http_status': response.status_code, 'pricing': findings})
                proofs.extend(p for p in findings if not proof_issues(p))
            except (httpx.HTTPError, ValueError):
                observations.append({'endpoint': endpoint, 'status': 'metadata_unavailable'})
    result = registration()
    result.update(metadata=observations, pricing_upper_bound_confirmed=bool(proofs),
        real_generation_requests=0, outer_labels_opened=False)
    if proofs:
        result['pricing_proof'] = proofs[0]
        result['status'] = 'pricing_available_requires_guarded_runner'
    else:
        result['status'] = 'blocked_pricing_upper_bound_unknown'
        for row in result['runs']:
            row.update(status=result['status'], stop_reason='No confirmed request upper price in gateway units')
    atomic_json(output, result)
    print(json.dumps({'status': result['status'], 'generation_requests': 0,
        'metadata': observations}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.output)
