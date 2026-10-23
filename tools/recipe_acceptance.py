"""Execute every catalog recipe in one declared synthetic mode without downloads."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automl_agent.contracts import PipelineSpec
from automl_agent.util import atomic_json, read_json
from tools.offline_acceptance import case, environment


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    results = []
    for path in sorted((root / 'knowledge_runtime/recipes').glob('*.json')):
        recipe = read_json(path)
        pipeline = PipelineSpec.model_validate(recipe['pipeline'])
        kind = recipe['task_kinds'][0]
        result = case(output, recipe['id'], kind, pipeline, text=recipe['id'].startswith('text_'))
        result['declared_test_mode'] = kind
        result['recipe_hash'] = hashlib.sha256(path.read_bytes()).hexdigest()
        results.append(result)
        report = {'environment': environment(), 'results': results, 'passed': sum(r['passed'] for r in results),
            'total': len(results), 'scope': 'One declared synthetic mode per recipe; other combinations remain unverified',
            'all_passed': all(r['passed'] for r in results)}
        atomic_json(output / 'REPORT.json', report)
        print(json.dumps({k: result[k] for k in ('name', 'passed', 'candidate_status', 'candidate_error', 'control', 'refit')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    report = run(parser.parse_args().output)
    raise SystemExit(0 if report['all_passed'] else 1)
