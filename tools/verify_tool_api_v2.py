"""Offline search and integrated-context evidence for the API v2 candidate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automl_agent.budget import Limits
from automl_agent.contracts import KnowledgeReadArgs, KnowledgeSearchArgs, TOOL_SCHEMAS, TOOL_DESCRIPTIONS
from automl_agent.controller import Controller, SYSTEM
from automl_agent.tool_results import RESULT_SCHEMAS
from automl_agent.util import atomic_json, canonical, digest, file_hash
from context_compressor import ProfileCache, ProfileKey
from context_compressor.verify import _measure
from ml_knowledge.runtime import ReferenceArgs, ReferenceStore
from tools.offline_acceptance import fixture

ROOT = Path(__file__).resolve().parents[1]


def search(c, output):
    frozen = output / 'search_heldout24.json'
    cases = json.loads(frozen.read_text())['cases']
    results = []
    for strategy in ('auto', 'fts', 'lexical'):
        outcomes = []
        for case in cases:
            requested = case['strategy'] if strategy == 'auto' else strategy
            args = KnowledgeSearchArgs(query=case['query'], filters=case['filters'], strategy=requested, limit=5, max_chars=12000)
            result = c.knowledge_tools.discover(args)
            entries = list(result.get('entries', []))
            while result.get('next_cursor') and len(entries) < 5:
                args = args.model_copy(update={'cursor': result['next_cursor'], 'limit': 5 - len(entries)})
                result = c.knowledge_tools.discover(args)
                entries.extend(result.get('entries', []))
            ids = [entry['id'] for entry in entries[:5]]
            rank = next((i + 1 for i, identifier in enumerate(ids) if identifier in case['allowed_ids']), None)
            critical = None
            if case['critical'] and rank:
                entry = entries[rank - 1]
                read_args = KnowledgeReadArgs(**entry['read']['arguments'], max_chars=6000)
                chunks, ranges, versions = [], [], []
                for _ in range(1000):
                    read = c.knowledge_tools.read(read_args)
                    doc = read.get('entry', {})
                    chunks.append(doc.get('content', ''))
                    ranges.append(read.get('range'))
                    versions.append(doc.get('version'))
                    if not read.get('next_cursor'):
                        break
                    assert read['complete'] is False and read['continuation']
                    read_args = KnowledgeReadArgs(**read['continuation']['arguments'])
                joined = ''.join(chunks)
                expected = c.references.document(entry['id'])['text']
                critical = {'required_terms': {term: term in joined for term in case['required_terms']},
                    'all_source_text_preserved': joined == expected, 'ranges': ranges,
                    'version_observed': sorted(set(versions)), 'version_expected': case['versions'][entry['id']],
                    'complete': read.get('complete'), 'pages': len(chunks)}
            positive = case['expected'] == 'positive'
            if positive:
                passed = rank is not None
            elif case['expected'] == 'no_matches':
                passed = result['status'] == 'no_matches' and not entries
            elif case['expected'] == 'missing_version':
                passed = result['status'] == 'missing_version' and not entries
            else:
                passed = all(not entry['answer_confirmed'] for entry in entries)
            outcomes.append({'id': case['id'], 'group': case['group'], 'query': case['query'], 'strategy': requested,
                'expected': case['expected'], 'status': result['status'], 'rank': rank, 'passed': passed,
                'returned_ids': ids, 'entries': entries, 'critical_read': critical})
        positives = [row for row in outcomes if row['expected'] == 'positive']
        groups = {group: sum(row['rank'] is not None for row in positives if row['group'] == group)
            for group in sorted({row['group'] for row in positives})}
        criticals = [row for row in outcomes if next(case for case in cases if case['id'] == row['id'])['critical']]
        critical_pass = all(row['critical_read'] and row['critical_read']['all_source_text_preserved']
            and all(row['critical_read']['required_terms'].values())
            and row['critical_read']['version_observed'] == [row['critical_read']['version_expected']] for row in criticals)
        hits = sum(row['rank'] is not None for row in positives)
        mrr = sum(1 / row['rank'] if row['rank'] else 0 for row in positives) / len(positives)
        negative_pass = all(row['passed'] for row in outcomes if row['expected'] != 'positive')
        results.append({'strategy': strategy, 'hit_at_5': hits, 'positive_count': len(positives), 'mrr_at_5': mrr,
            'group_hits': groups, 'critical_reads_pass': critical_pass, 'negative_checks_pass': negative_pass,
            'thresholds_pass': hits >= 14 and min(groups.values()) >= 3 and mrr >= .75 and critical_pass and negative_pass,
            'outcomes': outcomes})
    legacy = []
    for query, old_id in json.loads((ROOT / 'ml_knowledge/evaluation/legacy40.json').read_text()):
        expected = c.references.document(old_id)['id']
        baseline = ReferenceStore.query(c.references, ReferenceArgs(mode='search', query=query, limit=5, max_chars=12000))
        old_ids = [entry['id'] for entry in baseline.get('entries', [])]
        args = KnowledgeSearchArgs(query=query, limit=5, max_chars=12000)
        result = c.knowledge_tools.discover(args)
        new_ids = [entry['id'] for entry in result.get('entries', [])]
        legacy.append({'query': query, 'expected_id': expected, 'baseline_ids': old_ids, 'api2_ids': new_ids,
            'baseline_hit': expected in old_ids, 'api2_hit': expected in new_ids})
    report = {'frozen_sha256': file_hash(frozen), 'corpus_version': c.references.version, 'strategies': results,
        'legacy': {'cases': legacy, 'baseline_hits': sum(row['baseline_hit'] for row in legacy),
            'api2_hits': sum(row['api2_hit'] for row in legacy),
            'lost_successes': [row['query'] for row in legacy if row['baseline_hit'] and not row['api2_hit']]},
        'rankings_not_tuned_after_evaluation': True}
    atomic_json(output / 'search_final.json', report)
    return {item['strategy']: {key: value for key, value in item.items() if key != 'outcomes'} for item in results} | {'legacy': {key: value for key, value in report['legacy'].items() if key != 'cases'}}


def context(c, output):
    cases, inserted = [], 0
    spec, result = c.lookup('baseline')
    for count in (100, 1000, 10000):
        with c.journal.db:
            for i in range(inserted, count):
                identifier = 'history_' + str(i)
                current = spec.model_copy(update={'experiment_id': identifier})
                outcome = result.model_copy(update={'experiment_id': identifier, 'status': 'unsupported_mode',
                    'error': 'Historical verification diagnostic with a versioned permitted source', 'score': None})
                c.journal.db.execute('INSERT INTO experiments VALUES (?,?,?)',
                    (identifier, current.model_dump_json(), outcome.model_dump_json()))
        inserted = count
        for mode in ('native', 'json'):
            def build():
                snapshot = c.context_memory.snapshot(SYSTEM)
                return c.context_memory.compressor.prepare(snapshot, mode, budget=131072)
            for _ in range(3):
                build()
            measures = [_measure(build) for _ in range(30)]
            times = sorted(item['seconds'] for item in measures)
            rss = max(item['sampled_additional_rss_bytes'] for item in measures)
            peak = max(item['python_peak_bytes'] for item in measures)
            p95 = times[math.ceil(.95 * len(times)) - 1]
            item = {'records': count, 'mode': mode, 'samples': len(times), 'p95_seconds': p95, 'maximum_seconds': max(times),
                'median_seconds': statistics.median(times), 'additional_rss_bytes': rss, 'python_peak_bytes': peak,
                'thresholds_pass': p95 <= .5 and max(times) <= 2 and max(rss, peak) <= 64 * 1024**2,
                'overhead_bytes': c.context_memory.compressor.transport.overhead_bytes(mode), 'measurements': measures}
            cases.append(item)
            atomic_json(output / 'context_performance.json', {'cases': cases, 'complete': False})
            print(json.dumps({key: value for key, value in item.items() if key != 'measurements'}), flush=True)
    import numpy as np
    import pandas as pd
    from automl_agent.data import Dataset
    # Actual Dataset.profile implementation on a permitted 256 x 10000 fit frame.
    wide = object.__new__(Dataset)
    columns = ['wide_' + str(i) for i in range(10000)]
    matrix = np.arange(256 * 10000, dtype=float).reshape(256, 10000) % 29
    matrix[::23] = np.nan
    wide.train = pd.DataFrame(matrix, columns=columns)
    wide.test = wide.train.iloc[:8].copy()
    wide.task = c.task.model_copy(update={'feature_columns': columns, 'categorical_columns': [], 'text_columns': []})
    wide.source_hash = digest('wide-api2-profile')
    calls = []
    def calculate():
        calls.append(1)
        return wide.profile(list(range(256))).model_dump(mode='json')
    cache = ProfileCache(c.context_memory.store)
    key = ProfileKey(wide.source_hash, digest(list(range(256))))
    started = time.perf_counter()
    profile = cache.get_or_create(key, calculate)
    cold = time.perf_counter() - started
    for _ in range(4):
        assert cache.get_or_create(key, calculate) == profile
    report = {'cases': cases, 'complete': True, 'cold_profile': {'rows': 256, 'features': 10000,
        'seconds': cold, 'calculations': len(calls), 'thresholds_pass': cold <= 30 and len(calls) == 1},
        'rss_method': '1 ms sampling plus tracemalloc; sampled RSS is not a hard peak guarantee',
        'warm_fraction_of_live_run': 'evaluated separately from live request traces',
        'all_thresholds_pass': all(row['thresholds_pass'] for row in cases) and cold <= 30 and len(calls) == 1}
    atomic_json(output / 'context_performance.json', report)
    return {'all_thresholds_pass': report['all_thresholds_pass'], 'cold_profile': report['cold_profile']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('check', choices=['search', 'context', 'registry'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.check == 'registry':
        atomic_json(args.output / 'api_registry.json', {name: {'description': TOOL_DESCRIPTIONS[name],
            'arguments': model.model_json_schema(), 'result': RESULT_SCHEMAS[name].model_json_schema()} for name, model in TOOL_SCHEMAS.items()})
        return
    with tempfile.TemporaryDirectory(prefix='api2-verification-') as temporary:
        c = Controller(fixture(Path(temporary)), Limits(seconds=7200, cpu=1), run_id='verification_api2')
        try:
            c.prepare()
            print(json.dumps((search if args.check == 'search' else context)(c, args.output)), flush=True)
        finally:
            c.close()


if __name__ == '__main__':
    main()
