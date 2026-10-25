"""Independently verify the frozen cost study and replay its selected inner models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def verify(directory, source_root, baseline_source=None):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score
    sys.path.insert(0, str(source_root.resolve()))
    from automl_agent.contracts import MetricSpec, OutputSpec, PipelineSpec, TaskSpec
    from automl_agent.pipelines import DirectPipeline
    from automl_agent.validation import inner_split

    freeze = json.loads((directory / 'freeze.json').read_text())
    driver = Path(__file__).with_name('measure_hgb_snapshot_cost.py')
    assert freeze['script_sha256'] == hashlib.sha256(driver.read_bytes()).hexdigest()
    state = json.loads((directory / 'state.json').read_text())
    runs = json.loads((directory / 'runs.json').read_text())
    assert state['completed_pairs'] == state['expected_pairs'] == 12
    assert state['source_unchanged'] and state['script_unchanged']
    assert len(runs) == 12 and all(len(run['phases']) == 2 for run in runs)
    replay_scores, results = {}, []
    for case, config in freeze['cases'].items():
        observed = [run for run in runs if run['case'] == case]
        assert [run['source'] for run in observed] == freeze['order']
        recipes, calls = [], []
        for run in observed:
            root = directory / f"{case}_{run['order']}_{run['source']}"
            recipe = json.loads((root / 'select' / 'frozen.json').read_text())
            assert recipe == json.loads((root / 'refit' / 'frozen.json').read_text())
            recipes.append(recipe)
            calls.append([[(c['rows'], c['max_iter'], c['n_iter']) for c in phase['native_fit_calls']]
                for phase in run['phases']])
            for phase in run['phases']:
                assert phase['status'] == 'completed' and phase['pickle_prediction_identity_verified']
                assert phase['native_iterations'] == phase['frozen_iterations']['answer']
                assert phase['full_process_wall_seconds'] >= phase['fit_seconds'] + phase['prediction_seconds'] + phase['save_load_verify_seconds']
                root_path = source_root.resolve() if run['source'] == 'candidate' else (baseline_source.resolve() if baseline_source else Path(phase['source_root']))
                expected = freeze['source_sha256'][run['source']]
                assert expected == {str(p.relative_to(root_path)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((root_path / 'automl_agent').glob('*.py'))}
        assert all(recipe == recipes[0] for recipe in recipes)
        assert all(c == calls[0] for c in calls)
        # Reconstruct inputs independently of the measured runner. Only inner
        # fit labels are passed to this fresh frozen model, not held-out labels.
        n, width = config['rows'], config['features']
        rng = np.random.default_rng(freeze['seed'])
        raw = rng.normal(size=(n + n // 4 + 1000, width))
        classification = config['kind'] == 'classification'
        if classification:
            target = np.argmax(raw[:, :10] + rng.normal(scale=.6, size=(len(raw), 10)), axis=1)
            classes, metric = {'answer':list(reversed(range(10)))}, MetricSpec(name='f1', average='macro')
        else:
            target = 2 * raw[:, 0] + np.sin(raw[:, 1]) + raw[:, 2] * raw[:, 3] + rng.normal(scale=.5, size=len(raw))
            classes, metric = {}, MetricSpec(name='rmse')
        raw[rng.random(raw.shape) < .02] = np.nan
        X = pd.DataFrame(raw[:n], columns=[f'x{i}' for i in range(width)], index=[f'r{i}' for i in range(n)])
        y = pd.DataFrame({'answer':target[:n]}, index=X.index)
        task = TaskSpec(kind=config['kind'], targets=['answer'], feature_columns=list(X), classes=classes,
            metric=metric, output=OutputSpec(columns=['answer'], prediction_type='labels' if classes else 'values'),
            source_hash='synthetic_cost_probe_v1')
        a, b = inner_split(X, y, task, 'random', 42)
        if case == 'opposing_regression':
            y.iloc[b, 0] *= -1
        context = dict(cpu=1, memory_mb=4096, strategy='random', protocol_hash='synthetic_cost_probe_v1',
            split_id='audit', workdir=str(directory), budget_seconds=90, task=task.model_dump(mode='json'))
        model = DirectPipeline(PipelineSpec.model_validate(recipes[0]), task)
        model.fit(X.iloc[a], y.iloc[a], context)
        prediction = np.asarray(model.predict(X.iloc[b], context).values['answer'])
        if classification:
            labels = np.asarray(classes['answer'])[prediction.argmax(axis=1)]
            score = float(f1_score(y.iloc[b].answer, labels, labels=list(range(10)), average='macro', zero_division=0))
        else:
            score = float(np.sqrt(np.mean((y.iloc[b].answer.to_numpy() - prediction) ** 2)))
        replay_scores[case] = score
        for run in observed:
            reported = run['phases'][0]['trial_summary']['best']
            if run['source'] == 'candidate' or case == 'regression':
                assert abs(reported - score) < 1e-10, (case, reported, score)
            else:
                assert abs(reported - score) > 1e-4
        results.append(dict(case=case, frozen_recipes_identical=True, native_fit_sequences_identical=True,
            frozen_inner_replay_primary=score, candidate_trial_score_verified=True,
            baseline_trial_score_matches_frozen_replay=(case == 'regression')))
    return dict(status='verified', completed_pairs=12, fresh_processes=24, sources_unchanged=True,
        replay_checks=results, real_api_calls=0, quality_gain_claimed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--baseline-source', type=Path)
    args = parser.parse_args()
    result = verify(args.directory.resolve(), args.source_root, args.baseline_source)
    (args.directory / 'independent_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
