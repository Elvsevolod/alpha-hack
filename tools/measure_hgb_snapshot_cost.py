"""Offline synthetic resource probe; not a model-quality or contest benchmark.

Every select/refit phase starts a fresh interpreter, with one thread. Baseline
must be an immutable source export. Reports contain no rows or trained weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

CASES = {
    'regression': {'rows': 50000, 'features': 32, 'kind': 'regression'},
    'opposing_regression': {'rows': 50000, 'features': 32, 'kind': 'regression'},
    'ten_classes': {'rows': 20000, 'features': 24, 'kind': 'classification'},
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sources(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / 'automl_agent').glob('*.py'))}


def child(args):
    sys.path.insert(0, str(args.source_root.resolve()))
    import cloudpickle
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from automl_agent.contracts import MetricSpec, OutputSpec, PipelineSpec, TaskSpec
    from automl_agent.pipelines import build_pipeline
    from automl_agent.validation import inner_split

    cfg = CASES[args.case]
    n, width = cfg['rows'], cfg['features']
    rng = np.random.default_rng(2901)
    raw = rng.normal(size=(n + n // 4 + 1000, width))
    if cfg['kind'] == 'classification':
        targets = np.argmax(raw[:, :10] + rng.normal(scale=.6, size=(len(raw), 10)), axis=1)
        classes = {'answer': list(reversed(range(10)))}
        metric = MetricSpec(name='f1', average='macro')
    else:
        targets = 2 * raw[:, 0] + np.sin(raw[:, 1]) + raw[:, 2] * raw[:, 3] + rng.normal(scale=.5, size=len(raw))
        classes, metric = {}, MetricSpec(name='rmse')
    raw[rng.random(raw.shape) < .02] = np.nan
    frame = pd.DataFrame(raw, columns=[f'x{i}' for i in range(width)], index=[f'r{i}' for i in range(len(raw))])
    actual = pd.DataFrame({'answer': targets}, index=frame.index)
    task = TaskSpec(kind=cfg['kind'], targets=['answer'], feature_columns=list(frame), classes=classes,
        metric=metric, output=OutputSpec(columns=['answer'], prediction_type='labels' if classes else 'values'),
        source_hash='synthetic_cost_probe_v1')
    if args.case == 'opposing_regression':
        _, b = inner_split(frame.iloc[:n], actual.iloc[:n], task, 'random', 42)
        actual.iloc[b, 0] *= -1
    context = dict(cpu=1, memory_mb=4096, strategy='random', protocol_hash='synthetic_cost_probe_v1',
        split_id='probe', workdir=str(args.directory), budget_seconds=90, task=task.model_dump(mode='json'))
    counts = []
    for cls in (HistGradientBoostingClassifier, HistGradientBoostingRegressor):
        original = cls.fit
        def fit(model, X, y, _original=original, **kwargs):
            started = time.perf_counter()
            result = _original(model, X, y, **kwargs)
            counts.append(dict(rows=len(X), max_iter=model.max_iter, n_iter=model.n_iter_,
                seconds=time.perf_counter() - started))
            return result
        cls.fit = fit
    if args.phase == 'select':
        spec = PipelineSpec(engine='optuna', model='hist_gradient_boosting', n_trials=1,
            hyperparameters={'max_iter':80, 'min_samples_leaf':20}, early_stopping_rounds=20,
            evaluation_interval=10, search_space={'learning_rate':{'type':'categorical', 'choices':[.1]}})
        rows = n
    else:
        spec = PipelineSpec.model_validate_json(args.frozen_path.read_text())
        rows = n + n // 4
        context['execution_mode'] = 'refit'
    model = build_pipeline(spec, task, context)
    started = time.perf_counter()
    summary = model.fit(frame.iloc[:rows], actual.iloc[:rows], context)
    fit_seconds = time.perf_counter() - started
    frozen = model.frozen_spec()
    save(args.directory / 'frozen.json', frozen.model_dump(mode='json'))
    heldout = frame.iloc[-1000:]
    started = time.perf_counter()
    expected = model.predict(heldout, context).values['answer']
    prediction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    path = args.directory / 'model.pkl'
    with path.open('wb') as stream:
        cloudpickle.dump(model, stream)
    del model
    with path.open('rb') as stream:
        restored = cloudpickle.load(stream)
    np.testing.assert_allclose(restored.predict(heldout, context).values['answer'], expected, atol=1e-12)
    save_load_verify_seconds = time.perf_counter() - started
    native = restored.pipeline.models['answer'] if hasattr(restored, 'pipeline') else restored.models['answer']
    assert native.n_iter_ == frozen.iteration_schedule['answer']
    usage = resource.getrusage(resource.RUSAGE_SELF)
    save(args.directory / 'result.json', dict(status='completed', phase=args.phase, case=args.case, rows=rows,
        source_root=str(args.source_root), sklearn=sklearn.__version__, python=platform.python_version(),
        fit_seconds=fit_seconds, prediction_seconds=prediction_seconds,
        save_load_verify_seconds=save_load_verify_seconds, model_bytes=path.stat().st_size,
        cpu_seconds=usage.ru_utime + usage.ru_stime,
        peak_rss_mib=usage.ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024),
        frozen_iterations=frozen.iteration_schedule, native_iterations=native.n_iter_,
        native_fit_calls=counts, trial_summary=summary.inner_scores if args.phase == 'select' else None,
        pickle_prediction_identity_verified=True))


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    roots = dict(baseline=args.baseline.resolve(), candidate=args.candidate.resolve())
    hashes = {name:sources(root) for name,root in roots.items()}
    freeze = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=CASES,
        source_sha256=hashes, order=['baseline', 'candidate', 'candidate', 'baseline'], seed=2901,
        repetitions_are_timing_repeats_not_independent_quality_evidence=True,
        phase_budget_seconds=90, external_phase_timeout_seconds=120, cpu_threads=1,
        quality_claimed=False, real_api_calls=0, platform=platform.platform())
    save(output / 'freeze.json', freeze)
    environment = {k:v for k,v in os.environ.items() if not k.startswith(('OPENAI_', 'AGENT_', 'PYTHONPATH'))}
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    runs = []
    try:
        for case in CASES:
            for repeat, name in enumerate(freeze['order']):
                result = dict(case=case, order=repeat, source=name, phases=[])
                for phase in ('select', 'refit'):
                    directory = output / f'{case}_{repeat}_{name}' / phase
                    directory.mkdir(parents=True)
                    command = [sys.executable, '-I', str(Path(__file__).resolve()), '--child',
                        '--source-root', str(roots[name]), '--phase', phase, '--case', case,
                        '--directory', str(directory)]
                    if phase == 'refit':
                        command += ['--frozen-path', str(directory.parent / 'select' / 'frozen.json')]
                    started = time.perf_counter()
                    with (directory / 'process.log').open('wb') as log:
                        subprocess.run(command, cwd=directory, env=environment, check=True, timeout=120,
                            stdout=log, stderr=subprocess.STDOUT)
                    report = json.loads((directory / 'result.json').read_text())
                    report['full_process_wall_seconds'] = time.perf_counter() - started
                    result['phases'].append(report)
                result.update(total_wall_seconds=sum(p['full_process_wall_seconds'] for p in result['phases']),
                    total_cpu_seconds=sum(p['cpu_seconds'] for p in result['phases']),
                    peak_rss_mib=max(p['peak_rss_mib'] for p in result['phases']))
                runs.append(result)
                save(output / 'runs.json', runs)
                print(case, repeat, name, round(result['total_wall_seconds'], 3), 's',
                    round(result['peak_rss_mib'], 1), 'MiB', flush=True)
    finally:
        unchanged = hashes == {name:sources(root) for name,root in roots.items()}
        save(output / 'state.json', dict(completed_pairs=len(runs), expected_pairs=len(CASES)*4,
            source_unchanged=unchanged, script_unchanged=freeze['script_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            real_api_calls=0))
    assert unchanged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    for name in ('source-root', 'directory', 'frozen-path', 'output', 'baseline', 'candidate'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--phase', choices=['select', 'refit'])
    parser.add_argument('--case', choices=list(CASES))
    args = parser.parse_args()
    child(args) if args.child else run(args)
