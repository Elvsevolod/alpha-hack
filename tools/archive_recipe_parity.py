"""Offline same-recipe comparison of immutable extracted submission runtimes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ['v101', 'v116', 'v120']
RECIPES = {
    'regression_log_fixed': dict(model='lightgbm', representation='native', target_transform='log1p',
        hyperparameters=dict(n_estimators=1500, learning_rate=.03, num_leaves=63,
            min_child_samples=20, reg_lambda=1., colsample_bytree=.7, subsample=.8, subsample_freq=1)),
    'regression_stopped': dict(model='lightgbm', representation='native', early_stopping_rounds=100,
        hyperparameters=dict(n_estimators=1500, learning_rate=.03, num_leaves=31)),
    'classification_offsets': dict(model='lightgbm', representation='native', prediction_policy='class_offsets',
        postprocess_fit_policy='pooled_oof', hyperparameters=dict(n_estimators=300, learning_rate=.05,
            num_leaves=31, class_weight='balanced')),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def worker(out, version, recipe):
    started = time.monotonic()
    freeze = json.loads((out / 'freeze.json').read_text())
    runtime = Path(freeze['runtimes'][version]['path'])
    for rel, expected in freeze['runtimes'][version]['sources'].items():
        assert sha(runtime / rel) == expected, rel
    sys.path.insert(0, str(runtime))
    import cloudpickle
    import numpy as np
    import pandas as pd
    from automl_agent.contracts import TaskSpec, PipelineSpec
    from automl_agent.pipelines import build_pipeline
    from automl_agent.evaluation import score_predictions
    assert Path(sys.modules['automl_agent.pipelines'].__file__).resolve().is_relative_to(runtime)
    kind = 'regression' if recipe.startswith('regression') else 'classification'
    source = freeze['tasks'][kind]
    for rel, expected in source['hashes'].items():
        assert sha(ROOT / rel) == expected, rel
    task = TaskSpec.model_validate(json.loads((ROOT / source['task']).read_text()))
    plan = json.loads((ROOT / source['plan']).read_text())
    assert not set(plan['fit']) & set(plan['validation'])
    directory = out / f'{version}_{recipe}'
    directory.mkdir()
    # Only permitted fit/development rows are retained; control is never scored.
    columns = list(dict.fromkeys(task.feature_columns + task.targets))
    data = pd.read_csv(ROOT / source['train'], usecols=columns)
    fit, valid = data.iloc[plan['fit']].copy(), data.iloc[plan['validation']].copy()
    del data
    fit.index = ['train:' + str(i) for i in plan['fit']]
    valid.index = ['train:' + str(i) for i in plan['validation']]
    spec = PipelineSpec(**freeze['recipes'][recipe], seed=42)
    context = dict(cpu=1, memory_mb=3072, budget_seconds=170, strategy=source['strategy'],
        protocol_hash=source['hashes'][source['plan']], split_id='validation',
        workdir=str(directory), task=task.model_dump(mode='json'), fit_row_keys=list(fit.index))
    model = build_pipeline(spec, task, context)
    summary = model.fit(fit[task.feature_columns], fit[task.targets], context)
    bundle = model.predict(valid[task.feature_columns], context)
    score = score_predictions(task, valid[task.targets], bundle.values, bundle.prediction_type)
    restored = cloudpickle.loads(cloudpickle.dumps(model)).predict(valid[task.feature_columns], context)
    target = task.targets[0]
    values = np.asarray(bundle.values[target])
    np.testing.assert_array_equal(values, np.asarray(restored.values[target]))
    np.save(directory / 'values.npy', values)
    probabilities = bundle.raw_probabilities.get(target)
    if probabilities is not None:
        np.testing.assert_array_equal(probabilities, restored.raw_probabilities[target])
        np.save(directory / 'probabilities.npy', np.asarray(probabilities))
    models = getattr(model, 'models', {})
    counts = {k: getattr(m, 'n_estimators_', None) for k, m in models.items()}
    result = dict(version=version, recipe=recipe, score=score, prediction_type=bundle.prediction_type,
        fit_rows=len(fit), validation_rows=len(valid), target=target, iterations=counts,
        fit_summary=summary.model_dump(mode='json'), classes=bundle.classes,
        serialized_prediction_identity=True, seconds=time.monotonic() - started,
        loaded_pipeline_path=str(Path(sys.modules['automl_agent.pipelines'].__file__)))
    save(directory / 'result.json', result)


def run(out):
    import psutil
    import numpy as np
    out.mkdir(exist_ok=False, parents=True)
    runtimes = {}
    for version in VERSIONS:
        runtime = ROOT / 'outputs/official_regression_audit_20261026' / version / 'runtime'
        runtimes[version] = dict(path=str(runtime), sources={str(p.relative_to(runtime)): sha(p)
            for p in sorted(runtime.rglob('*.py'))})
    tasks = {}
    for kind, base in [('regression', 'agent_autonomy_regression_expanded_20261026'),
                       ('classification', 'agent_autonomy_budget_20261026')]:
        taskroot = Path('outputs') / base / 'task'
        work = taskroot / '.agent_work/bounded_autonomy'
        tasks[kind] = dict(train=str(taskroot / 'train.csv'), task=str(work / 'task_spec.json'),
            plan=str(work / 'validation_plan.json'), strategy='temporal' if kind == 'regression' else 'exact_group')
        tasks[kind]['hashes'] = {tasks[kind][key]: sha(ROOT / tasks[kind][key]) for key in ['train', 'task', 'plan']}
    save(out / 'freeze.json', dict(runtimes=runtimes, tasks=tasks, recipes=RECIPES,
        driver_sha256=sha(__file__), cpu=1, memory_mb=3072, wall_limit_per_worker=180,
        api_calls=0, control_scored=False, validation_previously_viewed=True,
        purpose='numerical parity, not autonomous choice or independent transfer'))
    records = []
    env = {k: v for k, v in os.environ.items() if not any(x in k.upper() for x in ['KEY', 'TOKEN', 'SECRET', 'OPENAI'])}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MPLCONFIGDIR=str(out / 'mpl'))
    for recipe in RECIPES:
        for version in VERSIONS:
            begin = time.monotonic(); peak = 0; stop = None
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            with (out / f'{version}_{recipe}.log').open('w') as log:
                process = subprocess.Popen([sys.executable, __file__, str(out), '--version', version,
                    '--recipe', recipe], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while process.poll() is None:
                        try:
                            parent = psutil.Process(process.pid)
                            peak = max(peak, sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)] if p.is_running()))
                        except psutil.NoSuchProcess:
                            pass
                        if time.monotonic() - begin > 180 or peak > 3072 * 1024**2:
                            stop = 'wall_or_memory_limit'
                            os.killpg(process.pid, 15)
                            try: process.wait(timeout=2)
                            except subprocess.TimeoutExpired: os.killpg(process.pid, 9)
                        time.sleep(.1)
                    process.wait()
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, 9); process.wait()
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            row = dict(version=version, recipe=recipe, returncode=process.returncode, stop=stop,
                cold_wall_seconds=time.monotonic()-begin, peak_process_tree_rss_mib=peak/1024**2,
                child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
            result = out / f'{version}_{recipe}' / 'result.json'
            if result.exists():
                d = json.loads(result.read_text())
                row.update(score=d['score'], iterations=d['iterations'], serialized_prediction_identity=d['serialized_prediction_identity'])
            records.append(row)
            save(out / 'results.json', dict(records=records, api_calls=0, control_scored=False))
            print(json.dumps(row), flush=True)
    parity = []
    for recipe in RECIPES:
        base = out / f'v101_{recipe}'
        for version in VERSIONS[1:]:
            other = out / f'{version}_{recipe}'
            item = dict(recipe=recipe, comparison=f'v101 vs {version}')
            for name in ['values', 'probabilities']:
                if (base / f'{name}.npy').exists() and (other / f'{name}.npy').exists():
                    a, b = np.load(base / f'{name}.npy'), np.load(other / f'{name}.npy')
                    item[name + '_exactly_equal'] = bool(np.array_equal(a, b))
                    if np.issubdtype(a.dtype, np.number):
                        item[name + '_max_abs_delta'] = float(np.max(np.abs(a-b)))
            parity.append(item)
    save(out / 'parity.json', parity)
    print(json.dumps(parity), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('output', type=Path)
    parser.add_argument('--version', choices=VERSIONS); parser.add_argument('--recipe', choices=list(RECIPES))
    args = parser.parse_args()
    if args.version:
        worker(args.output.resolve(), args.version, args.recipe)
    else:
        run(args.output.resolve())
