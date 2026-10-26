"""Development-only fixed-model feature ablation through the custom interface."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ["raw", "building", "clean_building", "engineered_building"]
BUILDING = ["State_Factor", "building_class", "facility_type", "floor_area",
            "year_built", "energy_star_rating", "ELEVATION"]
CODE = '''import numpy as np
from scipy.special import ndtri
class Features:
    def fit(self, X):
        return self
    def transform(self, X):
        result = X.copy()
        columns = BUILDING_COLUMNS
        if MODE in ['clean_building', 'engineered_building']:
            result['year_built'] = result.year_built.mask(result.year_built <= 0)
        if MODE == 'engineered_building':
            result['ref_log_area'] = np.log1p(result.floor_area.clip(lower=0))
            rating = result.energy_star_rating
            result['ref_rating_missing'] = rating.isna().astype(float)
            result['ref_rating_normal'] = ndtri(((rating + .5) / 101).clip(.005, .995))
            result['ref_state_facility'] = result.State_Factor.astype(str) + '|' + result.facility_type.astype(str)
            result['ref_facility_area'] = result.facility_type.astype(str) + '|' + np.floor(result.ref_log_area).astype('Int64').astype(str)
            columns = columns + ['ref_log_area', 'ref_rating_missing', 'ref_rating_normal', 'ref_state_facility', 'ref_facility_area']
        if MODE != 'raw':
            result = result[columns + ['Year_Factor']]
        return result
def build_transformer(context):
    return Features()
'''


def worker(out, arm):
    import resource
    import cloudpickle
    import numpy as np
    from automl_agent.contracts import PipelineSpec
    from automl_agent.custom import TransformerPipeline, load_module, validate_code
    from automl_agent.evaluation import score_predictions
    from automl_agent.util import atomic_json, file_hash
    from tools.pytorch_agent_study import load_dataset

    started = time.monotonic()
    freeze = json.loads((out / 'freeze.json').read_text())
    for path, expected in freeze['sources'].items():
        assert file_hash(ROOT / path) == expected, path
    directory = out / arm
    directory.mkdir()
    dataset = load_dataset('regression')
    assert dataset.source_hash == freeze['data_hash']
    split_path = ROOT / freeze['split_path']
    assert file_hash(split_path) == freeze['split_hash']
    with np.load(split_path) as split:
        fit, evaluate = split['fit'], split['validation']
    assert not set(fit) & set(evaluate)
    # Do not read targets or predictions on the control rows.
    X = dataset.train.iloc[fit][dataset.task.feature_columns].copy()
    V = dataset.train.iloc[evaluate][dataset.task.feature_columns].copy()
    y = dataset.train.iloc[fit][dataset.task.targets]
    z = dataset.train.iloc[evaluate][dataset.task.targets]
    source = CODE.replace('BUILDING_COLUMNS', repr(BUILDING)).replace('MODE', repr(arm))
    validate_code(source, 'transformer')
    path = directory / 'candidate.py'
    path.write_text(source)
    spec = PipelineSpec(engine='custom', model='lightgbm', supervised_transform=False,
        custom_component='transformer', target_transform='log1p',
        exclude_features=[] if arm == 'raw' else ['Year_Factor'],
        hyperparameters=freeze['hyperparameters'], seed=42)
    context = dict(cpu=1, memory_mb=3072, budget_seconds=90, strategy='temporal',
        protocol_hash=freeze['split_hash'], split_id='validation', workdir=str(directory),
        task=dataset.task.model_dump(mode='json'))
    model = TransformerPipeline(load_module(path).build_transformer(context), spec, dataset.task)
    summary = model.fit(X, y, context)
    prediction = model.predict(V, context)
    score = score_predictions(dataset.task, z, prediction.values, prediction.prediction_type)
    state = cloudpickle.dumps(model)
    restored = cloudpickle.loads(state).predict(V, context)
    target = dataset.task.targets[0]
    np.testing.assert_allclose(prediction.values[target], restored.values[target], rtol=0, atol=0)
    values = np.asarray(prediction.values[target])
    np.testing.assert_allclose(score, np.sqrt(np.mean((z[target].to_numpy() - values) ** 2)), rtol=0, atol=1e-10)
    np.save(directory / 'prediction.npy', values)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    atomic_json(directory / 'result.json', dict(arm=arm, rmse=float(score),
        features=model.pipeline.preprocessor.feature_columns, fit_rows=len(X), evaluation_rows=len(V),
        seconds=time.monotonic()-started, cpu_seconds=usage.ru_utime+usage.ru_stime,
        ru_maxrss_mib=usage.ru_maxrss/(1024**2 if sys.platform == 'darwin' else 1024),
        serialized_bytes=len(state), serialization_prediction_identity=True,
        actual_iterations=model.pipeline.models[target].n_estimators_,
        auxiliary_fits=summary.auxiliary_fits, control_scored=False, api_calls=0))


def run(out):
    import psutil
    from automl_agent.executor import terminate_tree
    from automl_agent.util import atomic_json, file_hash
    from tools.pytorch_agent_study import load_dataset

    out.mkdir(parents=True, exist_ok=False)
    split = 'research/pytorch_agent_20261026/reference_templates/regression_split.npz'
    sources = [Path(__file__), *sorted((ROOT/'automl_agent').glob('*.py'))]
    atomic_json(out/'freeze.json', dict(arms=ARMS, seed=42, cpu=1, memory_mb=3072,
        arm_wall_seconds=90, data_hash=load_dataset('regression').source_hash,
        sources={str(p.relative_to(ROOT)):file_hash(p) for p in sources},
        split_path=split, split_hash=file_hash(ROOT/split),
        hyperparameters=dict(n_estimators=1500, learning_rate=.03, num_leaves=63,
            min_child_samples=20, reg_lambda=1., colsample_bytree=.7, subsample=.8, subsample_freq=1),
        target_transform='log1p', log_inverse='naive', tuning=False,
        purpose='diagnostic feature effect at fixed model; not a reproduction of the selected early-stopped agent',
        control_previously_viewed=True, control_scored=False, api_calls=0,
        validation_status='previously viewed development split; no independent confirmation'))
    rows=[]
    for arm in ARMS:
        env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MPLCONFIGDIR=str(out/'mpl'))
        begin=time.monotonic(); peak=0
        with (out/(arm+'.log')).open('w') as log:
            process=subprocess.Popen([sys.executable,__file__,str(out),'--arm',arm],env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    try:
                        parent=psutil.Process(process.pid)
                        peak=max(peak,sum(p.memory_info().rss for p in [parent,*parent.children(recursive=True)] if p.is_running()))
                    except psutil.NoSuchProcess:
                        pass
                    if time.monotonic()-begin>90 or peak>3072*1024**2:
                        terminate_tree(process.pid)
                    time.sleep(.1)
                process.wait()
            finally:
                if process.poll() is None:
                    try:
                        terminate_tree(process.pid)
                    finally:
                        if process.poll() is None:
                            process.kill()
                        process.wait()
        row=dict(arm=arm,returncode=process.returncode,cold_wall_seconds=time.monotonic()-begin,
            sampled_tree_peak_mib=peak/1024**2)
        if process.returncode==0:
            row.update(json.loads((out/arm/'result.json').read_text()))
        rows.append(row);atomic_json(out/'results.json',dict(records=rows,api_calls=0,control_scored=False))
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    parser.add_argument('--arm',choices=ARMS);args=parser.parse_args()
    output=args.output.resolve()
    worker(output,args.arm) if args.arm else run(output)
