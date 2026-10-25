"""Offline probes of preprocessing contracts; records defects without changing runtime.

Re-run after a fix: reproduced=False is a valid outcome, not an audit failure.
The synthetic comparison illustrates one failure mode, not an agent benchmark.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(output):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    from threadpoolctl import threadpool_limits
    from automl_agent.contracts import MetricSpec, OutputSpec, PipelineSpec, TaskSpec, Transformation
    from automl_agent.preprocessing import FeatureMatrix, FrameTransform
    from automl_agent.pipelines import build_pipeline
    from automl_agent.util import atomic_json, file_hash

    rng = np.random.default_rng(42)
    n = 600
    codes = rng.integers(0, 4, n)
    X = pd.DataFrame({'code': codes, 'value': rng.normal(size=n)}, index=[f'row:{i}' for i in range(n)])
    y = pd.DataFrame({'y': np.asarray([0., 10., 0., 10.])[codes]}, index=X.index)
    task = TaskSpec(kind='regression', targets=['y'], feature_columns=list(X), categorical_columns=['code'],
        metric=MetricSpec(name='rmse'), output=OutputSpec(columns=['y'], prediction_type='values'), source_hash='synthetic')
    spec = PipelineSpec(model='ridge')
    context = {'task': task.model_dump(mode='json'), 'cpu': 1, 'memory_mb': 1024, 'strategy': 'random', 'protocol_hash': 'offline',
        'split_id': 'probe', 'budget_seconds': 20, 'deadline': 9999999999, 'workdir': str(output.parent)}
    scores, roles = {}, {}
    with threadpool_limits(limits=1):
        for representation in ('numeric_codes', 'string_codes'):
            frame = X.copy()
            if representation == 'string_codes':
                frame['code'] = frame['code'].astype(str)
            pipeline = build_pipeline(spec, task, {})
            pipeline.fit(frame.iloc[:480], y.iloc[:480], context)
            prediction = pipeline.predict(frame.iloc[480:], context)
            scores[representation] = float(mean_squared_error(y.iloc[480:]['y'], prediction.values['y']) ** .5)
            roles[representation] = {'numeric': pipeline.preprocessor.numeric_columns,
                'categorical': pipeline.preprocessor.categorical_columns}
        # The selected feature actually changes with the fit target. Its summary
        # must therefore distinguish target-dependent from target-free learning.
        a = FeatureMatrix(spec.model_copy(update={'transformations': [Transformation(name='select_kbest', parameters={'k': 1})]}), task)
        b = FeatureMatrix(a.spec, task)
        a.fit(X, pd.DataFrame({'y': X['code'] + rng.normal(scale=.01, size=n)}, index=X.index))
        b.fit(X, pd.DataFrame({'y': X['value'] + rng.normal(scale=.01, size=n)}, index=X.index))
        multi = task.model_copy(update={'kind': 'multioutput_regression', 'targets': ['first', 'second'],
            'output': OutputSpec(columns=['first', 'second'], prediction_type='values')})
        multi_y = pd.DataFrame({'first': X['code'] + rng.normal(scale=.01, size=n),
            'second': X['value'] + rng.normal(scale=.01, size=n)}, index=X.index)
        c = FeatureMatrix(a.spec, multi).fit(X, multi_y)
        reversed_task = multi.model_copy(update={'targets': ['second', 'first'],
            'output': OutputSpec(columns=['second', 'first'], prediction_type='values')})
        d = FeatureMatrix(a.spec, reversed_task).fit(X, multi_y[['second', 'first']])
    from tempfile import TemporaryDirectory
    from automl_agent.data import read_csv
    with TemporaryDirectory() as temp:
        csv_path = Path(temp) / 'numeric_codes.csv'
        X.to_csv(csv_path, index=False)
        loaded = read_csv(csv_path)
        csv_category_type = type(loaded['code'].iloc[0]).__name__
    dates = pd.DataFrame({'when': ['2020-01-01', '2021-01-01'], 'when__year': [7., 8.]})
    error, overwritten = None, False
    try:
        converted = FrameTransform([Transformation(name='datetime', parameters={'columns': ['when']})]).fit_transform(dates)
        overwritten = not converted['when__year'].equals(dates['when__year'])
    except ValueError as exc:
        error = str(exc)
    result = {'api_calls': 0, 'runtime_changed': False, 'synthetic_example_not_agent_evaluation': True,
        'source_sha256': file_hash(Path(__file__)),
        'runtime_source_sha256': file_hash(ROOT / 'automl_agent' / 'preprocessing.py'),
        'numeric_categorical_role': {'reproduced': 'code' in roles['numeric_codes']['numeric'],
            'roles': roles, 'synthetic_rmse': scores, 'standard_csv_loader_value_type': csv_category_type,
            'ordinary_csv_path_protected_by_string_loading': csv_category_type == 'str', 'seed': 42, 'fit_rows': 480, 'held_out_rows': 120,
            'target': 'fixed nonmonotonic category effects; no tuning'},
        'datetime_collision': {'reproduced': overwritten, 'input': dates['when__year'].tolist(), 'rejection': error},
        'multioutput_selection': {'reproduced': c.output_features != d.output_features,
            'forward_target_order': multi.targets, 'forward_features': c.output_features,
            'reversed_target_order': reversed_task.targets, 'reversed_features': d.output_features,
            'metric_target_aggregation': multi.metric.aggregation},
        'supervised_summary': {'reproduced': a.output_features != b.output_features
            and a.summary['training_scope'] == b.summary['training_scope'] == 'fit_only_no_target',
            'first_features': a.output_features, 'second_features': b.output_features,
            'reported_scopes': [a.summary['training_scope'], b.summary['training_scope']],
            'interpretation': 'Scope is serialized under imputation; do not infer the entire pipeline is target-free' }}
    atomic_json(output, result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
