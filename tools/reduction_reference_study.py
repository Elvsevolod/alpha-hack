"""Frozen new-family comparisons: OvA, native CatBoost and pattern specialists."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.further_reference_study import configurations
from tools.literature_reference_study import run_case, write_report


def candidates(classification):
    baseline = configurations(classification)
    if classification:
        # Take the hierarchy by identity rather than relying on list order.
        anchor = dict(next(c for c in baseline if c['name'] == 'hierarchy_balanced'),
            name='current_hierarchy', fixed_iterations=True,
            alternative_component_iterations={'gate': 757, 'attack': 304})
        params = dict(n_estimators=900, num_leaves=31, learning_rate=.04, reg_lambda=5,
            objective='multiclassova')
        ova = dict(name='ova', spec=dict(model='lightgbm', hyperparameters=params, seed=42), feature_plan='network')
        weighted = dict(ova, name='ova_weighted', class_weight_exponent=.5)
        cb = dict(name='native_catboost', spec=dict(model='catboost', seed=42,
            hyperparameters=dict(iterations=900, depth=6, learning_rate=.06,
                l2_leaf_reg=5, auto_class_weights='SqrtBalanced')), feature_plan='network')
        return [baseline[0], anchor, ova, weighted, cb]
    lgb, cb = baseline[:2]
    return [lgb, cb, dict(lgb, name='pattern_lgb', alternative='pattern'),
            dict(lgb, name='interaction_lgb', feature_plan='energy_interactions'),
            dict(cb, name='interaction_cb', feature_plan='energy_interactions')]


def study(output, kind, seconds, compact_native=False, reuse_root=None):
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    classification = kind == 'classification'
    configs = candidates(classification)
    additional = None
    if compact_native:
        if not classification:
            raise ValueError('Compact native series is a classification experiment')
        cb = dict(configs[-1], name='compact_native_catboost')
        cb['spec'] = dict(cb['spec'], hyperparameters=dict(cb['spec']['hyperparameters'],
            iterations=250, depth=5, learning_rate=.1))
        configs = configs[:4] + [cb]
        additional = [dict(names=['current_weighted', 'current_hierarchy', name],
            weights=[(1-weight)/2, (1-weight)/2, weight], tau=.5, offsets=True)
            for name in ['ova', 'ova_weighted'] for weight in [.1, .25, .5]]
    pair = ['current_weighted', 'current_hierarchy'] if classification else ['current_lgb', 'current_cb']
    run_case(root, kind, time.monotonic() + seconds, config_override=configs, reuse_root=reuse_root,
        baseline_override=.6565257846316546 if classification else 39.03614722027966,
        reference_pair=pair, additional_recipes=additional)
    write_report(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['classification', 'regression'], required=True)
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--compact-native', action='store_true')
    parser.add_argument('--reuse-root', type=Path)
    args = parser.parse_args()
    study(args.output, args.kind, args.seconds, args.compact_native, args.reuse_root)
