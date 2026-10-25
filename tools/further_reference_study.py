"""Further source-informed CPU alternatives; development-only, no network."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.literature_reference_study import run_case, write_report
from tools.solve_participant_examples import ROOT, write


def configurations(classification):
    def item(name, model, params, **options):
        return dict(name=name, spec=dict(model=model, hyperparameters=params, seed=42), **options)
    if classification:
        xgb = dict(n_estimators=589, max_depth=5, learning_rate=.06, reg_lambda=5,
                   min_child_weight=2, subsample=.85, colsample_bytree=.85)
        hierarchy = dict(n_estimators=900, num_leaves=31, learning_rate=.05, reg_lambda=5)
        return [
            item('current_weighted', 'xgboost', xgb, class_weight_exponent=.5, feature_plan='network', fixed_iterations=True),
            item('current_plain', 'xgboost', dict(xgb, n_estimators=558), feature_plan='network', fixed_iterations=True),
            item('hierarchy', 'lightgbm', hierarchy, alternative='hierarchical', feature_plan='network'),
            item('hierarchy_balanced', 'lightgbm', dict(hierarchy, class_weight='balanced'), alternative='hierarchical', feature_plan='network'),
        ]
    lgb = dict(n_estimators=2000, num_leaves=31, learning_rate=.03, reg_lambda=10,
               min_child_samples=50, colsample_bytree=.9, subsample=.8, subsample_freq=1)
    return [
        item('current_lgb', 'lightgbm', dict(lgb, n_estimators=1035), building_only=True, feature_plan='energy_building', fixed_iterations=True),
        item('current_cb', 'catboost', dict(iterations=979, depth=8, learning_rate=.04, l2_leaf_reg=8), building_only=True, feature_plan='energy_building', fixed_iterations=True),
        *[item(name, 'lightgbm', dict(lgb, objective=objective), building_only=True,
               feature_plan='energy_building', regression_eval_metric='rmse')
          for name, objective in [('poisson', 'poisson'), ('tweedie15', 'tweedie'), ('gamma', 'gamma')]],
        item('tabm_mini', 'mlp', dict(max_iter=60), alternative='tabm_mini', building_only=True, feature_plan='energy_building'),
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=2700)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.seconds
    previous = json.loads((ROOT / 'docs/reports/examples/2026-10-25-literature-evidence.json').read_text())
    baselines = {'regression': previous['studies']['initial_source_study']['regression']['summary']['audit_score'],
                 'classification': previous['studies']['classification_factorial_followup']['classification']['summary']['audit_score']}
    write(out / 'series.json', {'baseline_commit': 'd9c947c', 'baseline_audit_scores': baselines,
        'api_calls': 0, 'device': 'cpu', 'development_only': True,
        'baseline_predictions_preserved': True, 'test_labels_unavailable': True})
    for kind in ['regression', 'classification']:
        pair = ['current_weighted', 'current_plain'] if kind == 'classification' else ['current_lgb', 'current_cb']
        run_case(out, kind, deadline, config_override=configurations(kind == 'classification'),
                 baseline_override=baselines[kind], reference_pair=pair)
        write_report(out)
