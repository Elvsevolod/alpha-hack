"""Read-only development audit of inputs and saved reference predictions.

No fitting, downloads, external labels, API calls or changes to working CSVs.
These controls were previously inspected; this is descriptive reanalysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from tools.literature_reference_study import probabilities
from tools.reference_features import macro_f1
from tools.solve_participant_examples import ROOT, predict_saved, sha, write


def exact_group_facts(train, test, target):
    columns = [c for c in test if c != 'id']
    combined = pd.concat([train[columns], test[columns]], ignore_index=True)
    # Exact grouping, including missing values; not approximate similarity or
    # an assumption that equal feature vectors identify a network session.
    groups = combined.groupby(columns, dropna=False, sort=False, observed=True).ngroup().to_numpy()
    fit, query = groups[:len(train)], groups[len(train):]
    table = pd.crosstab(fit, train[target])
    conflicting = (table > 0).sum(axis=1) > 1
    return dict(rows=len(train), distinct_feature_groups=len(table),
        duplicate_rows_beyond_first=len(train)-len(table),
        conflicting_groups=int(conflicting.sum()),
        rows_in_conflicting_groups=int(table.sum(axis=1)[conflicting].sum()),
        feature_only_deterministic_training_accuracy_upper_bound=float(table.max(axis=1).sum()/len(train)),
        upper_bound_scope='empirical train labels only; not a bound on external-test performance or Macro F1',
        test_rows=len(test), test_rows_with_train_feature_key=int(np.isin(query, np.unique(fit)).sum()),
        exact_feature_equality_is_not_session_identity=True)


def aggregate_rmse(scores, counts):
    scores, counts = np.asarray(scores, dtype=float), np.asarray(counts, dtype=float)
    if len(scores) != len(counts) or not len(scores) or not np.isfinite(scores).all() or not np.isfinite(counts).all() or (counts <= 0).any() or (scores < 0).any():
        raise ValueError('Need positive validation counts for all folds')
    return dict(equal_period_mean_rmse=float(scores.mean()),
        equal_period_root_mean_mse=float(np.sqrt(np.mean(scores**2))),
        pooled_row_rmse=float(np.sqrt(np.average(scores**2, weights=counts))))


def frozen_stage_audit(run, frame):
    """Same inspected control and frozen offsets, different saved fit stages."""
    split = np.load(run/'splits.npz')
    protocol = json.loads((run/'protocol.json').read_text())
    directory = ROOT/'data_for_participants/Многоклассовая классификация'
    if any(sha(directory/name) != expected for name, expected in protocol['source_hashes'].items()):
        raise ValueError('Historical fitted models refer to different task inputs')
    selection = json.loads((run/'selection.json').read_text())
    manifest = json.loads((run/'inference.json').read_text())
    classes = manifest['classes']
    y = pd.Categorical(frame.attack_cat, categories=classes).codes
    rows = split['control']; query = frame.iloc[rows]
    offsets = np.mean(selection['learned_offsets'], axis=0) if selection['offsets'] else np.zeros(len(classes))
    counts = {s: np.bincount(y[split[key]], minlength=len(classes))
              for s, key in [('inner_fit', 'fit0'), ('development', 'development')]}
    predictions, model_sources = {}, []
    # A reused CV model can live in another study; resolve only recorded reuse
    # links when its model bundle wasn't copied into the current run.
    reused = json.loads((run/'reused_predictions.json').read_text()) if (run/'reused_predictions.json').exists() else {}
    for name in selection['names']:
        cv = run/f'{name}_cv0/model.joblib'
        if not cv.exists():
            source = reused.get(name)
            if isinstance(source, dict):
                source = source.get('source')
            if source:
                cv = Path(source)/f'{name}_cv0/model.joblib'
            else:
                # Current compact study explicitly recorded the baseline reuse
                # in its completed CV entries, whose configurations match.
                source = ROOT/'outputs/reduction_classification_20261025/classification'
                old = json.loads((source/'completed.json').read_text())
                now = json.loads((run/'completed.json').read_text())
                if now[name] != old[name]:
                    raise ValueError('Cannot resolve a matching historical CV model')
                previous = np.load(source/'splits.npz')
                for key in ['fit0', 'eval0', 'control']:
                    np.testing.assert_array_equal(split[key], previous[key])
                cv = source/f'{name}_cv0/model.joblib'
        predictions.setdefault('inner_fit', {})[name] = predict_saved(cv, query)
        fitted = run/f'{name}_audit/model.joblib'
        predictions.setdefault('development', {})[name] = predict_saved(fitted, query)
        model_sources.extend(dict(name=name, stage=stage, path=str(path.relative_to(ROOT)), sha256=sha(path))
            for stage,path in [('inner_fit',cv),('development',fitted)])
    cells = []
    for stage, values in predictions.items():
        for prior, frequency in counts.items():
            p = probabilities(values, selection['names'], selection['weights'], frequency, selection['tau'])
            labels = (np.log(np.maximum(p, 1e-15))+offsets).argmax(axis=1)
            cells.append(dict(model_training_stage=stage, prior_count_stage=prior,
                macro_f1=macro_f1(y[rows], labels, len(classes)),
                predicted_counts=np.bincount(labels, minlength=len(classes)).tolist()))
    return dict(selection=selection, classes=classes, cells=cells,
        data_hashes_match_fitted_study=True,
        model_sources=model_sources,
        fit_rows=len(split['fit0']), development_rows=len(split['development']), control_rows=len(rows),
        cv_fit_rows_identical=np.array_equal(split['fit0'], split['fit1']),
        cv_evaluation_rows_identical=np.array_equal(split['eval0'], split['eval1']),
        interpretation='descriptive counterfactual on reused control; no new fit or policy selection; training rows, learned vocabularies and fit-derived weights may change together')


def audit(output):
    out=Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    directory=ROOT/'data_for_participants'
    ctrain=pd.read_csv(directory/'Многоклассовая классификация/train.csv')
    ctest=pd.read_csv(directory/'Многоклассовая классификация/test.csv')
    rtrain=pd.read_csv(directory/'Регрессия/train.csv')
    rtest=pd.read_csv(directory/'Регрессия/test.csv')
    facts={'classification': exact_group_facts(ctrain, ctest, 'attack_cat'), 'regression_periods': []}
    for period, values in rtrain.groupby('Year_Factor', sort=True):
        facts['regression_periods'].append(dict(period=int(period), rows=len(values),
            missing_rating=float(values.energy_star_rating.isna().mean()),
            target_mean=float(values.site_eui.mean()), target_q95=float(values.site_eui.quantile(.95))))
    facts['regression_test']={'rows':len(rtest), 'periods':sorted(rtest.Year_Factor.unique().tolist()),
        'missing_rating':float(rtest.energy_star_rating.isna().mean()), 'target_accessed':False}
    cases={
        'incumbent':ROOT/'outputs/further_reference_20261025/classification',
        'compact_catboost':ROOT/'outputs/reduction_compact_20261025/classification'}
    for name, run in cases.items():
        facts[name]=frozen_stage_audit(run, ctrain)
    path=ROOT/'outputs/rating_factorization_20261025/regression'
    split=np.load(path/'splits.npz')
    recipes=json.loads((path/'recipes.json').read_text())
    counts=[len(split[f'eval{i}']) for i in range(2)]
    facts['rmse_aggregation']=[dict(names=r['names'],weights=r['weights'],
        **aggregate_rmse(r['fold_scores'],counts)) for r in recipes]
    write(out/'audit.json', facts)
    write(out/'protocol.json', {'api_calls':0,'fits':0,'all_control_labels_previously_used':True,
        'scope':'read-only development research; no external-test labels; no model promotion',
        'source_sha256':sha(Path(__file__)),
        'data_sha256':{str(p.relative_to(ROOT)):sha(p) for p in directory.glob('*/*.csv')
                       if p.name in {'train.csv','test.csv'}}})
    print(json.dumps({k:v for k,v in facts.items() if k in ['classification','regression_periods','regression_test']},ensure_ascii=False,indent=2))
    for name in cases:
        print(name, facts[name]['cells'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output)
