"""Describe reference errors without fitting corrections to audit labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from tools.solve_participant_examples import ROOT, write


def regression_segments(frame, labels, prediction, tail_cutoff):
    """Overlapping descriptive segments; cutoff learned on earlier rows only."""
    error = np.asarray(prediction) - np.asarray(labels)
    squared = error ** 2
    masks = {'all': np.ones(len(frame), dtype=bool),
        'rating_missing': frame.energy_star_rating.isna().to_numpy(),
        'rating_present': frame.energy_star_rating.notna().to_numpy(),
        'target_above_earlier_q95': np.asarray(labels) > tail_cutoff,
        'target_at_most_earlier_q95': np.asarray(labels) <= tail_cutoff}
    masks.update({f'building_class={value}': frame.building_class.eq(value).to_numpy()
                  for value in sorted(frame.building_class.dropna().unique())})
    masks.update({f'facility_type={value}': frame.facility_type.eq(value).to_numpy()
                  for value in sorted(frame.facility_type.dropna().unique())})
    rows = []
    for name, mask in masks.items():
        if not mask.any():
            continue
        rows.append(dict(segment=name, count=int(mask.sum()),
            rmse=float(np.sqrt(squared[mask].mean())), mae=float(np.abs(error[mask]).mean()),
            bias=float(error[mask].mean()), squared_error_share=float(squared[mask].sum() / squared.sum())))
    return rows


def report(regression, output):
    regression = Path(regression)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(ROOT / 'data_for_participants/Регрессия/train.csv')
    split = np.load(regression / 'splits.npz')
    segments = []
    for period in [3, 4]:
        rows = split[f'eval{period}']
        p = sum(np.load(regression / f'{m}_fixed_existing_period{period}/prediction.npy') for m in ['lgb', 'cb']) / 2
        cutoff = float(frame.site_eui.iloc[split[f'full{period}']].quantile(.95))
        for item in regression_segments(frame.iloc[rows], frame.site_eui.iloc[rows], p, cutoff):
            segments.append(dict(period=period, earlier_target_q95=cutoff, **item))
    pd.DataFrame(segments).to_csv(out / 'regression_segments.csv', index=False)
    source = ROOT / 'outputs/further_reference_20261025/classification'
    frame = pd.read_csv(ROOT / 'data_for_participants/Многоклассовая классификация/train.csv')
    split = np.load(source / 'splits.npz')
    manifest = json.loads((source / 'inference.json').read_text())
    classes = manifest['classes']; normal = classes.index('Normal')
    rows = split['control']; labels = pd.Categorical(frame.attack_cat.iloc[rows], categories=classes).codes
    p = np.load(source / 'audit_prediction.npy')
    confusion = np.bincount(labels * len(classes) + p, minlength=len(classes) ** 2).reshape(len(classes), len(classes))
    pd.DataFrame(confusion, index=classes, columns=classes).to_csv(out / 'classification_confusion.csv')
    wrong = labels != p
    gate = (labels == normal) != (p == normal)
    pairs = sorted([(int(confusion[i, j]), classes[i], classes[j]) for i in range(len(classes))
                    for j in range(len(classes)) if i != j], reverse=True)
    write(out / 'classification_errors.json', {
        'scope': 'previously reused group audit; descriptive only; no correction fitted',
        'rows': len(rows), 'errors': int(wrong.sum()), 'normal_attack_errors': int(gate.sum()),
        'among_attack_type_errors': int((wrong & ~gate).sum()),
        'largest_confusions': [dict(count=n, actual=a, predicted=b) for n, a, b in pairs[:15]]})
    write(out / 'protocol.json', {'regression_scope': 'forward development years 3 and 4',
        'tail_threshold_source': '95th percentile of earlier training labels only',
        'segments_overlap': True, 'audit_labels_not_used_for_corrections': True,
        'small_segment_scores_are_unstable': True, 'api_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regression', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.regression, args.output)
