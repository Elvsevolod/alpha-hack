"""Optional target-free category context, selected by a research hypothesis.

This transformer does not infer task semantics or select categories on its own.
Statistics and category support are learned only from its fit population.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class GroupContext:
    def __init__(self, numeric, categorical, min_support=20, max_groups=2000):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.min_support = min_support
        self.max_groups = max_groups

    @staticmethod
    def keys(series):
        return ('V:' + series.astype(str)).where(series.notna(), 'M:')

    def fit(self, frame):
        if not len(frame) or self.min_support < 1 or self.max_groups < 1:
            raise ValueError('Nonempty fit and positive support bounds required')
        raw = frame[self.numeric].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        self.global_median = raw.median().fillna(0)
        self.global_scale = (raw.quantile(.75) - raw.quantile(.25)).fillna(0).clip(lower=1e-6)
        self.tables = {}
        self.fit_rows = len(frame)
        for column in self.categorical:
            key = self.keys(frame[column])
            counts = key.value_counts()
            permitted = counts[counts >= self.min_support].head(self.max_groups).index
            medians = raw.groupby(key, sort=False).median().reindex(permitted)
            self.tables[column] = (counts.reindex(permitted), medians)
        return self

    def transform(self, frame):
        result = frame.copy()
        for group, (counts, medians) in self.tables.items():
            key = self.keys(frame[group])
            result[f'ctx_frequency__{group}'] = key.map(counts).fillna(0) / self.fit_rows
            for column in self.numeric:
                value = pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
                center = key.map(medians[column]).fillna(self.global_median[column])
                # Both absolute-scale deviation and relative deviation are
                # bounded engineering coordinates, not asserted causal effects.
                result[f'ctx_center__{group}__{column}'] = ((value-center)/self.global_scale[column]).clip(-20, 20)
                result[f'ctx_relative__{group}__{column}'] = ((value-center)/(center.abs()+1)).clip(-20, 20)
        return result
