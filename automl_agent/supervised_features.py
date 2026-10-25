"""Declared entity history and externally cross-fitted target encodings."""
import numpy as np
import pandas as pd
from .validation import ordered_periods


class SupervisedFeatures:
    def __init__(self, spec, task, strategy):
        self.spec, self.task, self.strategy = spec, task, strategy
        self.encodings, self.histories, self.report = [], [], []

    def responses(self, y):
        for target in self.task.targets:
            if self.task.classes:
                for i, label in enumerate(self.task.classes[target]):
                    yield target, f'{target}_{i}', (y[target] == label).astype(float)
            else:
                yield target, target, y[target].astype(float)

    @staticmethod
    def uncovered_prior(response, periods, uncovered, neutral):
        """Mean of strictly earlier periods, with a label-free first-period fallback.

        A row that no temporal fold ever validates is a warmup row. Its encoding
        may use labels observed before its own period, never its own label or a
        later one, so changing future targets cannot rewrite early features.
        """
        positions = np.flatnonzero(uncovered)
        if periods is None:
            return np.full(len(positions), neutral)
        ordered = np.asarray(periods)
        table = pd.DataFrame({'period': ordered, 'value': response.to_numpy()})
        stats = table.groupby('period', sort=True)['value'].agg(['sum', 'count'])
        earlier = stats.cumsum().shift(1)
        prior = (earlier['sum'] / earlier['count']).fillna(neutral).to_dict()
        return np.asarray([prior.get(ordered[position], neutral) for position in positions])

    @staticmethod
    def table(frame, response, columns, smoothing):
        key = pd.MultiIndex.from_frame(frame[columns].fillna('__missing__').astype(str))
        data = pd.DataFrame({'value': response.to_numpy()}, index=key)
        stats = data.groupby(level=list(range(len(columns))), sort=False)['value'].agg(['sum', 'count'])
        prior = float(response.mean())
        mapping = (stats['sum'] + smoothing * prior) / (stats['count'] + smoothing)
        if len(columns) == 1:
            mapping.index = pd.MultiIndex.from_tuples([(value,) for value in mapping.index])
        return mapping, prior

    @staticmethod
    def lookup(frame, columns, mapping, prior):
        key = pd.MultiIndex.from_frame(frame[columns].fillna('__missing__').astype(str))
        return mapping.reindex(key).fillna(prior).to_numpy()

    def fit_transform(self, frame, y, raw):
        result = frame.copy()
        for operation in self.spec.transformations:
            if operation.name not in {'history', 'target_encode'}:
                continue
            if y is None:
                raise ValueError('Supervised features require fit labels')
            p = operation.parameters
            columns = p.get('columns', [])
            if not columns or not set(columns) <= set(result):
                raise ValueError('Explicit observed entity/encoding columns are required')
            if operation.name == 'target_encode':
                from .research_evaluation import folds
                temporal = self.strategy in {'temporal', 'temporal_group'}
                splits = folds(raw, y, self.task, self.strategy, self.spec.seed,
                    min_folds=0 if temporal else 2)
                smoothing = float(p.get('smoothing', 20))
                if smoothing < 1:
                    raise ValueError('Target encoding requires positive regularization')
                for target, name, response in self.responses(y):
                    generated = '__te_' + '_'.join(columns) + '_' + name
                    if generated in result:
                        raise ValueError('Generated target-encoding feature collides')
                    neutral = 1.0 / max(1, len(self.task.classes[target])) if self.task.classes else 0.0
                    time_column = self.task.time_column
                    # A grouped window excludes every held group; an arbitrary
                    # past mean would reintroduce their labels. Multi-period
                    # horizons likewise have no row-specific observed origin.
                    periods = (ordered_periods(raw[time_column]).reindex(frame.index)
                        if self.strategy == 'temporal' and time_column in raw
                        and (self.task.kind != 'forecasting' or self.task.horizon == 1) else None)
                    values, covered = np.full(len(frame), neutral), np.zeros(len(frame), dtype=bool)
                    for a, b in splits:
                        mapping, prior = self.table(frame.iloc[a], response.iloc[a], columns, smoothing)
                        values[b] = self.lookup(frame.iloc[b], columns, mapping, prior)
                        covered[b] = True
                    if (~covered).any():
                        values[~covered] = self.uncovered_prior(response, periods, ~covered, neutral)
                    mapping, prior = self.table(frame, response, columns, smoothing)
                    self.encodings.append((generated, columns, mapping, prior))
                    result[generated] = values
                    self.report.append({'feature': generated, 'scope': 'external_split_aware_oof',
                        'coverage': float(covered.mean()),
                        'uncovered_prior': 'strictly_past_period_mean' if periods is not None else 'neutral',
                        'uncovered_fallback': neutral,
                        'key_semantics': 'joint_columns',
                        'ignored_parameters': sorted(set(p) - {'columns', 'smoothing'}),
                        'uncovered_rows': int((~covered).sum()), 'smoothing': smoothing})
            else:
                time_column = p.get('time_column', self.task.time_column)
                if not time_column or time_column not in raw:
                    raise ValueError('History requires an explicit time column')
                keys = pd.MultiIndex.from_frame(frame[columns].fillna('__missing__').astype(str))
                times = ordered_periods(raw[time_column])
                if times.isna().any():
                    raise ValueError('History times must be complete')
                for target, name, response in self.responses(y):
                    generated = '__history_' + '_'.join(columns) + '_' + name
                    table = pd.DataFrame({'key': list(keys), 'time': times.to_numpy(), 'value': response.to_numpy()})
                    table = table.groupby(['key', 'time'], sort=True)['value'].agg(['sum', 'count']).reset_index()
                    table['cumulative_sum'] = table.groupby('key')['sum'].cumsum()
                    table['cumulative_count'] = table.groupby('key')['count'].cumsum()
                    history = {key: part for key, part in table.groupby('key', sort=False)}
                    item = (generated, columns, time_column, history)
                    self.histories.append(item)
                    self.add_history(result, raw, item)
                    self.report.append({'feature': generated, 'scope': 'strictly_past_targets',
                        'entity_columns': columns, 'entities': len(history),
                        'same_entity_time_groups': int((table['count'] > 1).sum()),
                        'cold_fraction': float((result[generated + '_count'] == 0).mean()),
                        'identity_is_hypothesis': True})
        return result

    def add_history(self, result, raw, item):
        generated, columns, time_column, history = item
        keys = pd.MultiIndex.from_frame(result[columns].fillna('__missing__').astype(str))
        counts, means = np.zeros(len(result)), np.zeros(len(result))
        for i, (key, timestamp) in enumerate(zip(keys, ordered_periods(raw[time_column]))):
            part = history.get(key)
            if part is None:
                continue
            # Equal timestamps are excluded, including the current row itself.
            position = int(part['time'].searchsorted(timestamp, side='left')) - 1
            if position >= 0:
                counts[i] = part['cumulative_count'].iloc[position]
                means[i] = part['cumulative_sum'].iloc[position] / counts[i]
        for suffix, values in (('_mean', means), ('_count', counts), ('_cold', (counts == 0).astype(float))):
            if generated + suffix in result:
                raise ValueError('Generated history feature collides')
            result[generated + suffix] = values

    def transform(self, frame, raw):
        result = frame.copy()
        for generated, columns, mapping, prior in self.encodings:
            result[generated] = self.lookup(frame, columns, mapping, prior)
        for item in self.histories:
            self.add_history(result, raw, item)
        return result
