"""Join only inventoried supplied tables, preserving base row identity."""
import numpy as np
import pandas as pd
from .util import file_hash
from .validation import ordered_periods


def attach(dataset, task):
    binding = task.hash
    if getattr(dataset, '_related_binding', None) == binding:
        return
    if hasattr(dataset, '_base_train'):
        dataset.train, dataset.test = dataset._base_train.copy(), dataset._base_test.copy()
    else:
        dataset._base_train, dataset._base_test = dataset.train.copy(), dataset.test.copy()
    for relation in task.related_tables:
        item = dataset.catalog.get(relation.file)
        if item is None:
            raise ValueError('Related table is outside the supplied inventory')
        path = (dataset.task_path.parent / relation.file).resolve()
        if not path.is_relative_to(dataset.task_path.parent.resolve()) or file_hash(path) != item['sha256']:
            raise ValueError('Related table path or source hash changed')
        if path in dataset.paths.values():
            raise ValueError('Related table cannot be a base train/test/output source')
        from .data import read_csv
        table = read_csv(path)
        required = set(relation.keys + list(relation.aggregations) + ([relation.time_column] if relation.time_column else []))
        if not required <= set(table) or set(task.targets) & set(table):
            raise ValueError('Related table has missing roles or task target columns')
        if len(relation.keys) != len(relation.base_keys):
            raise ValueError('Join key arity differs')
        for column in relation.aggregations:
            table[column] = pd.to_numeric(table[column], errors='raise')
        table_keys = pd.MultiIndex.from_frame(table[relation.keys].fillna('__missing__').astype(str))
        table = table.copy(); table['__join_key'] = list(table_keys)
        if relation.time_column:
            table[relation.time_column] = ordered_periods(table[relation.time_column])
        for attribute in ('train', 'test'):
            frame = getattr(dataset, attribute).copy()
            if not set(relation.base_keys) <= set(frame):
                raise ValueError('Base join keys are missing')
            query_keys = pd.MultiIndex.from_frame(frame[relation.base_keys].fillna('__missing__').astype(str))
            if relation.time_column:
                cutoff = relation.cutoff_column or task.time_column
                if not cutoff or cutoff not in frame:
                    raise ValueError('Event joins require a query cutoff')
                query_times = ordered_periods(frame[cutoff])
                groups = {key: part.sort_values(relation.time_column, kind='stable') for key, part in table.groupby('__join_key', sort=False)}
                positions_by_key = {}
                for position, key in enumerate(query_keys):
                    positions_by_key.setdefault(key, []).append(position)
                for column, aggregations in relation.aggregations.items():
                    for aggregation in aggregations:
                        name = relation.prefix + '__' + column + '__' + aggregation
                        if name in frame:
                            raise ValueError('Related feature name collision')
                        values = np.full(len(frame), np.nan)
                        for key, positions in positions_by_key.items():
                            positions = np.asarray(positions, dtype=int)
                            part = groups.get(key)
                            if part is None:
                                continue
                            n = part[relation.time_column].searchsorted(query_times.iloc[positions].to_numpy(), side='left')
                            source = part[column].to_numpy(dtype=float)
                            valid = np.isfinite(source)
                            count = np.r_[0, np.cumsum(valid)]
                            total = np.r_[0, np.cumsum(np.where(valid, source, 0))]
                            if aggregation == 'count': result = count[n]
                            elif aggregation == 'sum': result = total[n]
                            elif aggregation == 'mean': result = np.divide(total[n], count[n], out=np.full(len(n), np.nan), where=count[n]>0)
                            elif aggregation == 'min': result = np.r_[np.nan, np.minimum.accumulate(np.where(valid, source, np.inf))][n]
                            elif aggregation == 'max': result = np.r_[np.nan, np.maximum.accumulate(np.where(valid, source, -np.inf))][n]
                            else: result = np.where(n>0, source[np.maximum(n-1, 0)], np.nan)
                            values[positions] = result
                        values[~np.isfinite(values)] = np.nan
                        frame[name] = values
            else:
                for column, aggregations in relation.aggregations.items():
                    for aggregation in aggregations:
                        name = relation.prefix + '__' + column + '__' + aggregation
                        if name in frame:
                            raise ValueError('Related feature name collision')
                        mapping = table.groupby('__join_key')[column].agg(aggregation)
                        frame[name] = pd.Series(list(query_keys)).map(mapping).to_numpy()
            setattr(dataset, attribute, frame)
    dataset._related_binding = binding
