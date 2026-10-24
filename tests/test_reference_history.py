import numpy as np
import pandas as pd

from tools.solve_participant_examples import history_features


def test_link_history_excludes_current_and_future_labels():
    reference = pd.DataFrame({'signature': ['a'] * 4, 'period': [1, 2, 2, 3],
                              'target': [10., 20., 30., 1000.]})
    query = pd.DataFrame({'signature': ['a', 'a', 'b'], 'period': [1, 2, 3]})
    before = history_features(reference, query, ['signature'], 'target', 'period')
    assert np.isnan(before.iloc[0].history_last_mean)
    assert before.iloc[1].history_last_mean == 10
    assert np.isnan(before.iloc[2].history_last_mean)
    reference.loc[reference.period >= 2, 'target'] = -999
    after = history_features(reference, query, ['signature'], 'target', 'period')
    pd.testing.assert_frame_equal(before, after)


def test_ambiguous_links_average_same_period_and_preserve_support():
    reference = pd.DataFrame({'signature': ['a'] * 3, 'period': [1, 2, 2], 'target': [10., 20., 40.]})
    query = pd.DataFrame({'signature': ['a'], 'period': [3]}, index=[7])
    result = history_features(reference, query, ['signature'], 'target', 'period')
    assert result.loc[7, 'history_last_mean'] == 30
    assert result.loc[7, 'history_count'] == 3
    assert result.loc[7, 'history_gap'] == 1
    assert result.loc[7, 'history_mean'] == 70 / 3


def test_query_labels_are_never_read():
    reference = pd.DataFrame({'signature': ['a'], 'period': [1], 'target': [10.]})
    query = pd.DataFrame({'signature': ['a'], 'period': [2], 'target': [-1000.]})
    before = history_features(reference, query, ['signature'], 'target', 'period')
    query['target'] = 50000
    pd.testing.assert_frame_equal(before, history_features(reference, query, ['signature'], 'target', 'period'))
