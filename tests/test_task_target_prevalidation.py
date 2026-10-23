import json

import pandas as pd
import pytest

from automl_agent.contracts import TaskSpec
from automl_agent.data import Dataset
from conftest import write_task


def target_case(tmp_path, kind, first, classes=None, positive=None, all_zero=False):
    path = write_task(tmp_path)
    frame = pd.read_csv(tmp_path / 'train.csv', dtype=str, keep_default_na=False)
    frame['answer'] = [str(0 if all_zero else i % 2) for i in range(len(frame))]
    if first is not None:
        frame.loc[0, 'answer'] = first
    frame.to_csv(tmp_path / 'train.csv', index=False)
    data = Dataset(path)
    values = json.loads((tmp_path / 'task_spec.json').read_text())
    values.update(source_hash=data.source_hash, kind=kind, classes=classes or {},
        metric={'name': 'rmse'} if kind == 'regression' else {'name': 'f1', 'positive_label': positive},
        output={'columns': values['output']['columns'],
            'prediction_type': 'values' if kind == 'regression' else 'labels'})
    return data, TaskSpec.model_validate(values)


@pytest.mark.parametrize('kind,first,classes,positive,error', [
    ('regression', 'PRIVATE_TARGET_SENTINEL', None, None, 'Non-numeric target'),
    ('regression', '18446744073709551616', None, None, 'Non-numeric target'),
    ('regression', 'inf', None, None, 'Nonfinite training target'),
    ('regression', '', None, None, 'Nonfinite training target'),
    ('classification', 'PRIVATE_TARGET_SENTINEL', None, 1, 'Non-numeric target'),
    ('classification', '', None, None, 'Missing training target'),
    ('multilabel', '0.5', None, None, 'Multilabel targets must be 0/1'),
    ('multilabel', None, {'answer': ['0', '1']}, None, 'Observed label outside declared class set'),
])
def test_target_errors_are_detected_before_activation_without_mutation(tmp_path, kind, first, classes, positive, error):
    data, task = target_case(tmp_path, kind, first, classes, positive)
    before = data.train.copy(deep=True)
    with pytest.raises(ValueError, match=error) as caught:
        data.validate(task)
    assert 'PRIVATE_TARGET_SENTINEL' not in str(caught.value)
    import traceback
    detail = ''.join(traceback.format_exception(caught.value))
    assert 'PRIVATE_TARGET_SENTINEL' not in detail and '18446744073709551616' not in detail
    pd.testing.assert_frame_equal(before, data.train)
    assert data.task is None


@pytest.mark.parametrize('classes', [{}, {'answer': [0, 1]}])
def test_valid_binary_multilabel_values_validate_before_integer_activation(tmp_path, classes):
    data, task = target_case(tmp_path, 'multilabel', None, classes)
    before = data.train.copy(deep=True)
    data.validate(task)
    pd.testing.assert_frame_equal(before, data.train)
    data.activate(task)
    assert set(data.train.answer) == {0, 1} and task.classes == {'answer': [0, 1]}


def test_constant_multilabel_target_still_requires_the_full_binary_class_universe(tmp_path):
    data, task = target_case(tmp_path, 'multilabel', None, {'answer': [0]}, all_zero=True)
    with pytest.raises(ValueError, match='Observed label outside declared class set'):
        data.validate(task)
