import json

import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import TaskResolutionProposal, TaskSpec
from automl_agent.controller import Controller
from automl_agent.data import Dataset
from automl_agent.publication import constant_bundle
from automl_agent.validation import choose_validation
from conftest import write_task


def task_values():
    return {'kind': 'regression', 'targets': ['y'], 'id_columns': ['id'], 'feature_columns': ['x'],
        'metric': {'name': 'rmse'}, 'output': {'columns': ['id', 'y'], 'prediction_type': 'values'},
        'source_hash': 'fixture'}


@pytest.mark.parametrize('kind,classes', [
    ('regression', {'x': ['a', 'b']}), ('regression', {'y': [1, 2]}),
    ('classification', {'x': ['a', 'b']})])
def test_class_metadata_cannot_describe_features_or_a_numeric_task(kind, classes):
    values = task_values(); values.update(kind=kind, classes=classes)
    if kind == 'classification':
        values.update(metric={'name': 'f1'}, output={'columns': ['id', 'y'], 'prediction_type': 'labels'})
    with pytest.raises(ValueError, match='Class labels'):
        TaskSpec.model_validate(values)


def test_classification_target_labels_and_empty_regression_metadata_remain_valid():
    values = task_values()
    assert TaskSpec.model_validate(values).classes == {}
    values.update(kind='classification', metric={'name': 'f1'}, classes={'y': ['a', 'b']},
        output={'columns': ['id', 'y'], 'prediction_type': 'labels'})
    assert TaskSpec.model_validate(values).classes == {'y': ['a', 'b']}


@pytest.mark.parametrize('labels', [[], ['a', 'a'], [float('nan')], [float('inf')]])
def test_invalid_declared_labels_are_rejected_but_unspecified_labels_are_allowed(labels):
    values = task_values()
    values.update(kind='classification', metric={'name': 'f1'},
        output={'columns': ['id', 'y'], 'prediction_type': 'labels'})
    assert TaskSpec.model_validate(values).classes == {}
    with pytest.raises(ValueError, match='Class labels'):
        TaskSpec.model_validate({**values, 'classes': {'y': labels}})


@pytest.mark.parametrize('declared', [['a'], [0, 1]])
def test_inconsistent_declared_labels_are_rejected_before_activation(tmp_path, declared):
    import pandas as pd
    path = write_task(tmp_path)
    frame = pd.read_csv(tmp_path / 'train.csv', dtype=str, keep_default_na=False)
    frame['answer'] = ['a' if i % 2 else 'b' for i in range(len(frame))]
    frame.to_csv(tmp_path / 'train.csv', index=False)
    data = Dataset(path, deferred=True)
    values = json.loads((tmp_path / 'task_spec.json').read_text())
    values.update(contract_version=2, source_hash=data.source_hash, kind='classification',
        metric={'name': 'f1'}, classes={'answer': declared},
        output={'columns': values['output']['columns'], 'prediction_type': 'labels'},
        file_roles={n: n for n in ('train.csv', 'test.csv', 'sample_submission.csv')})
    proposal = TaskResolutionProposal.model_validate({'request_id': 'request', 'source_hash': data.source_hash,
        'status': 'proposed', 'task_spec': values, 'evidence': {f: ['task', 'inventory'] for f in values}})
    with pytest.raises(ValueError, match='(?:Observed label outside declared class set|Non-numeric target)') as error:
        data.confirm_proposal(proposal, 'request')
    assert not hasattr(data, 'train') and 'answer' in str(error.value)


def test_numeric_class_labels_validate_without_mutating_data(tmp_path):
    import pandas as pd
    path = write_task(tmp_path)
    frame = pd.read_csv(tmp_path / 'train.csv', dtype=str, keep_default_na=False)
    frame['answer'] = [str(i % 2) for i in range(len(frame))]
    frame.to_csv(tmp_path / 'train.csv', index=False)
    data = Dataset(path)
    values = json.loads((tmp_path / 'task_spec.json').read_text())
    values.update(source_hash=data.source_hash, kind='classification', metric={'name': 'f1'},
        classes={'answer': [0, 1]}, output={'columns': values['output']['columns'], 'prediction_type': 'labels'})
    task = TaskSpec.model_validate(values)
    before = data.train.copy(deep=True)
    data.validate(task)
    pd.testing.assert_frame_equal(data.train, before)
    assert data.activate(task).classes == {'answer': [0, 1]}


def test_controller_repairs_feature_classes_before_first_csv(tmp_path, monkeypatch):
    c = Controller(write_task(tmp_path), Limits(seconds=120, cpu=1)); requests = []
    monkeypatch.setattr(c.qwen, 'key', 'offline-fixture')
    monkeypatch.setattr(c.qwen, 'base_url', 'http://offline')
    def resolve(system, request):
        requests.append(request)
        values = json.loads((tmp_path / 'task_spec.json').read_text())
        values.update(contract_version=2, source_hash=c.dataset.source_hash,
            file_roles={n: n for n in ('train.csv', 'test.csv', 'sample_submission.csv')},
            classes={'code': ['NA', '-']} if len(requests) == 1 else {})
        if len(requests) == 2:
            assert 'Class labels' in request['validation_errors'][0]
        assert len(requests) <= 2
        return {'request_id': request['request_id'], 'source_hash': c.dataset.source_hash,
            'status': 'proposed', 'task_spec': values, 'evidence': {f: ['task', 'inventory'] for f in values}}
    monkeypatch.setattr(c.qwen, 'resolve', resolve)
    try:
        task = c._task()
        assert len(requests) == 2 and task.classes == {}
        bundle = constant_bundle(c.dataset, choose_validation(c.dataset))
        # Only the safety template is published before prepare; no trained file.
        assert bundle is not None
        assert (tmp_path / 'submission.csv').read_bytes() == (tmp_path / 'sample_submission.csv').read_bytes()
    finally:
        c.close()
