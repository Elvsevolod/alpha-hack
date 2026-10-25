import json

import pytest

from tools.research_memory_evaluation import evaluate


@pytest.mark.parametrize('kind,target,metric,values,expected', [
    ('classification', 'label', 'f1', ['a', 'b'], 1.),
    ('regression', 'value', 'rmse', [2., 5.], 0.),
])
@pytest.mark.parametrize('declared_classes', [True, False])
def test_evaluator_grades_exact_published_csv_and_freezes_before_labels(
        tmp_path, kind, target, metric, values, expected, declared_classes):
    import pandas as pd
    output = tmp_path / 'series'
    case = output / 'runs' / 'synthetic_B' / 'task'
    case.mkdir(parents=True)
    sealed = tmp_path / 'sealed' / 'synthetic'
    sealed.mkdir(parents=True)
    pd.DataFrame({'id': ['001', '002'], 'x': [3, 4]}).to_csv(case / 'test.csv', index=False)
    answer = pd.DataFrame({'id': ['001', '002'], target: values})
    answer.to_csv(case / 'submission.csv', index=False)
    answer.to_csv(case / 'train.csv', index=False)
    answer.to_csv(sealed / 'labels.csv', index=False)
    reference = {'kind': kind, 'targets': [target], 'id_columns': ['id'], 'feature_columns': ['x'],
        'metric': {'name': metric}, 'output': {'columns': ['id', target],
        'prediction_type': 'labels' if kind == 'classification' else 'values'}}
    if kind == 'classification' and declared_classes:
        reference['classes'] = {target: ['a', 'b']}
    (sealed / 'reference_contract.json').write_text(json.dumps(reference))
    protocol = {'commits': {'B': 'frozen'}, 'runs': [{'id': 'synthetic_B', 'task': 'synthetic',
        'arm': 'B', 'state': 'exited'}], 'labels_source_outside_tasks': str(sealed.parent)}
    (output / 'protocol.json').write_text(json.dumps(protocol))
    (output / 'technical_audit.json').write_text('{}')
    evaluate(output)
    score, = json.loads((output / 'external_evaluation.json').read_text())
    assert score['status'] == 'scored' and score['external_score'] == expected
    assert json.loads((output / 'protocol.json').read_text())['evaluation_frozen_at']
    answer.iloc[::-1].to_csv(case / 'submission.csv', index=False)
    evaluate(output)
    score, = json.loads((output / 'external_evaluation.json').read_text())
    assert score['status'] == 'not_scored'
    assert 'ordering' in score['reason']


def test_evaluation_cannot_open_labels_while_variants_are_pending(tmp_path):
    (tmp_path / 'protocol.json').write_text(json.dumps({'runs': [{'state': 'registered'}]}))
    with pytest.raises(ValueError, match='every registered variant'):
        evaluate(tmp_path)
    assert not (tmp_path / 'external_evaluation.json').exists()
