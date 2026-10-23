import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from automl_agent.budget import Limits
from automl_agent.contracts import MetricSpec, OutputSpec, TaskSpec
from automl_agent.controller import Controller
from automl_agent.data import parse_metric
from automl_agent.evaluation import score_predictions
from automl_agent.compatibility import incompatibilities
from automl_agent.contracts import PipelineSpec
from conftest import write_task


def task_for(name, targets, classes=None, weights=None):
    return TaskSpec(kind=('multioutput_classification' if len(targets)>1 else 'classification') if classes else
        ('multioutput_regression' if len(targets)>1 else 'regression'), targets=targets, feature_columns=['x'],
        metric=MetricSpec(name=name, aggregation='weighted' if weights else 'uniform', target_weights=weights),
        classes=classes or {}, output=OutputSpec(columns=targets, prediction_type='labels' if classes else 'values'),
        source_hash='fixture')


@pytest.mark.parametrize('name,actual,predicted,expected,classes,weights', [
    ('mape', [[2],[4]], [[1],[4]], .25, None, None),
    ('mape', [[-2],[-4]], [[-1],[-6]], .5, None, None),
    ('mape', [[0],[0]], [[0],[0]], 0., None, None),
    ('mape', [[0],[1]], [[1],[1]], 1/(2*np.finfo(np.float64).eps), None, None),
    ('mape', [[1e-20]], [[2e-20]], 1e-20/np.finfo(np.float64).eps, None, None),
    ('mape', [[2,4],[4,8]], [[1,0],[2,0]], .75, None, None),
    ('mape', [[2,4],[4,8]], [[1,0],[2,0]], .875, None, [1,3]),
    ('balanced_accuracy', [[0],[0],[0],[1]], [[0],[0],[0],[0]], .5, [[0,1]], None),
    ('balanced_accuracy', [['a'],['a'],['b'],['c']], [['a'],['a'],['a'],['c']], 2/3, [['a','b','c']], None),
    ('balanced_accuracy', [[0],[0]], [[0],[1]], .5, [[0,1,2]], None),
    ('balanced_accuracy', [[0,0],[0,1],[1,0],[1,1]], [[0,1],[1,1],[1,1],[1,0]], .375, [[0,1],[0,1]], [1,3]),
])
def test_trusted_metric_arithmetic(name,actual,predicted,expected,classes,weights):
    names=[f'y{i}' for i in range(len(actual[0]))]
    task=task_for(name,names,dict(zip(names,classes)) if classes else None,weights)
    values=np.asarray(predicted)
    measured=score_predictions(task,pd.DataFrame(actual,columns=names),dict(zip(names,values.T)),task.output.prediction_type)
    assert measured==pytest.approx(expected,rel=1e-13,abs=1e-15)
    assert task.metric.maximize==(name=='balanced_accuracy')


@pytest.mark.parametrize('name', ['invented', 'silhouette', [], None])
def test_unknown_metric_has_structured_error(name):
    with pytest.raises(ValidationError) as caught:MetricSpec(name=name)
    assert caught.value.errors()[0]['ctx']['error'].code=='unsupported_metric'


def test_controller_preserves_unknown_metric_code(tmp_path):
    controller=Controller(write_task(tmp_path),Limits(seconds=60,cpu=1))
    try:
        with pytest.raises(ValidationError) as caught:MetricSpec(name='unknown')
        diagnostic=controller.error_diagnostic(caught.value,'task_resolution')
        assert diagnostic['error_code']=='unsupported_metric'
        assert (controller.workdir/diagnostic['traceback_ref']).is_file()
    finally:controller.close()


def test_old_metrics_remain_and_parser_distinguishes_balanced():
    for name in ('rmse','mse','mae','r2','f1','accuracy','roc_auc','log_loss','average_precision','ndcg','map'):
        assert MetricSpec(name=name).name==name
    assert parse_metric('Primary metric balanced accuracy').name=='balanced_accuracy'
    assert parse_metric('Primary metric MAPE').name=='mape'
    assert parse_metric('Primary metric accuracy').name=='accuracy'


def test_dataset_rejects_forged_probability_metric_label_output(tmp_path):
    from automl_agent.data import Dataset

    dataset = Dataset(write_task(tmp_path, kind="classification"), deferred=True)
    dataset.bind({"train.csv": "train.csv", "test.csv": "test.csv", "sample_submission.csv": "sample_submission.csv"})
    task = TaskSpec(kind="classification", targets=["answer"], id_columns=["row"],
        feature_columns=["value", "empty", "code"], categorical_columns=["code"],
        metric=MetricSpec(name="f1"),
        classes={"answer": ["negative", "positive"]},
        output=OutputSpec(columns=["row", "answer"], prediction_type="labels"), source_hash=dataset.source_hash)
    forged = task.model_copy(update={"metric": MetricSpec(name="roc_auc", positive_label="positive")})
    with pytest.raises(ValueError, match="Probability metrics require probability output"):
        dataset.validate(forged)


def test_balanced_accuracy_labels_and_probabilities():
    task=task_for('balanced_accuracy',['y'],{'y':['a','b','c']})
    y=pd.DataFrame({'y':['a','a','b','c']})
    probability=np.eye(3)[[0,0,0,2]]
    assert score_predictions(task,y,{'y':probability},'probabilities')==pytest.approx(2/3)
    with pytest.raises(ValueError,match='unknown labels'):
        score_predictions(task,y,{'y':np.array(['a','a','b','unknown'])},'labels')
    assert 'metric_adapter_unverified' in ';'.join(incompatibilities(PipelineSpec(engine='autogluon'),task))


@pytest.mark.parametrize('kind,name',[('regression','mape'),('classification','balanced_accuracy')])
def test_metric_full_controller_completion(tmp_path,kind,name):
    path=write_task(tmp_path,kind=kind)
    f=tmp_path/'task_spec.json';spec=json.loads(f.read_text());spec['metric']['name']=name;f.write_text(json.dumps(spec))
    controller=Controller(path,Limits(seconds=120,cpu=1,max_candidates=0))
    try:
        report=controller.run(no_llm=True)
        assert report['state']=='completed',report
        assert report['control_status']==report['refit_status']=='completed'
        output=pd.read_csv(tmp_path/'submission.csv',dtype={'row':str})
        assert len(output)==25 and output['row'].tolist()==pd.read_csv(tmp_path/'test.csv',dtype={'row':str})['row'].tolist()
    finally:controller.close()
