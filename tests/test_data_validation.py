from pathlib import Path
import json

import numpy as np
import pytest

from automl_agent.contracts import InspectArgs, TaskResolutionProposal
from automl_agent.data import Dataset
from automl_agent.validation import choose_validation, inner_split, pilot_indices
from conftest import write_task


def test_identifier_spelling_and_special_categories(tmp_path):
    path=write_task(tmp_path)
    d=Dataset(path); task,_=d.draft(); d.activate(task)
    assert d.test.row.iloc[0]=="00180"
    assert set(d.train.code)=={"NA","-"}
    assert not d.train.code.isna().any()
    assert d.train['empty'].isna().all()


def test_temporal_and_inner_splits_preserve_periods(tmp_path):
    d=Dataset(write_task(tmp_path,temporal=True)); t,_=d.draft(); d.activate(t); p=choose_validation(d)
    assert p.strategy=="temporal"
    assert set(d.train.iloc[p.control].when)=={4}
    assert set(d.train.iloc[p.validation].when)=={3}
    X=d.features(p.fit); y=d.train.iloc[p.fit][t.targets]
    a,b=inner_split(X,y,t,p.strategy,42)
    assert X.iloc[a].when.max()<X.iloc[b].when.min()


def test_groups_and_pilots_remain_whole(tmp_path):
    d=Dataset(write_task(tmp_path,groups=True)); t,_=d.draft(); d.activate(t); p=choose_validation(d)
    sets=[set(d.train.iloc[getattr(p,k)]['query']) for k in ("fit","validation","control")]
    assert not sets[0]&sets[1] and not sets[1]&sets[2] and not sets[0]&sets[2]
    X=d.features(p.fit).reset_index(drop=True); y=d.train.iloc[p.fit][t.targets].reset_index(drop=True)
    rows=pilot_indices(X,y,t,"group",25,17)
    groups=set(X.iloc[rows]['query'])
    assert len(rows)==int(X['query'].isin(groups).sum())


def test_fit_target_slices_only(tmp_path):
    d=Dataset(write_task(tmp_path)); t,_=d.draft(); d.activate(t); p=choose_validation(d)
    with pytest.raises(ValueError):d.inspect(InspectArgs(scope="validation_features",include_target=True),p)
    with pytest.raises(ValueError):d.inspect(InspectArgs(scope="test_features",columns=["answer"]),p)
    with pytest.raises(ValueError):d.inspect(InspectArgs(scope="validation_features",predicate={"column":"value","op":"gt","value":0}),p)
    report=d.inspect(InspectArgs(include_target=True,columns=["answer"],predicate={"column":"value","op":"gt","value":0}),p)
    assert report["scope"]=="fit"


def test_explicit_contract_cannot_override_source_hash(tmp_path):
    path=write_task(tmp_path); p=tmp_path/'task_spec.json'; values=json.loads(p.read_text()); values['source_hash']='wrong'; p.write_text(json.dumps(values))
    with pytest.raises(ValueError): Dataset(path).draft()


def test_conflicting_train_test_schema(tmp_path):
    path=write_task(tmp_path); p=tmp_path/'test.csv'; p.write_text(p.read_text().replace('value','novel',1))
    with pytest.raises(ValueError): Dataset(path).draft()


def test_multilabel_marginals_and_constant_label(tmp_path):
    d=Dataset(write_task(tmp_path,multilabel=True)); t,_=d.draft(); d.activate(t); p=choose_validation(d)
    assert p.fit and p.validation and p.control
    assert set(d.train.iloc[p.fit].answer)=={0,1}


def test_missing_ambiguous_contract_does_not_invent_metric(tmp_path):
    path=write_task(tmp_path); (tmp_path/'task_spec.json').unlink()
    d=Dataset(path); task,questions=d.draft()
    assert task is None and 'metric' in questions


def test_proposal_wrong_request_is_rejected(tmp_path):
    d=Dataset(write_task(tmp_path)); t,_=d.draft()
    proposal=TaskResolutionProposal(request_id='wrong',source_hash=d.source_hash,status='proposed',task_spec=t)
    with pytest.raises(ValueError): d.confirm_proposal(proposal,'expected')
