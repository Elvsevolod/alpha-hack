import json
from pathlib import Path

import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import RunFilters, RunListArgs, RunReadArgs, TaskResolutionProposal
from automl_agent.controller import Controller
from automl_agent.data import Dataset,parse_metric
from automl_agent.validation import choose_validation
from conftest import write_task


def proposed(dataset,request_id='r',metric='mae'):
    spec=json.loads((dataset.task_path.parent/'task_spec.json').read_text())
    spec.update(contract_version=2,source_hash=dataset.source_hash,file_roles={name:name for name in ('train.csv','test.csv','sample_submission.csv')})
    spec['metric']={'name':metric}
    return {'request_id':request_id,'source_hash':dataset.source_hash,'status':'proposed','task_spec':spec,
        'evidence':{field:['task','inventory'] for field in spec}}


def test_v2_semantic_primary_metric_is_not_vetoed_by_first_regex_match(tmp_path):
    task=write_task(tmp_path)
    task.write_text('Evaluation metric summary: auxiliary RMSE is reported for comparison; the primary metric is MAE, lower is better. Regression answer. Preserve row IDs. Use the supplied feature schema.')
    data=Dataset(task,deferred=True)
    assert parse_metric(data.text).name=='rmse'  # Reproduce the old heuristic's ambiguity.
    proposal=TaskResolutionProposal.model_validate(proposed(data))
    assert data.confirm_proposal(proposal,'r').metric.name=='mae'


def test_dictionary_semantics_available_before_dataset_activation_without_labels(tmp_path):
    task=write_task(tmp_path,temporal=True)
    dictionary=tmp_path/'nested'/'roles.csv';dictionary.parent.mkdir()
    dictionary.write_text('feature,role,dtype,description_ru\nwhen,feature,numeric,Календарный год наблюдения\nrow,id,string,Идентификатор строки\nanswer,target,numeric,Предсказываемая величина\n')
    # Data looking like a dictionary is not descriptive unless its keys are schema fields.
    (tmp_path/'misleading.csv').write_text('feature,description\nrow_001,CLOSED_LABEL_SENTINEL\n')
    dataset=Dataset(task,deferred=True)
    assert not hasattr(dataset,'train')
    source=dataset.sources['description_csv:nested/roles.csv']
    assert 'Календарный год наблюдения' in source['text']
    assert 'description_csv:misleading.csv' not in dataset.sources
    assert 'CLOSED_LABEL_SENTINEL' not in json.dumps(dataset.sources,ensure_ascii=False)


def test_only_cited_supporting_material_remains_readable_for_research(tmp_path,monkeypatch):
    path=write_task(tmp_path)
    (tmp_path/'feature_notes.md').write_text('The value feature is measured in square meters.\n')
    (tmp_path/'unused_notes.md').write_text('UNREFERENCED_SOURCE_SENTINEL\n')
    c=Controller(path,Limits(seconds=120,cpu=1));monkeypatch.setattr(c.qwen,'key','offline');monkeypatch.setattr(c.qwen,'base_url','http://offline')
    def resolve(system,request):
        reply=proposed(c.dataset,request['request_id'])
        reply['evidence']['feature_columns']=['description:feature_notes.md']
        return reply
    monkeypatch.setattr(c.qwen,'resolve',resolve)
    try:
        c.task=c._task()
        assert c.task.evidence['feature_columns']==['description:feature_notes.md']
        c.plan=choose_validation(c.dataset)
        c.context_memory.snapshot('offline research context')
        listed=c.run_tools.search(RunListArgs(filters=RunFilters(kind='source',component='task_supporting_material')),catalog=True)
        assert listed['total']==1
        read=c.run_tools.read(RunReadArgs(ids=[listed['records'][0]['record_id']]))
        content=read['records'][0]['content']
        assert 'description:feature_notes.md' in content
        assert 'square meters' in content
        assert 'UNREFERENCED_SOURCE_SENTINEL' not in content
    finally:c.close()


def test_rejected_semantic_proposal_keeps_full_redacted_trace(tmp_path,monkeypatch):
    task=write_task(tmp_path);task.write_text('Regression. Evaluation metric: MAE. Preserve row IDs.')
    c=Controller(task,Limits(seconds=120,cpu=1));calls=[]
    monkeypatch.setattr(c.qwen,'key','offline');monkeypatch.setattr(c.qwen,'base_url','http://offline')
    def resolve(system,request):
        calls.append(request)
        if len(calls)==1:raise ValueError('malformed proposal sk-syntheticSecret0123456789')
        return proposed(c.dataset,request['request_id'])
    monkeypatch.setattr(c.qwen,'resolve',resolve)
    try:
        assert c._task().metric.name=='mae'
        row=c.journal.db.execute("SELECT payload FROM events WHERE kind='task_resolution_rejected'").fetchone()
        payload=json.loads(row[0]);ref=payload['diagnostic']['traceback_ref']
        text=(c.workdir/ref).read_text()
        assert 'Traceback' in text and 'ValueError' in text and 'sk-syntheticSecret0123456789' not in text
    finally:c.close()


def test_ambiguous_source_roles_halt_before_csv_loading(tmp_path,monkeypatch):
    import shutil
    task=write_task(tmp_path);(tmp_path/'task_spec.json').unlink()
    (tmp_path/'train.csv').rename(tmp_path/'observations_a.csv')
    shutil.copyfile(tmp_path/'observations_a.csv',tmp_path/'observations_b.csv')
    task.write_text('Regression. Two equally plausible observation sources are supplied; the training role is not designated. Primary metric MAE.')
    c=Controller(task,Limits(seconds=120,cpu=1));requests=[]
    monkeypatch.setattr(c.qwen,'key','offline');monkeypatch.setattr(c.qwen,'base_url','http://offline')
    monkeypatch.setattr(c.dataset,'bind',lambda *a:pytest.fail('Ambiguous roles must not activate CSV data'))
    def unresolved(system,request):
        requests.append(request)
        return {'request_id':request['request_id'],'source_hash':request['source_hash'],'status':'unresolved','unresolved_fields':['file_roles']}
    monkeypatch.setattr(c.qwen,'resolve',unresolved)
    try:
        result=c.run()
        assert result['state']=='contract_unresolved' and len(requests)==1
        assert not hasattr(c.dataset,'train') and not (tmp_path/'submission.csv').exists()
    finally:c.close()


def test_rejected_dictionary_never_reaches_next_qwen_request(tmp_path,monkeypatch):
    import pandas as pd
    path=write_task(tmp_path);path.write_text('Regression. Primary metric MAE. Preserve row IDs.')
    train=pd.read_csv(tmp_path/'train.csv');train.loc[0,'answer']=123.45678901
    train.to_csv(tmp_path/'train.csv',index=False);train.to_csv(tmp_path/'extra.csv',index=False)
    c=Controller(path,Limits(seconds=120,cpu=1));calls=[]
    monkeypatch.setattr(c.qwen,'key','offline');monkeypatch.setattr(c.qwen,'base_url','http://offline')
    def resolve(system,request):
        assert '123.45678901' not in json.dumps(request['sources'])
        assert not hasattr(c.dataset,'train')
        calls.append(request)
        reply=proposed(c.dataset,request['request_id'])
        if len(calls)==1:
            reply['task_spec']['file_roles']['data_dictionary.csv']='extra.csv';reply['evidence']={}
        return reply
    monkeypatch.setattr(c.qwen,'resolve',resolve)
    try:
        assert c._task().metric.name=='mae' and len(calls)==2
        assert '123.45678901' not in json.dumps(c.dataset.sources)
        assert 'dictionary' not in c.dataset.sources
    finally:c.close()


def test_rejected_proposal_rolls_back_binding_and_existing_source_reference(tmp_path):
    path=write_task(tmp_path)
    (tmp_path/'roles.csv').write_text('feature,description\nanswer,Predicted quantity\nrow,Identifier\n')
    data=Dataset(path,deferred=True);source_reference=data.sources;before=json.loads(json.dumps(data.sources))
    reply=proposed(data);reply['task_spec']['file_roles']['data_dictionary.csv']='roles.csv';reply['evidence']={}
    with pytest.raises(ValueError,match='source evidence'):
        data.confirm_proposal(TaskResolutionProposal.model_validate(reply),'r')
    assert data.sources is source_reference and data.sources==before
    assert not hasattr(data,'train') and data.paths=={}


@pytest.mark.parametrize('link_kind',['symlink','hardlink'])
@pytest.mark.parametrize('channel',['semantic','explicit'])
def test_source_alias_is_rejected_before_supervised_loading(tmp_path,link_kind,channel):
    import os
    path=write_task(tmp_path);alias=tmp_path/'test_alias.csv'
    if link_kind=='symlink':alias.symlink_to(tmp_path/'train.csv')
    else:os.link(tmp_path/'train.csv',alias)
    roles={'train.csv':'train.csv','test.csv':'test_alias.csv'}
    if channel=='explicit':
        f=tmp_path/'task_spec.json';values=json.loads(f.read_text());values['file_roles']=roles;f.write_text(json.dumps(values))
    data=Dataset(path,deferred=True)
    with pytest.raises(ValueError,match='same physical source'):
        if channel=='explicit':data.explicit()
        else:
            reply=proposed(data);reply['task_spec']['file_roles']=roles
            data.confirm_proposal(TaskResolutionProposal.model_validate(reply),'r')
    assert not hasattr(data,'train') and data.paths=={}


def test_changed_source_is_rejected_before_binding(tmp_path):
    path=write_task(tmp_path);data=Dataset(path,deferred=True)
    f=tmp_path/'train.csv';f.write_text(f.read_text()+'\n')
    with pytest.raises(ValueError,match='Source changed'):
        data.bind({'train.csv':'train.csv','test.csv':'test.csv'})
    assert not hasattr(data,'train') and data.paths=={}


def test_interrupted_confirmation_restores_binding(tmp_path,monkeypatch):
    path=write_task(tmp_path);data=Dataset(path,deferred=True);reference=data.sources;before=reference.copy()
    def interrupted(task):raise KeyboardInterrupt()
    monkeypatch.setattr(data,'validate',interrupted)
    with pytest.raises(KeyboardInterrupt):
        data.confirm_proposal(TaskResolutionProposal.model_validate(proposed(data)),'r')
    assert data.sources is reference and data.sources==before and not hasattr(data,'train')
