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


def test_model_validation_strategy_needs_protocol_evidence_not_column_roles(tmp_path):
    task=write_task(tmp_path,temporal=True,groups=True)
    task.write_text('The when field is the observation year; query is the group identifier.')
    data=Dataset(task,deferred=True)
    reply=proposed(data)
    reply['task_spec']['explicit_validation']='group'
    reply['evidence']['explicit_validation']=['task']
    with pytest.raises(ValueError,match='explicit_validation'):
        data.confirm_proposal(TaskResolutionProposal.model_validate(reply),'r')
    assert not hasattr(data,'train') and data.paths=={}


@pytest.mark.parametrize(('instruction','strategy'),[
    ('Use a random validation split.', 'random'),
    ('Use group-based validation.', 'group'),
    ('Use chronological validation.', 'temporal'),
    ('Use chronological split and group-based validation.', 'temporal_group'),
    ('Use chronological validation. Keep query groups separate.', 'temporal_group'),
    ('Validation on future periods follows past training periods.', 'temporal'),
    ('Используйте валидацию на будущих периодах.', 'temporal'),
    ('Разбейте выборку по группам.', 'group'),
    ('Use group-based validation and do not use random validation.', 'group'),
    ('Use group-based validation and avoid random validation.', 'group'),
    ('Use group-based validation and never use random validation.', 'group'),
    ('Groups must not overlap across train and validation.', 'group'),
    ('Internal validation must use disjoint subjects.', 'group'),
    ('Validation must use disjoint participants.', 'group'),
    ('Cross-validation shall use disjoint entities.', 'group'),
    ('Do not split groups across train and validation.', 'group'),
    ('Группы не должны пересекаться между train и validation.', 'group'),
    ('Groups must not be split across train and validation.', 'group'),
    ('Не разделяйте группы между train и validation.', 'group'),
    ('Prefer group-based validation.', 'group'),
    ('Рекомендуется групповая валидация.', 'group'),
    ('Use random validation. No random validation was used in the previous run.', 'random'),
])
def test_model_validation_strategy_accepts_matching_task_protocol(tmp_path,instruction,strategy):
    task=write_task(tmp_path,temporal=True,groups=True)
    task.write_text('The when field is the observation year; query is the group identifier. '+instruction)
    data=Dataset(task,deferred=True)
    reply=proposed(data)
    reply['task_spec']['explicit_validation']=strategy
    reply['evidence']['explicit_validation']=['task']
    resolved=data.confirm_proposal(TaskResolutionProposal.model_validate(reply),'r')
    assert resolved.explicit_validation==strategy


@pytest.mark.parametrize(('instruction','strategy'),[
    ('Use chronological validation.', 'group'),
    ('Use group-based validation.', 'temporal'),
])
def test_model_validation_strategy_rejects_protocol_mismatch(tmp_path,instruction,strategy):
    task=write_task(tmp_path,temporal=True,groups=True)
    task.write_text('The when field is the observation year; query is the group identifier. '+instruction)
    data=Dataset(task,deferred=True)
    reply=proposed(data)
    reply['task_spec']['explicit_validation']=strategy
    reply['evidence']['explicit_validation']=['task']
    with pytest.raises(ValueError,match='explicit_validation'):
        data.confirm_proposal(TaskResolutionProposal.model_validate(reply),'r')


@pytest.mark.parametrize(('instruction','strategy'),[
    ('Do not use group-based validation.', 'group'),
    ('Do not use chronological validation.', 'temporal'),
    ('Group-based validation should not be used.', 'group'),
    ('Групповая валидация не должна использоваться.', 'group'),
    ('Group-based validation is used in the previous run.', 'group'),
    ('Групповая валидация используется в предыдущем запуске.', 'group'),
    ('Group-based validation follows the preprocessing pipeline used in the previous run.', 'group'),
    ('Use group-based validation. Group-based validation should not be used.', 'group'),
    ('Use group-based validation. You shouldn\'t use group-based validation.', 'group'),
    ('Use group-based validation. Avoid using group-based validation.', 'group'),
    ('Use group-based validation. You are prohibited from using group-based validation.', 'group'),
    ('Используйте групповую валидацию. Групповую валидацию использовать нельзя.', 'group'),
    ('Use group-based validation. Group-based validation is prohibited.', 'group'),
    ('Use group-based validation. Group-based validation is forbidden.', 'group'),
    ('Use group-based validation. Group-based validation is disallowed.', 'group'),
    ('Use group-based validation. Group-based validation is banned.', 'group'),
    ('Use group-based validation. Group-based validation is not permitted.', 'group'),
    ('Use group-based validation. Group-based validation isn\'t allowed.', 'group'),
    ('Use group-based validation. Group-based validation isn\'t recommended.', 'group'),
    ('Use group-based validation. Group-based validation is discouraged.', 'group'),
    ('Use group-based validation. Group-based validation cannot be used.', 'group'),
    ('Use group-based validation. Group-based validation can\'t be used.', 'group'),
    ('Use group-based validation. Group-based validation shouldn\'t be used.', 'group'),
    ('Use group-based validation. Group-based validation mustn\'t be used.', 'group'),
    ('Use group-based validation. Group-based validation must be avoided.', 'group'),
    ('Use group-based validation. Group-based validation should be avoided.', 'group'),
    ('Use group-based validation but do not use group-based validation.', 'group'),
    ('Groups must not overlap across train and validation, and group-based validation should not be used.', 'group'),
    ('Use group-based validation as an example in the report.', 'group'),
    ('Use group-based validation or random validation.', 'group'),
    ('Групповая валидация не подходит для этой задачи.', 'group'),
    ('Groups must not overlap within the source dataset.', 'group'),
    ('Groups must not overlap within the test dataset.', 'group'),
    ('Группы не должны пересекаться внутри исходного набора.', 'group'),
    ('Use random validation. Random validation isn\'t recommended.', 'random'),
    ('Use random validation. You shouldn\'t use random validation.', 'random'),
    ('Use random validation. You are prohibited from using random validation.', 'random'),
    ('Use random validation. No random validation should be used.', 'random'),
    ('Use random validation. No random validation should ever be used.', 'random'),
    ('Use random validation. No random validation should be performed.', 'random'),
    ('Use random validation. Random validation must not be performed.', 'random'),
    ('Используйте случайную валидацию. Случайную валидацию использовать нельзя.', 'random'),
    ('Use chronological validation. Chronological validation is discouraged.', 'temporal'),
    ('Use chronological validation. You shouldn\'t use chronological validation.', 'temporal'),
    ('Use chronological validation. You are prohibited from using chronological validation.', 'temporal'),
    ('Use chronological validation. Chronological validation should never be used.', 'temporal'),
    ('Use chronological validation. Chronological validation should never be performed.', 'temporal'),
    ('Use group-based validation. Group-based validation should never be used.', 'group'),
    ('Use group-based validation. Group-based validation must not be performed.', 'group'),
    ('Train on past years and test on future periods.', 'temporal'),
])
def test_model_validation_strategy_rejects_negation_and_test_chronology_only(tmp_path,instruction,strategy):
    task=write_task(tmp_path,temporal=True,groups=True)
    task.write_text('The when field is the observation year; query is the group identifier. '+instruction)
    data=Dataset(task,deferred=True)
    reply=proposed(data)
    reply['task_spec']['explicit_validation']=strategy
    reply['evidence']['explicit_validation']=['task']
    with pytest.raises(ValueError,match='explicit_validation'):
        data.confirm_proposal(TaskResolutionProposal.model_validate(reply),'r')


@pytest.mark.parametrize(('instruction','strategy'),[
    ('Use random validation instead of group-based validation.', 'group'),
    ('Use group-based validation, not temporal validation.', 'temporal_group'),
    ('Use the handbook to see that the previous run used group-based validation.', 'group'),
])
def test_model_validation_strategy_rejects_alternatives_and_unrelated_directives(tmp_path,instruction,strategy):
    task=write_task(tmp_path,temporal=True,groups=True)
    task.write_text('The when field is the observation year; query is the group identifier. '+instruction)
    data=Dataset(task,deferred=True)
    reply=proposed(data)
    reply['task_spec']['explicit_validation']=strategy
    reply['evidence']['explicit_validation']=['task']
    with pytest.raises(ValueError,match='explicit_validation'):
        data.confirm_proposal(TaskResolutionProposal.model_validate(reply),'r')


@pytest.mark.parametrize(('instruction','strategy'),[
    ('Group-based validation is one option in the guide.', 'group'),
    ('The previous run used disjoint subjects for internal validation.', 'group'),
    ('Internal validation must not use disjoint subjects.', 'group'),
    ('Internal validation must use disjoint subjects or random validation.', 'group'),
    ('The guide says internal validation must use disjoint subjects.', 'group'),
    ('The previous run used group-based validation.', 'group'),
    ('Group-based validation is one option. Chronological validation is another.', 'temporal_group'),
])
def test_model_validation_strategy_rejects_descriptive_options_and_history(tmp_path,instruction,strategy):
    task=write_task(tmp_path,temporal=True,groups=True)
    task.write_text('The when field is the observation year; query is the group identifier. '+instruction)
    data=Dataset(task,deferred=True)
    reply=proposed(data)
    reply['task_spec']['explicit_validation']=strategy
    reply['evidence']['explicit_validation']=['task']
    with pytest.raises(ValueError,match='explicit_validation'):
        data.confirm_proposal(TaskResolutionProposal.model_validate(reply),'r')


def test_descriptive_alternatives_do_not_block_a_later_direct_protocol(tmp_path):
    task=write_task(tmp_path,temporal=True,groups=True)
    task.write_text('Possible methods include group-based or random validation. Use chronological validation.')
    data=Dataset(task,deferred=True)
    reply=proposed(data)
    reply['task_spec']['explicit_validation']='temporal'
    reply['evidence']['explicit_validation']=['task']
    resolved=data.confirm_proposal(TaskResolutionProposal.model_validate(reply),'r')
    assert resolved.explicit_validation=='temporal'


def test_structural_source_cannot_substitute_for_cited_protocol_text(tmp_path):
    task=write_task(tmp_path,temporal=True,groups=True)
    task.write_text('The when field is the observation year; query is the group identifier. Use group-based validation.')
    data=Dataset(task,deferred=True)
    reply=proposed(data)
    reply['task_spec']['explicit_validation']='group'
    reply['evidence']['explicit_validation']=['inventory']
    with pytest.raises(ValueError,match='explicit_validation'):
        data.confirm_proposal(TaskResolutionProposal.model_validate(reply),'r')


def test_arbitrary_supporting_notes_cannot_prescribe_validation(tmp_path):
    task=write_task(tmp_path,temporal=True,groups=True)
    task.write_text('The when field is the observation year; query is the group identifier.')
    notes=tmp_path/'analyst_notes.md'
    notes.write_text('Use group-based validation.')
    readme=tmp_path/'README.md'
    readme.write_text('Use random validation.')
    data=Dataset(task,deferred=True)
    reply=proposed(data)
    reply['task_spec']['explicit_validation']='group'
    reply['evidence']['explicit_validation']=['description:analyst_notes.md']
    assert data.sources['description:analyst_notes.md']['scope']=='supporting_material'
    with pytest.raises(ValueError,match='explicit_validation'):
        data.confirm_proposal(TaskResolutionProposal.model_validate(reply),'r')


def test_generic_readme_and_description_are_not_organizer_instruction_sources(tmp_path):
    task=write_task(tmp_path)
    (tmp_path/'README.md').write_text('General project notes.')
    (tmp_path/'description.md').write_text('Additional analysis notes.')
    (tmp_path/'instructions.md').write_text('General workspace instructions.')
    (tmp_path/'rules.md').write_text('General workspace rules.')
    data=Dataset(task,deferred=True)
    assert data.sources['description:README.md']['scope']=='supporting_material'
    assert data.sources['description:description.md']['scope']=='supporting_material'
    assert data.sources['description:instructions.md']['scope']=='supporting_material'
    assert data.sources['description:rules.md']['scope']=='supporting_material'


def test_organizer_instruction_document_can_prescribe_validation(tmp_path):
    task=write_task(tmp_path,temporal=True,groups=True)
    task.write_text('The when field is the observation year; query is the group identifier.')
    rules=tmp_path/'organizer_rules.txt'
    rules.write_text('Use group-based validation.')
    data=Dataset(task,deferred=True)
    reply=proposed(data)
    reply['task_spec']['explicit_validation']='group'
    reply['evidence']['explicit_validation']=['description:organizer_rules.txt']
    assert data.sources['description:organizer_rules.txt']['scope']=='organizer_instruction'
    resolved=data.confirm_proposal(TaskResolutionProposal.model_validate(reply),'r')
    assert resolved.explicit_validation=='group'


def test_local_task_spec_validation_strategy_is_preserved(tmp_path):
    task_path=write_task(tmp_path,temporal=True,groups=True)
    spec_path=tmp_path/'task_spec.json'
    spec=json.loads(spec_path.read_text())
    spec['explicit_validation']='group'
    spec_path.write_text(json.dumps(spec))
    data=Dataset(task_path)
    task,_=data.draft()
    data.task=task
    assert data.fixed['explicit_validation']=='group'
    assert choose_validation(data).strategy=='group'


def test_future_time_and_group_inference_remains_temporal_group_without_explicit_protocol(tmp_path):
    task_path=write_task(tmp_path,temporal=True,groups=True)
    data=Dataset(task_path)
    task,_=data.draft()
    data.task=task
    assert task.explicit_validation is None
    assert choose_validation(data).strategy=='temporal_group'


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
