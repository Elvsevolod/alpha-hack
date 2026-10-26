import json
from types import SimpleNamespace

import pytest

from automl_agent.context_memory import experiment_summary, iteration_feedback
from automl_agent.contracts import PipelineSpec
from automl_agent.util import canonical
from context_compressor import Source
from test_layered_prompt_context import controller, read_all
from test_research_learning import trial


def measured(task, *, selected=216, applied=216):
    target=task.targets[0]
    return dict(status='completed',scope='full',training_summary=dict(rows=100,
        learning_curve={target:dict(iterations=selected,scope='fit_inner_holdout',private='CONTROL_SENTINEL')},
        budget=dict(iteration_schedule={target:applied})))


def test_compact_summary_distinguishes_recipe_limit_selection_and_application(tabular):
    _,_,task=tabular
    spec,_=trial('measured',1,pipeline=PipelineSpec(model='lightgbm',hyperparameters=dict(n_estimators=4000)))
    result=measured(task,applied=300)
    source=Source('experiment:measured','a'*64,'full','registered_feedback')
    change=dict(basis='standalone_recipe',declared_component='capacity',effective_fields=[],
        claimed_field_unchanged=None,mixed_change=False)
    summary=experiment_summary(SimpleNamespace(task=task),spec,result,source,change)
    schedule=summary['training_iterations']
    assert schedule['recipe_parameter']==dict(name='n_estimators',value=4000)
    assert schedule['targets']==[dict(target_index=0,selected=216,applied_schedule=300,selection_scope='fit_inner_holdout')]
    assert schedule['omitted_targets']==0 and summary['evidence_ref']['id']=='measured'
    assert 'CONTROL_SENTINEL' not in canonical(summary) and len(canonical(summary).encode())<=1536


@pytest.mark.parametrize('status',['timeout','error','outcome_unknown'])
def test_noncompleted_fit_does_not_claim_applied_schedule(tabular,status):
    _,_,task=tabular
    spec,_=trial('failed',1,pipeline=PipelineSpec(model='lightgbm'))
    result=measured(task);result['status']=status
    assert iteration_feedback(SimpleNamespace(task=task),spec,result) is None


def test_missing_schedule_is_not_filled_with_declared_cap(tabular):
    _,_,task=tabular
    spec,_=trial('no_measurement',1,pipeline=PipelineSpec(model='lightgbm',hyperparameters=dict(n_estimators=4000)))
    result=measured(task,selected=None,applied=None)
    result['training_summary']['budget']['iteration_schedule']['unknown_target']=999
    assert iteration_feedback(SimpleNamespace(task=task),spec,result) is None


def test_iteration_feedback_keeps_small_declared_parameters_ahead_of_extra_slices(tabular):
    _,_,task=tabular
    parameters=dict(n_estimators=4000,learning_rate=.015,num_leaves=63,min_child_samples=40,
        reg_lambda=5,colsample_bytree=.6,subsample=.7,subsample_freq=1)
    spec,_=trial('measured',1,pipeline=PipelineSpec(model='lightgbm',hyperparameters=parameters))
    result=measured(task)
    result['training_summary']['temporal_stability']=dict(status='complete',scope='fit_only_stress_windows',
        windows_hash='a'*64,scores={'expanding_1':54.65,'expanding_2':56.09})
    result['feedback']={str(i):dict(status='available',scope='external_validation',
        slices={'slice_'+str(i):dict(rows=100,score=70+i)}) for i in range(5)}
    source=Source('experiment:measured','v','full','registered_feedback')
    change=dict(basis='normalized_parent_recipe',declared_component='hyperparameters',
        effective_fields=['hyperparameters.'+key for key in parameters],
        claimed_field_unchanged=None,mixed_change=False)
    summary=experiment_summary(SimpleNamespace(task=task),spec,result,source,change)
    assert summary['parameters']==parameters and summary['training_iterations']['targets'][0]['selected']==216
    assert summary['effective_change']['field_count']==len(parameters)
    assert len(canonical(summary).encode())<=1536


def test_optuna_recipe_default_is_not_presented_as_selected_trial_cap(tabular):
    _,_,task=tabular
    spec,_=trial('tuned',1,pipeline=PipelineSpec(engine='optuna',model='lightgbm',hyperparameters=dict(n_estimators=150)))
    feedback=iteration_feedback(SimpleNamespace(task=task),spec,measured(task,selected=400,applied=400))
    assert 'recipe_parameter' not in feedback and feedback['targets'][0]['selected']==400


def test_wide_target_names_keep_indices_and_lossless_route_without_crowding_summary(tabular):
    _,_,task=tabular
    targets=['длинная целевая переменная '*100+str(i) for i in range(80)]
    task=task.model_copy(update=dict(targets=targets))
    result=measured(task)
    result['training_summary']['learning_curve']={t:dict(iterations=216,scope='fit_inner_holdout') for t in targets}
    result['training_summary']['budget']['iteration_schedule']={t:300 for t in targets}
    spec,_=trial('wide',1,pipeline=PipelineSpec(model='lightgbm'))
    feedback=iteration_feedback(SimpleNamespace(task=task),spec,result)
    assert feedback['omitted_targets']+len(feedback['targets'])==80
    assert feedback['omitted_targets']>0 and len(canonical(feedback).encode())<=512
    assert feedback['read']['arguments']['ids']==['wide']
    assert all('target_index' in row for row in feedback['targets'])
    assert targets[0] not in canonical(feedback)


def test_real_baseline_feedback_and_full_schedule_remain_readable_at_every_stage(controller):
    c=controller
    spec,result=c.lookup(c.journal.get('baseline_id'))
    expected=result.training_summary['budget']['iteration_schedule'][c.task.targets[0]]
    assert type(expected) is int and expected>0
    before=c.publisher.output.read_bytes()
    snap=c.context_memory.snapshot(c.system_prompt)
    for mode in ['native','json']:
        for stage in range(4):
            prepared=c.context_memory.compressor.prepare(snap,mode,budget=262144,minimum_stage=stage)
            wire=json.loads(json.loads(prepared.wire_json)['messages'][1]['content'])
            if stage==0:
                entry=next(d for d in wire['history_details'] if d['record_id']==spec.experiment_id)
                summary=entry['summary']
                feedback=summary['training_iterations']
                assert feedback['targets'][0]['applied_schedule']==expected
                route=entry['read']
            else:
                assert wire['history_details']==[]
                route=dict(tool='read_run_records',arguments=dict(ids=[wire['core']['current_work']['baseline_id']],max_chars=12000))
            detail=json.loads(read_all(c,route))
            assert detail['result']['training_summary']['budget']['iteration_schedule'][c.task.targets[0]]==expected
    assert c.publisher.output.read_bytes()==before and not c.journal.get('control_opened')
