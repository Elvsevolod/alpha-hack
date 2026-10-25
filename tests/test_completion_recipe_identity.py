"""Forecast and executor must agree on the identity of a reusable frozen fit."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from automl_agent.completion import estimate
from automl_agent.contracts import PipelineSpec
from automl_agent.controller import Controller
from automl_agent.util import atomic_json,digest,file_hash


def cache_fixture(tmp_path,source='policy',minimal=False):
    pipeline=PipelineSpec(model='lightgbm',hyperparameters={'n_estimators':17},
        iteration_schedule={'answer':17},iteration_scale_rows=5)
    full=pipeline.model_dump(mode='json')
    legacy=pipeline.model_dump(mode='json',exclude_defaults=True)
    rows=list(range(10));keys=['r'+str(i) for i in rows]
    attempt=tmp_path/'attempts'/'candidate';attempt.mkdir(parents=True)
    original=attempt/'worker_result.json'
    atomic_json(original,{'frozen_pipeline':legacy if minimal else full})
    result=SimpleNamespace(status='completed',training_summary={'rows':5,'seconds':2.},seconds=2.,
        artifacts={'directory':str(attempt),'worker_manifest':str(original)})
    spec=SimpleNamespace(pipeline=pipeline)
    data={'completion_costs':[]}
    if source=='policy':
        data['finalization_plan']={'policies':{'candidate':{'experiment_id':'candidate',
            'pipeline_hash':pipeline.hash,'frozen_pipeline':full}}}
    journal=SimpleNamespace(get=data.get,event=lambda *a:None)
    c=SimpleNamespace(workdir=tmp_path,plan=SimpleNamespace(fit=list(range(5))),
        dataset=SimpleNamespace(row_keys=keys),lookup=lambda identifier:(spec,result),journal=journal)
    directory=tmp_path/'finalization'/'control'/'candidate';directory.mkdir(parents=True)
    prediction=directory/'control_predictions.json';prediction.write_text('{}')
    saved={'status':'completed','fit_row_keys_hash':digest(keys),
        'frozen_pipeline':deepcopy(legacy if minimal else full),'hashes':{prediction.name:file_hash(prediction)}}
    atomic_json(directory/'worker_result.json',saved)
    return c,pipeline,rows,directory,data,saved


@pytest.mark.parametrize('source',['policy','manifest'])
@pytest.mark.parametrize('minimal',[False,True])
def test_equivalent_saved_recipe_is_ready_for_forecast_and_executor(tmp_path,source,minimal):
    c,pipeline,rows,directory,_,_=cache_fixture(tmp_path,source,minimal)
    if source=='policy':
        # Existing executor already accepts absent default fields in old worker JSON.
        path,_=Controller._prepare_finalist(c,'candidate',rows,'control',.01)
        assert path==directory
    graph=estimate(c,['candidate'],rows,'control')
    assert graph['nodes'][0]['pipeline_hash']==pipeline.hash
    assert graph['nodes'][0]['ready'] is True
    assert graph['seconds']==graph['disk_bytes']==0


@pytest.mark.parametrize('damage',['parameter','rows','artifact','malformed_recipe'])
def test_changed_or_invalid_preparation_is_not_forecast_as_ready(tmp_path,damage):
    c,pipeline,rows,directory,_,saved=cache_fixture(tmp_path)
    if damage=='parameter':saved['frozen_pipeline']['hyperparameters']['n_estimators']=18
    elif damage=='rows':saved['fit_row_keys_hash']='foreign'
    elif damage=='artifact':(directory/'control_predictions.json').write_text('changed')
    else:saved['frozen_pipeline']={'hyperparameters':'invalid'}
    atomic_json(directory/'worker_result.json',saved)
    graph=estimate(c,['candidate'],rows,'control')
    assert graph['nodes'][0]['ready'] is False and graph['seconds']>0
    with pytest.raises(ValueError):Controller._prepare_finalist(c,'candidate',rows,'control',.01)


def test_observed_cost_uses_canonical_recipe_identity_from_legacy_manifest(tmp_path):
    c,pipeline,rows,_,data,_=cache_fixture(tmp_path,source='manifest',minimal=True)
    data['completion_costs']=[{'experiment_id':'candidate','pipeline_hash':pipeline.hash,
        'actual_seconds':100.,'rows':5}]
    graph=estimate(c,['candidate'],rows)
    assert graph['nodes'][0]['pipeline_hash']==pipeline.hash
    assert graph['seconds']>=250.


def test_real_finalization_reuses_legacy_control_then_refits_once(tmp_path,monkeypatch):
    from automl_agent.budget import Limits
    from automl_agent.util import read_json
    from conftest import write_task
    c=Controller(write_task(tmp_path),Limits(seconds=180,cpu=1),run_id='canonical_completion')
    try:
        c.prepare();c.freeze('Reuse equivalent legacy control artifact without new training')
        identifier=c.journal.get('baseline_id')
        rows=sorted(c.plan.fit+c.plan.validation)
        directory,raw=c._prepare_finalist(identifier,rows,'control',40)
        pipeline=PipelineSpec.model_validate(raw['frozen_pipeline'])
        raw['frozen_pipeline']=pipeline.model_dump(mode='json',exclude_defaults=True)
        atomic_json(directory/'worker_result.json',raw)
        assert estimate(c,[identifier],rows,'control')['seconds']==0
        execute=c.executor.run;calls=[]
        def only_refit(directory,request,seconds):
            assert 'control' not in request.get('predict_splits',[])
            calls.append(directory.parent.name)
            return execute(directory,request,seconds)
        monkeypatch.setattr(c.executor,'run',only_refit)
        c.finalize()
        assert calls==['refit']
        assert c.journal.get('control_status')==c.journal.get('refit_status')=='completed'
        provenance=read_json(c.workdir/'submission_provenance.json')
        assert provenance['scope']=='full_train_refit' and provenance['rows']==25
        assert c.budget.counter('llm_calls')==0
    finally:c.close()
