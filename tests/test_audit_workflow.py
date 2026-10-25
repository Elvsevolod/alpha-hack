import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis, PipelineSpec, ToolCommand, DatasetReportArgs, RelatedTable
from automl_agent.controller import Controller
from automl_agent.custom import TransformerPipeline
from automl_agent.cost_model import performance_curve, phase_forecast
from automl_agent.learned_blend import fit_blend, combine
from automl_agent.related_tables import attach
from automl_agent.shift import importance_weights
from automl_agent.tool_results import payload
from automl_agent.util import file_hash
from conftest import write_task
from test_pipelines import context
from tools.quality_promotion_gate import evaluate


def hypothesis(component):
    return Hypothesis(observation='Offline test: observed numeric features',expected_effect='Measure a bounded alternative',
        changed_component=component,acceptance='Untouched research metric and a complete frozen refit')


def test_batch_domain_and_learned_blend_controller(tmp_path,monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    c=Controller(write_task(tmp_path),Limits(seconds=240,cpu=1),run_id='audit_batch')
    try:
        c.prepare()
        before=c.publisher.output.read_bytes()
        domain=c.run_tools.transfer(DatasetReportArgs(report='domain',budget_seconds=15))
        assert domain['scope']=='unlabelled_train_vs_test_features'
        for directory in (c.workdir/'feature_diagnostics').iterdir():
            assert not (directory/'fit_y.json').exists()
        candidates=[{'pipeline':PipelineSpec(model='ridge',hyperparameters={'alpha':alpha}).model_dump(),
            'hypothesis':hypothesis('alpha_'+str(alpha)).model_dump(),'budget_seconds':35} for alpha in (1,5)]
        command=ToolCommand(operation_id='batch_test',tool='run_experiments',arguments={'candidates':candidates,'budget_seconds':80})
        result=payload(c.dispatch(command))
        assert len(result['results'])==2 and all(row['status']=='completed' for row in result['results']),result
        count=c.budget.counter('candidate_count')
        assert payload(c.dispatch(command))==result
        assert c.budget.counter('candidate_count')==count
        ids=[row['experiment_id'] for row in result['results']]
        blend=payload(c.dispatch(ToolCommand(operation_id='oof_blend_test',tool='build_ensemble',arguments={
            'members':ids,'strategy':'weighted_oof','hypothesis':hypothesis('weights').model_dump(),'budget_seconds':50})))
        assert blend['result']['status']=='completed',blend
        _, saved = c.lookup(blend['experiment']['experiment_id'])
        frozen=json.loads((__import__('pathlib').Path(saved.artifacts['directory'])/'worker_result.json').read_text())['frozen_pipeline']
        assert np.isclose(sum(frozen['ensemble_weights']),1)
        assert not c.journal.get('control_opened')
        c.freeze('Internal workflow verification');c.finalize()
        assert c.journal.get('control_status')==c.journal.get('refit_status')=='completed'
        assert c.journal.get('publication')['scope']=='full_train_refit'
    finally:c.close()


class MemorizingTransformer:
    def fit(self,X,y):
        self.mapping=dict(zip(X['category'],y.iloc[:,0]));return self
    def fit_transform(self,X,y):
        self.fit(X,y);return self.transform(X)
    def transform(self,X):
        return pd.DataFrame({'encoded':X['category'].map(self.mapping).fillna(0.)},index=X.index)


def test_supervised_custom_transformer_cannot_learn_own_target(tabular):
    X,y,task=tabular;X=X.copy();X['category']=['unique_'+str(i) for i in range(len(X))]
    spec=PipelineSpec(engine='custom',model='ridge',custom_component='transformer')
    model=TransformerPipeline(MemorizingTransformer(),spec,task)
    summary=model.fit(X,y,context(task))
    assert summary.preprocessing['custom_transformer']['covered_rows']==len(X)
    assert model.pipeline.models['answer'].coef_[0]==0
    assert len(model.transformer.mapping)==len(X)
    for fold in model.crossfit['folds']:
        assert set(fold['fit_keys']).isdisjoint(fold['heldout_keys'])


def test_native_probabilities_survive_class_offsets(tabular):
    from automl_agent.contracts import MetricSpec,OutputSpec
    from automl_agent.pipelines import DirectPipeline
    X,y,task=tabular;y=y.copy();y['answer']=np.where(y.answer>1,'a',np.where(y.answer<0,'b','c'))
    task=task.model_copy(update={'kind':'classification','classes':{'answer':['c','a','b']},
        'metric':MetricSpec(name='f1'), 'output':OutputSpec(columns=['answer'],prediction_type='labels')})
    spec=PipelineSpec(model='extra_trees',hyperparameters={'n_estimators':12,'min_samples_leaf':5},prediction_policy='class_offsets',postprocess_fit_policy='inner_holdout')
    model=DirectPipeline(spec,task);model.fit(X,y,context(task));bundle=model.predict(X.iloc[:10],context(task))
    assert bundle.prediction_type=='labels'
    raw=np.asarray(bundle.raw_probabilities['answer'])
    assert raw.shape==(10,3) and np.allclose(raw.sum(axis=1),1)
    assert np.any((raw>0)&(raw<1)) and len(bundle.decision_values['answer'])==10


def test_related_events_obey_strict_cutoff_and_row_order(tabular,tmp_path):
    X,y,task=tabular
    events=pd.DataFrame({'entity':['a','a','a'],'at':[1,2,3],'amount':[10.,20.,100000.]})
    path=tmp_path/'events.csv';events.to_csv(path,index=False)
    base=pd.DataFrame({'entity':['a','a','cold'],'when':[2,3,10]},index=['second','first','third'])
    relation=RelatedTable(file='events.csv',keys=['entity'],base_keys=['entity'],prefix='events',aggregations={'amount':['sum','count']},time_column='at',cutoff_column='when')
    task=task.model_copy(update={'related_tables':[relation]})
    dataset=SimpleNamespace(train=base.copy(),test=base.copy(),task_path=tmp_path/'task.md',catalog={'events.csv':{'sha256':file_hash(path)}},paths={})
    attach(dataset,task)
    assert dataset.train.index.tolist()==base.index.tolist()
    assert dataset.train.events__amount__sum.iloc[:2].tolist()==[10,30]
    assert dataset.train.events__amount__count.iloc[:2].tolist()==[1,2]
    assert np.isnan(dataset.train.events__amount__sum.iloc[2])
    attach(dataset,task)
    assert len(dataset.train.columns)==4


def test_domain_weights_are_clipped_and_target_free(tabular):
    X,y,task=tabular;future=X.copy();future['number']=5
    weights,report=importance_weights(X,future,task,1,42)
    assert len(weights)==len(X) and np.isclose(weights.mean(),1)
    assert report['task_labels_used'] is False and report['effective_rows']<=len(X)+1e-6
    assert weights.max()/weights.min()<=25+1e-6


def test_fixed_phase_cost_curve_does_not_extrapolate_startup():
    samples=[{'training_summary':{'rows':rows,'seconds':3+rows*.001,'budget':{'selected_fit_actual_seconds':3+rows*.001}}} for rows in (400,1600)]
    curve=performance_curve(samples)
    assert curve['fixed_fit_seconds']==pytest.approx(3)
    assert curve['seconds_per_row']==pytest.approx(.001)
    summary={'rows':1600,'cost_curve':curve,'seconds':4.6,'execution_phases':{'fit_seconds':4.6,'load_and_construct_seconds':10,'save_load_seconds':1,'predict_and_verify_seconds':2}}
    assert phase_forecast(summary,17.6,10000,1600)<40


def test_quality_gate_rejects_offline_or_incomplete_evidence():
    manifest={'protocol_hash':'p','versions':['v074','candidate'],'tasks':['task'],'seeds':[42],
        'resources':{'seconds':10800},'metric_directions':{'task':'minimize'},'evaluation_independence':'reused_public'}
    result=evaluate(manifest,[])
    assert result['status']=='quality_advantage_unestablished' and not result['automatic_promotion']


def test_quality_gate_requires_real_agent_trajectories_and_finite_metrics():
    manifest={'protocol_hash':'frozen','versions':['v074','v075'],'tasks':['task'],'seeds':[42,137,2026],
        'resources':{'seconds':10800},'metric_directions':{'task':'minimize'},'evaluation_independence':'fresh_frozen_audit'}
    rows=[{'protocol_hash':'frozen','version':version,'task':'task','seed':seed,'resources':manifest['resources'],
        'valid_csv':True,'control_status':'completed','refit_status':'completed','publication_scope':'full_train_refit',
        'raw_metric':2. if version=='v074' else 1.,'real_qwen':False}
        for version in manifest['versions'] for seed in manifest['seeds']]
    assert evaluate(manifest,rows)['status']=='quality_advantage_unestablished'
    rows=[{**row,'real_qwen':True} for row in rows]
    result=evaluate(manifest,rows)
    assert result['status']=='eligible_for_quality_review' and not result['automatic_promotion']
    for invalid in (float('nan'),float('inf'),True):
        assert evaluate(manifest,[{**rows[0],'raw_metric':invalid},*rows[1:]])['status']=='quality_advantage_unestablished'


def test_autonomous_clustering_contract_resolution_without_paid_api(tmp_path,monkeypatch):
    from automl_agent.clustering_route import resolve_and_run
    from automl_agent.qwen import QwenClient
    rng=np.random.default_rng(42)
    points=np.r_[rng.normal(-3,.1,(50,2)),rng.normal(3,.1,(50,2))]
    train=pd.DataFrame(points,columns=['x','z']);train['key']=['a'+str(i) for i in range(100)]
    future=train.iloc[:12].copy();future['key']=['future_'+str(i) for i in range(12)]
    train.to_csv(tmp_path/'observations.csv',index=False);future.to_csv(tmp_path/'new_observations.csv',index=False)
    path=tmp_path/'task.md';path.write_text('Target-free clustering. Use silhouette, choose 2 or 3 clusters. Identifier: key. Numeric features x,z. Output submission.csv with key,cluster.')
    calls=[]
    def resolve(self,system,request):
        calls.append(request)
        return {'contract':{'kind':'clustering','train_file':'observations.csv','test_file':'new_observations.csv',
            'features':['x','z'],'id_columns':['key'],'n_clusters':2,'n_clusters_candidates':[2,3]},'evidence_quote':'Target-free clustering.'}
    monkeypatch.setattr(QwenClient,'resolve',resolve)
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    import time
    report=resolve_and_run(path,Limits(seconds=60,cpu=1),time.time(),'autonomous_cluster')
    assert calls and report['state']=='completed' and report['resolution']=='qwen_semantic_contract',report
    output=pd.read_csv(tmp_path/'submission.csv')
    assert output['key'].tolist()==future['key'].tolist() and list(output)==['key','cluster']
    assert report['publication']['scope']=='full_train_refit'


def test_related_table_contract_reaches_complete_controller_refit(tmp_path,monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    path=write_task(tmp_path)
    for filename in ('train.csv','test.csv'):
        frame=pd.read_csv(tmp_path/filename,dtype=str);frame['entity']=['e'+str(i%8) for i in range(len(frame))];frame.to_csv(tmp_path/filename,index=False)
    pd.DataFrame({'entity':['e'+str(i) for i in range(8)],'amount':np.arange(8)}).to_csv(tmp_path/'related.csv',index=False)
    spec_path=tmp_path/'task_spec.json';spec=json.loads(spec_path.read_text())
    spec['feature_columns']+=['entity','rel__amount__mean'];spec['categorical_columns']+=['entity']
    spec['related_tables']=[{'file':'related.csv','keys':['entity'],'base_keys':['entity'],'prefix':'rel','aggregations':{'amount':['mean']}}]
    spec_path.write_text(json.dumps(spec))
    c=Controller(path,Limits(seconds=90,cpu=1),run_id='related_contract')
    try:
        report=c.run(no_llm=True)
        assert report['failure'] is None,report['failure']
        assert report['state']=='completed' and report['refit_status']=='completed',report
        assert 'rel__amount__mean' in c.dataset.train
        assert c.journal.get('publication')['scope']=='full_train_refit'
    finally:c.close()


def test_quality_cost_prefers_affordable_leader_and_retains_fallback():
    from automl_agent.finalists import affordable_incumbent
    choices=[{'id':'baseline','score':.6,'seconds':90},{'id':'leader','score':.7,'seconds':140}]
    assert affordable_incumbent(choices,150,True)=='leader'
    assert affordable_incumbent(choices,100,True)=='baseline'
    assert affordable_incumbent(choices,80,True) is None


def test_regularized_stacked_oof_preserves_prediction_shape(tabular):
    X,y,task=tabular;rng=np.random.default_rng(42)
    reports=[{'row_keys':X.index.tolist(),'classes':{},'folds_hash':'same_external_folds','folds':[{'fold':0},{'fold':1}],
        'values':{'answer':(y.answer.to_numpy()+rng.normal(0,noise,len(X))).tolist()}} for noise in (.1,10)]
    weights,meta,evidence=fit_blend(task,reports,y,'stacked_oof',42)
    spec=PipelineSpec(engine='ensemble',members=['a','b'],ensemble_strategy='stacked_oof',ensemble_meta=meta)
    result=combine(task,[r['values'] for r in reports],spec)
    assert np.asarray(result['answer']).shape==(len(X),)
    assert np.mean((result['answer']-y.answer.to_numpy())**2)<1
    assert evidence['unbiased_meta_score_claimed'] is False


def test_guarded_pseudo_label_positive_and_negative_simulations(tabular):
    from automl_agent.contracts import MetricSpec,OutputSpec
    from automl_agent.shift import ShiftPipeline
    X,y,task=tabular;y=y.copy();y['answer']=X.category
    task=task.model_copy(update={'kind':'classification','classes':{'answer':['B','A']},'metric':MetricSpec(name='f1'),
        'output':OutputSpec(columns=['identifier','answer'],prediction_type='labels')})
    future=X.iloc[:30].copy();future.index=['future_'+str(i) for i in range(len(future))]
    c={**context(task),'unlabelled_features':future}
    spec=PipelineSpec(model='extra_trees',hyperparameters={'n_estimators':32},shift_strategy='pseudo_label')
    model=ShiftPipeline(spec,task)
    # Perfect teacher precision alone is not evidence that adding pseudo rows
    # improves a student on the separate gate population.
    with pytest.raises(ValueError,match='student did not improve'):
        model.fit(X,y,c)
    evidence=model.evidence
    assert evidence['admitted'] and evidence['precision']>=.95
    assert evidence['student_gate']['gain']<=1e-12
    rng=np.random.default_rng(111);random_y=y.copy();random_y['answer']=rng.choice(['A','B'],len(X))
    shallow=spec.model_copy(update={'hyperparameters':{'n_estimators':32,'max_depth':1}})
    with pytest.raises(ValueError,match='simulation failed'):
        ShiftPipeline(shallow,task).fit(X,random_y,c)


def test_clustering_mention_does_not_override_supervised_task(tmp_path,monkeypatch):
    import time
    from automl_agent.clustering_route import resolve_and_run
    from automl_agent.qwen import QwenClient
    path=write_task(tmp_path);path.write_text('Supervised regression with RMSE. Clustering is an optional feature idea.')
    seen=[]
    def resolve(self,system,request):
        seen.append(request);return {'not_clustering':True}
    monkeypatch.setattr(QwenClient,'resolve',resolve)
    assert resolve_and_run(path,Limits(seconds=60,cpu=1),time.time()) is None
    assert resolve_and_run(path,Limits(seconds=60,cpu=1),time.time()) is None
    assert len(seen)==1


def test_terminal_model_failure_finalizes_without_recovery_generations(tmp_path,monkeypatch):
    from automl_agent.qwen import QwenUnavailable
    c=Controller(write_task(tmp_path),Limits(seconds=180,cpu=1),run_id='provider_unavailable')
    try:
        monkeypatch.setattr(type(c.qwen),'configured',property(lambda self:True))
        calls=[]
        def unavailable(*args):
            calls.append(True)
            raise QwenUnavailable('Explicit synthetic HTTP 402')
        monkeypatch.setattr(c.qwen,'next_command',unavailable)
        monkeypatch.setattr(c,'recover_search',lambda error:pytest.fail('Terminal failure must not request another critic'))
        report=c.run()
        assert len(calls)==1 and report['failure'] is None,report
        assert report['state']==report['control_status']==report['refit_status']=='completed'
        assert report['publication']['scope']=='full_train_refit'
    finally:c.close()
