"""Deterministic tool replays for later integration; never reported as live Qwen."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .core import canonical, digest, write_json
from .context import focus, pack, prepare_request
from .memory import KnowledgeMemory, DEPENDENCIES
from .runtime import ReferenceStore


def replay(store, output):
    import numpy as np
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    rng=np.random.default_rng(42);results=[];start=time.perf_counter()
    X=rng.normal(size=(120,4));y=2*X[:,0]-X[:,1]
    observed_fits=[];provided={}
    def provide(identifier, query):
        entry=focus(store,identifier,query,max_content_chars=3500)
        packet=pack([entry])
        request,audit=prepare_request([{'role':'user','content':query}],packet,model='deterministic-replay')
        if not packet['answer_usable']:raise ValueError('Reference sequence requires a complete evidence packet: '+identifier)
        provided[identifier]={'packet':packet,'request':request,'audit':audit,'prepared_before_fit_count':len(observed_fits)}
    def fit_observed(model,features,target=None,*,row_ids,**kwargs):
        observed_fits.append({'class':type(model).__module__+'.'+type(model).__name__,'rows':list(row_ids),
                              'features_sha256':digest(np.asarray(features).tolist()) if np.isfinite(np.asarray(features,dtype=float)).all() else 'contains_missing_values',
                              'keyword_names':sorted(kwargs)})
        return model.fit(features,target,**kwargs)
    def record(name,checks,steps,material):
        checks['evidence_prepared']=material in provided and provided[material]['audit']['send_allowed']
        results.append({'scenario':name,'checks':checks,'passed':all(checks.values()),'decisions':steps,'provided_material':material,
                        'prepared_evidence':provided.get(material),'evidence_kind':'deterministic_real_tool_replay_not_live_model'})
    # Deliberately hold empty columns outside IterativeImputer. This pinned
    # version can otherwise skip the iterative process with keep_empty_features.
    provide('api_sklearn_impute_IterativeImputer','sklearn.impute.IterativeImputer sample_posterior')
    missing=X.copy();missing[rng.random(X.shape)<.18]=np.nan
    imputer=IterativeImputer(max_iter=2,n_nearest_features=3,random_state=42)
    transformed=imputer.fit_transform(missing[:90]);future=imputer.transform(missing[90:])
    transformed=np.column_stack([transformed,np.zeros(90)]);future=np.column_stack([future,np.zeros(30)])
    model=Ridge().fit(transformed,y[:90]);prediction=model.predict(future)
    record('missing_model_imputation',{'actual_auxiliary_fits':len(imputer.imputation_sequence_)>0,'actual_iterations':imputer.n_iter_>0,'fit_rows_only':imputer.initial_imputer_.statistics_.shape==(4,), 'empty_column_consistent':transformed.shape[1]==future.shape[1]==5,'finite_prediction':bool(np.isfinite(prediction).all())},4,'api_sklearn_impute_IterativeImputer')
    provide('api_sklearn_preprocessing_OneHotEncoder','sklearn.preprocessing.OneHotEncoder handle_unknown')
    encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(np.array([['a'],['b'],['a']]))
    result=encoder.transform(np.array([['new'],['a']]))
    categorical_model=LogisticRegression(random_state=42).fit(encoder.transform(np.array([['a'],['b'],['a']])),[0,1,0])
    category_predictions=categorical_model.predict(result);keys=['unseen-row','known-row'];output_records=list(zip(keys,category_predictions.tolist()))
    record('unseen_categories',{'unknown_predict_succeeds':category_predictions.shape==(2,),'test_not_in_dictionary':'new' not in encoder.categories_[0], 'original_order':bool([key for key,value in output_records]==keys and (result[0]==0).all() and result[1].sum()==1)},4,'api_sklearn_preprocessing_OneHotEncoder')
    provide('method:target_encoding','target encoding temporal validation contraindications')
    recommendation=store.query(mode='methods',reference_id='target_encoding',context={'validation':'temporal'})
    temporal_model=fit_observed(Ridge(),X[:90],y[:90],row_ids=range(90));temporal_pred=temporal_model.predict(X[90:])
    record('temporal_encoding',{'incompatible_recommendation_rejected':recommendation.get('entry',{}).get('applicability')=='inapplicable','specific_rejection_reason':bool(recommendation.get('entry',{}).get('applicability_reasons')),'causal_training_prefix':max(observed_fits[-1]['rows'])<90,'finite_future_prediction':bool(np.isfinite(temporal_pred).all())},4,'method:target_encoding')
    provide('api_torch_no_grad','torch.no_grad There is an exception')
    import torch
    torch.set_num_threads(1);torch.manual_seed(42)
    net=torch.nn.Linear(4,1);before=[parameter.detach().clone() for parameter in net.parameters()]
    optim=torch.optim.SGD(net.parameters(),lr=.03);features=torch.tensor(X[:90],dtype=torch.float32);targets=torch.tensor(y[:90,None],dtype=torch.float32)
    for _ in range(6):
        optim.zero_grad();loss=torch.nn.functional.mse_loss(net(features),targets);loss.backward();optim.step()
    net.eval()
    with torch.no_grad():pred=net(torch.tensor(X[90:],dtype=torch.float32))
    record('custom_cpu_torch',{'parameters_changed':any(not torch.equal(a,b) for a,b in zip(before,net.parameters())),'cpu_only':all(p.device.type=='cpu' for p in net.parameters()),'inference_no_grad':not pred.requires_grad,'finite_prediction':bool(torch.isfinite(pred).all())},4,'api_torch_no_grad')
    info=store.query(mode='api',query='lightautoml.automl.presets.tabular_presets.TabularAutoML',library='lightautoml',version='0.4.2',max_chars=6000)
    lama_id=info['entry']['id']
    provide(lama_id,'lightautoml.automl.presets.tabular_presets.TabularAutoML timeout')
    capabilities={'lightautoml':False,'ridge':True}
    selected='ridge' if not capabilities['lightautoml'] and capabilities['ridge'] else None
    if selected is None:raise ValueError('No verified adapter')
    alternative=fit_observed(Ridge(),X[:90],y[:90],row_ids=range(90));prediction=alternative.predict(X[90:])
    record('disabled_lightautoml_adapter',{'exact_reference_available':info.get('entry',{}).get('version')=='0.4.2','disabled_adapter_not_called':all(not event['class'].startswith('lightautoml') for event in observed_fits),'supported_alternative_executed':bool(hasattr(alternative,'coef_') and np.isfinite(prediction).all())},4,lama_id)
    document=store.document('api_sklearn_neural_network_MLPClassifier_fit');end=document['text'].find('\n')
    memory_path=output.parent/'replay_memory.sqlite';memory=KnowledgeMemory(memory_path)
    identifier=memory.fact(store,document['id'],0,end,contraindications=['Do not pass sample_weight to MLPClassifier.fit in scikit-learn 1.5.2.'])
    count=memory.progress_count;memory.provided(identifier,packet_sha256='before-other-tool');memory.close()
    memory=KnowledgeMemory(memory_path);context={key:'fixed' for key in DEPENDENCIES};packet=pack([],memory=memory.active(store,context))
    request,audit=prepare_request([{'role':'user','content':'Continue after context reduction and process restart'}],packet,model='deterministic-replay',compressor=lambda request:request)
    provided[document['id']]={'packet':packet,'request':request,'audit':audit,'prepared_before_fit_count':len(observed_fits)}
    retained=memory.active(store,context)
    signature=next((item['claim'] for item in retained if item['type']=='documentation_fact' and item['source']['version']=='1.5.2'),'')
    if not signature or 'sample_weight' in signature:raise ValueError('The required API constraint was not retained; no weighted step is authorized')
    alternate=fit_observed(LogisticRegression(random_state=42),X[:90],(y[:90]>0).astype(int),row_ids=range(90),sample_weight=np.ones(90))
    record('memory_after_restart',{'constraint_in_actual_request':'Do not pass sample_weight' in request['messages'][-1]['content'],'next_supported_alternative_executed':np.isfinite(alternate.predict_proba(X[90:])).all().item(),'repeated_delivery_not_progress':memory.progress_count==count,'transport_gate_passed':audit['send_allowed']},5,document['id'])
    memory.close()
    report={'corpus':store.version,'scenarios':results,'all_passed':all(r['passed'] for r in results),'seconds_after_corpus_open':time.perf_counter()-start,
            'observed_fits':observed_fits,
            'live_qwen_runs':0,'full_agent_integration':False,'status':'standalone_replay_only','note':'These are known working tool sequences. They do not establish autonomous model behavior or whole-agent acceptance.'}
    write_json(output,report);print(canonical({k:v for k,v in report.items() if k!='scenarios'}));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with ReferenceStore(a.corpus) as store:replay(store,a.output)
