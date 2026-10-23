"""Frozen, source-addressed evaluation, kept outside the deliverable runtime."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import time

from .context import evaluate_atoms, focus, pack, prepare_request, packet_audit
from .core import canonical, digest, file_hash, read_json, write_json
from .runtime import ReferenceArgs, ReferenceStore

# These topic/parameter specifications are fixed before the held-out run. Each
# atomic target is a complete source paragraph, not a keyword hit or a doc ID.
TOPICS = {
 'iterative': ('sklearn.impute.IterativeImputer','sample_posterior'),
 'onehot': ('sklearn.preprocessing.OneHotEncoder','handle_unknown'),
 'mlpfit': ('sklearn.neural_network.MLPClassifier.fit',''),
 'nograd': ('torch.no_grad','There is an exception'),
 'mse': ('torch.nn.MSELoss','reduction'),
 'lama': ('lightautoml.automl.presets.tabular_presets.TabularAutoML.fit_predict','roles'),
 'optimize': ('optuna.study.Study.optimize','timeout'),
 'catfit': ('catboost.CatBoostClassifier.fit','cat_features'),
 'xgbpredict': ('xgboost.XGBRegressor.predict','iteration_range'),
 'correlated': ('feature_engine.selection.DropCorrelatedFeatures','threshold'),
 'flamlfit': ('flaml.AutoML.fit','time_budget'),
 'merge': ('pandas.DataFrame.merge','validate'),
 'simple': ('sklearn.impute.SimpleImputer','keep_empty_features'),
 'ordinal': ('sklearn.preprocessing.OrdinalEncoder','handle_unknown'),
 'hist': ('sklearn.ensemble.HistGradientBoostingClassifier','missing values'),
 'inference': ('torch.inference_mode','autograd'),
 'sgdtorch': ('torch.optim.SGD','lr'),
 'ranker': ('lightgbm.LGBMRanker.fit','group'),
 'nanmean': ('numpy.nanmean','all-NaN'),
 'csr': ('scipy.sparse.csr_matrix','shape'),
 'ols': ('statsmodels.regression.linear_model.OLS','constant'),
 'agfit': ('autogluon.tabular.TabularPredictor.fit','time_limit'),
 'smote': ('imblearn.over_sampling.SMOTE','k_neighbors'),
 'catencode': ('category_encoders.TargetEncoder','handle_unknown'),
 'skencode': ('sklearn.preprocessing.TargetEncoder','cross fitting'),
 'logisticfit': ('sklearn.linear_model.LogisticRegression.fit','sample_weight'),
 'trainmode': ('torch.nn.Module.train','mode'),
 'timesplit': ('sklearn.model_selection.TimeSeriesSplit','test_size'),
 'flamlrefit': ('flaml.AutoML.retrain_from_log','time_budget'),
 'readcsv': ('pandas.read_csv','dtype'),
 'datetime': ('pandas.to_datetime','errors'),
 'suggest': ('optuna.trial.Trial.suggest_float','log'),
 'indicator': ('sklearn.impute.MissingIndicator','features'),
 'calibrate': ('sklearn.calibration.CalibratedClassifierCV','cv'),
 'predictproba': ('sklearn.linear_model.LogisticRegression.predict_proba','classes'),
 'group': ('sklearn.model_selection.GroupShuffleSplit','groups'),
 'refit': ('autogluon.tabular.TabularPredictor.refit_full','model'),
 'scaler': ('sklearn.preprocessing.StandardScaler','with_mean'),
 'tfidf': ('sklearn.feature_extraction.text.TfidfVectorizer','max_features'),
 'zero_grad': ('torch.optim.Optimizer.zero_grad','set_to_none'),
}
SINGLE_DEV = ['iterative','onehot','mlpfit','nograd','mse','lama','optimize','catfit','xgbpredict','correlated','flamlfit','merge']
SINGLE_HOLD = ['simple','ordinal','hist','inference','sgdtorch','ranker','nanmean','csr','ols','agfit','smote','catencode']
MULTI_DEV = [('iterative','simple'),('skencode','catencode'),('mlpfit','logisticfit'),('trainmode','nograd'),('timesplit','onehot'),('flamlfit','flamlrefit'),('readcsv','datetime'),('suggest','optimize')]
MULTI_HOLD = [('simple','indicator'),('ordinal','catencode'),('calibrate','predictproba'),('group','timesplit'),('agfit','refit'),('scaler','csr'),('tfidf','logisticfit'),('zero_grad','inference')]


def resolve(store,symbol):
    row=store.db.execute('SELECT doc FROM symbols WHERE symbol=? ORDER BY doc',(symbol,)).fetchone()
    if not row:raise ValueError('Missing annotated API: '+symbol)
    return store.document(row[0])


def passage(doc,needle):
    text=doc['text']
    if not needle:
        return 0,text.index('\n')
    definition=re.search(r'(?m)^([ \t]*)'+re.escape(needle)+r'(?:\s*\([^\n]*\))?\s*:',text)
    if definition:
        start=definition.start();tail=text[definition.end():];indentation=definition.group(1)
        boundaries=[match.start() for match in re.finditer(r'(?m)^'+re.escape(indentation)+r'[A-Za-z_]\w*(?:\s*\([^\n]*\))?\s*:',tail)]
        boundaries += [match.start() for match in re.finditer(r'(?m)^\S[^\n]*\n[-=]{3,}\s*\n',tail)]
        boundaries += [match.start() for match in re.finditer(r'(?m)^[A-Z][A-Za-z ]+::?\s*$',tail)]
        return start,definition.end()+min(boundaries) if boundaries else len(text)
    matches=list(re.finditer(re.escape(needle),text,re.I))
    if not matches:raise ValueError('No source passage for '+doc['id']+': '+needle)
    # Prefer the parameter definition over its occurrence in the signature.
    match=next((m for m in matches if m.start()>text.find('\n\n') and (m.start()==0 or text[m.start()-1]=='\n')),None)
    match=match or next((m for m in matches if m.start()>text.find('\n\n')),matches[0])
    start=text.rfind('\n\n',0,match.start())+2
    if start==1:start=0
    end=text.find('\n\n',match.end())
    if end<0:end=len(text)
    # Parameter blocks normally end at the next empty line. Very long examples
    # are not truncated into a fake gold answer: the preflight fails explicitly.
    return start,end


def gold_entry(store,doc,start,end):
    ranges=[(start,end)]+[(s['start'],s['end']) for s in doc.get('required_spans',[])]
    merged=[]
    for a,b in sorted(set(ranges)):
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    entry=store.entry({'doc':doc['id'],'start':start,'end':end,'text':doc['text'][start:end],'sha':digest(doc['text'][start:end])},record=doc)
    entry.pop('fragment',None);entry['fragments']=[{'start':a,'end':b,'sha256':digest(doc['text'][a:b])} for a,b in merged]
    entry['content']='\n\n[…]\n\n'.join(doc['text'][a:b] for a,b in merged);entry['required_reading']=[];entry['answer_usable']=doc['content_status']=='complete'
    return entry


def annotate(root,corpus):
    target=root/'evaluation/annotations.json'
    if target.exists():raise FileExistsError('Frozen annotations must not be silently revised')
    store=ReferenceStore(corpus);cases=[]
    def positive(identifier,split,topics,query=None,legacy=None):
        entries,atoms,versions=[],[],[]
        for index,key in enumerate(topics):
            symbol,needle=TOPICS[key]
            doc=resolve(store,symbol);start,end=passage(doc,needle)
            entry=gold_entry(store,doc,start,end);entries.append(entry);versions.append(doc['version'])
            atoms.append({'id':key,'critical':key in {'mlpfit','onehot','skencode','catencode','timesplit','nograd','inference','simple','ordinal','lama','group'},
                          'alternatives':[{'library':doc['library'],'version':doc['version'],'doc_sha256':doc['doc_sha256'],'reference_id':doc['id'],'start':start,'end':end,'required_text':[doc['text'][start:end]]}]})
        question=query or ' ; '.join(TOPICS[key][0]+' '+TOPICS[key][1] for key in topics)
        packet=pack(entries)
        request,audit=prepare_request([],packet,model='evaluation-fixture')
        case={'id':identifier,'split':split,'category':'single' if len(topics)==1 else 'combination','query':question,'atoms':atoms,'versions':sorted(set(versions)),
              'expected_ids':[entry['id'] for entry in entries],'gold_packet':packet,'gold_audit':audit,'legacy_expected':legacy}
        result=evaluate_atoms(request,case)
        if not result['sufficient'] or not packet['answer_usable'] or len(packet.get('materials',[]))!=len(entries):raise ValueError('Gold packet infeasible: '+identifier+' '+canonical(packet_audit(packet)))
        case['gold_result']=result;cases.append(case)
    for split,singles,pairs in [('development',SINGLE_DEV,MULTI_DEV),('heldout',SINGLE_HOLD,MULTI_HOLD)]:
        for i,key in enumerate(singles):positive(split+'-single-'+str(i+1),split,[key])
        for i,keys in enumerate(pairs):positive(split+'-combination-'+str(i+1),split,keys)
        for i,query in enumerate(['нотариальное наследование завещание','лицензия авиапилота аэродром','тектонические плиты вулканология','кулинария борщ выпечка','средневековая геральдика гербы']):
            cases.append({'id':split+'-negative-'+str(i+1),'split':split,'category':'negative','query':query,'expected_status':['not_found']})
        for i,spec in enumerate([{'mode':'search','query':'fit','library':'catboost','version':'999.0'}, {'mode':'api','query':'MLPClassifier','library':'scikit-learn','version':'1.7.0'}, {'mode':'api','query':'Tensor','library':'torch','version':'9.9.9'}, {'mode':'api','query':'TargetEncoder'}, {'mode':'api','query':'Pipeline'}]):
            cases.append({'id':split+'-version-'+str(i+1),'split':split,'category':'version_or_ambiguity','arguments':spec,'query':spec['query'],'expected_status':['missing_version'] if 'version' in spec else ['ambiguous_symbol']})
    # Preserve the historical 40 queries as development only. Gold packets for
    # these audit preservation of the API signature/first documented paragraph.
    for i,(query,identifier) in enumerate(read_json(root/'evaluation/legacy40.json')):
        doc=store.document(identifier)
        if not doc:raise ValueError('Legacy ID lost: '+identifier)
        end=doc['text'].find('\n\n');end=end if end>0 else min(500,len(doc['text']))
        entry=gold_entry(store,doc,0,end);packet=pack([entry]);request,audit=prepare_request([],packet,model='evaluation-fixture')
        case={'id':'legacy-'+str(i+1),'split':'legacy_development','category':'single','query':query,'legacy_expected':identifier,'expected_ids':[doc['id']],
              'atoms':[{'id':'documented_contract','critical':False,'alternatives':[{'library':doc['library'],'version':doc['version'],'doc_sha256':doc['doc_sha256'],'reference_id':doc['id'],'start':0,'end':end,'required_text':[doc['text'][:end]]}]}],
              'versions':[doc['version']],'gold_packet':packet,'gold_audit':audit,'annotation_scope':'legacy API contract preservation; not a new semantic validation claim'}
        case['gold_result']=evaluate_atoms(request,case)
        if not case['gold_result']['sufficient']:raise ValueError('Invalid legacy gold')
        cases.append(case)
    dataset={'corpus':store.version,'cases':cases,'authoring':'Source-addressed, machine-assisted annotation; no independent human semantic review is claimed',
             'heldout_policy':'Freeze before measurement. After first heldout report, repeats are regression runs, never new independent evidence.'}
    write_json(target,dataset)
    write_json(root/'evaluation/annotations.lock.json',{'annotations_sha256':file_hash(target),'corpus':store.version,
                'ranking_source_sha256':file_hash(root/'runtime.py'),'context_source_sha256':file_hash(root/'context.py'),
                'counts':{split:sum(c['split']==split for c in cases) for split in ['development','heldout','legacy_development']}})
    store.close();print(canonical({'cases':len(cases),'gold_preflight':True}))


def evaluate(root,corpus,split,output,implementation_hashes=None):
    dataset=read_json(root/'evaluation/annotations.json');lock=read_json(root/'evaluation/annotations.lock.json')
    if file_hash(root/'evaluation/annotations.json')!=lock['annotations_sha256']:raise ValueError('Annotation hash mismatch')
    store=ReferenceStore(corpus)
    if store.version!=dataset['corpus']:raise ValueError('Evaluation corpus changed')
    first_heldout=not (root/'evaluation/heldout_exposed.json').exists()
    implementation_hashes=implementation_hashes or {'runtime.py':file_hash(root/'runtime.py'),'context.py':file_hash(root/'context.py')}
    matches_freeze=(implementation_hashes['runtime.py']==lock['ranking_source_sha256'] and implementation_hashes['context.py']==lock['context_source_sha256'])
    if split=='heldout':
        if first_heldout and not matches_freeze:raise ValueError('Freeze implementation before revealing heldout')
        if first_heldout:write_json(root/'evaluation/heldout_exposed.json',{'annotations_sha256':lock['annotations_sha256'],'corpus':store.version})
    results=[]
    for case in dataset['cases']:
        if case['split']!=split:continue
        began=time.perf_counter()
        if 'expected_status' in case:
            response=store.query(case.get('arguments',{'query':case['query']}));passed=response['status'] in case['expected_status']
            results.append({'id':case['id'],'category':case['category'],'status':response['status'],'passed':passed,'false_confident':not passed})
            continue
        hits=store.search(ReferenceArgs(query=case['query']))[:5]
        actual=[hit['doc'] for hit in hits]
        entries=[focus(store,hit['doc'],case['query']) for hit in hits]
        packet=pack(entries)
        request,audit=prepare_request([],packet,model='evaluation-fixture')
        result=evaluate_atoms(request,case)
        results.append({'id':case['id'],'category':case['category'],'hit_at_5':any(identifier in actual for identifier in case['expected_ids']),
                        'annotated_source_recall_at_5':len(set(case['expected_ids'])&set(actual))/len(set(case['expected_ids'])),
                        'actual':actual,'sufficiency':result,'packet':packet,'audit':audit,'seconds':time.perf_counter()-began})
    positive=[r for r in results if 'sufficiency' in r];negative=[r for r in results if 'false_confident' in r]
    critical=[r for r in positive if any(a['critical'] for a in r['sufficiency']['atoms'])]
    sufficient=sum(r['sufficiency']['sufficient'] for r in positive)
    report={'split':split,'corpus':store.version,'annotations_sha256':lock['annotations_sha256'],'first_independent_heldout_run':split=='heldout' and first_heldout,
            'implementation_sha256':implementation_hashes,'matches_original_implementation_freeze':matches_freeze,
            'positive_cases':len(positive),'sufficient':sufficient,'sufficiency_rate':sufficient/len(positive) if positive else None,
            'Hit@5':sum(r['hit_at_5'] for r in positive)/len(positive) if positive else None,
            'Recall@5':None,'recall_note':'All relevant documents are not exhaustively annotated; annotated source coverage is reported separately.',
            'critical_passed':all(r['sufficiency']['critical_passed'] for r in critical),'false_confident':sum(r['false_confident'] for r in negative),
            'threshold_passed':bool(positive) and sufficient/len(positive)>=.95 and all(r['sufficiency']['critical_passed'] for r in critical) and all(r['passed'] for r in negative),
            'results':results}
    write_json(output,report);print(canonical({k:v for k,v in report.items() if k!='results'}));store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['annotate','evaluate']);p.add_argument('--root',type=Path,default=Path(__file__).parent);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--split',default='development');p.add_argument('--output',type=Path)
    p.add_argument('--implementation-hashes',type=Path)
    a=p.parse_args();annotate(a.root,a.corpus) if a.action=='annotate' else evaluate(a.root,a.corpus,a.split,a.output,read_json(a.implementation_hashes) if a.implementation_hashes else None)
