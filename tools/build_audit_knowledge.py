"""Audited v074 corpus delta: explicit replacements/additions only."""
import json
import sqlite3
from pathlib import Path
from ml_knowledge.core import canonical, digest, file_hash, read_json, write_json
from automl_agent.contracts import PipelineSpec

ROOT=Path(__file__).resolve().parents[1]


def update_catalogs():
    catalog=ROOT/'ml_knowledge/catalogs'
    path=catalog/'methods/hard_case_ml11.json'
    obj=read_json(path)
    intervention=('get_dataset_report report=domain is permitted: a feature-only train-vs-test model uses domain labels, never task-y. '
        'Use report=transfer first. Compare an opt-in direct pipeline shift_strategy=importance_weighting or pseudo_label only on a legal transfer simulation; '
        'domain AUC measures P(X) differences and does not establish P(y|X) or better target quality.')
    obj.update(version=2,hypothesis=intervention,minimum_experiment=intervention,
        constraints='No test task labels. Clip domain weights and inspect effective rows/support. Pseudo-labels require held-out fit precision, coverage and per-class false-positive checks. Preserve external validation/control boundaries.',
        contraindications=['Unverified covariate-shift assumption; low support or unstable pseudo-label precision.'])
    write_json(path,obj)
    recipes=[
        ('xgboost_class_offsets','Multiclass Macro F1: XGBoost and frozen inner decision offsets',
         PipelineSpec(model='xgboost',hyperparameters={'n_estimators':400,'max_depth':6,'learning_rate':.05},
                      prediction_policy='class_offsets',postprocess_fit_policy='inner_holdout',early_stopping_rounds=30),
         ['classification'],'Three or more classes, label output, Macro F1/accuracy; compare with the same model default decision. Historical gain is conditional and is not a task-specific winner.'),
        ('lightgbm_schedule','CPU LightGBM with honest inner early stopping',
         PipelineSpec(model='lightgbm',hyperparameters={'n_estimators':1000,'num_leaves':31,'learning_rate':.05},early_stopping_rounds=30),
         ['regression','classification'],'Inner split preserves group/time; selected iterations freeze before outer validation and complete refit.'),
        ('autogluon_bagging','Bounded AutoGluon bagging and weighted ensemble',
         PipelineSpec(engine='autogluon',model='autogluon',learners=['GBM','CAT'],autogluon_profile='bagging',bag_folds=3),
         ['regression','classification'],'Random or explicit group validation only; temporal bagging is rejected. CPU/GPU/memory/time are controller-owned. Use run_automl.'),
        ('lightautoml_cpu','LightAutoML 0.4.2 CPU single-family frozen refit',
         PipelineSpec(engine='lightautoml',model='lightautoml',learners=['lgb']),
         ['regression','classification'],'Use run_automl. One untuned GBM family with inner iteration selection and a fixed-iteration full-data refit; no nested HPO or external downloads.')]
    for identifier,title,pipeline,kinds,conditions in recipes:
        write_json(catalog/'recipes'/f'{identifier}.json',{'id':identifier,'version':1,'title':title,
            'pipeline':pipeline.model_dump(mode='json'),'task_kinds':kinds,'prediction_types':['values','probabilities','labels'],
            'method_refs':['boosting'],'conditions':[conditions],'constraints':conditions,
            'capability_status':'verified_only_after_actual_fit_predict_save_load_refit','alternatives':['linear','forest','boosting'],
            'refit_policy':{'rows':'permitted_training_rows','configuration':'frozen_selected_parameters','new_search':False}})
    return [('method','hard_case_ml11')]+[('recipe',row[0]) for row in recipes]


def build_delta():
    changes=update_catalogs();frozen=ROOT/'ml_knowledge/frozen'
    baseline=ROOT/'outputs/audit_plan_v074_20261024/knowledge_v074_source'
    import shutil,zipfile,ast
    with zipfile.ZipFile(ROOT/'dist/solution_v074.zip') as archive:
        tree=ast.parse(archive.read('ml_knowledge_bundle.py'))
        manifest=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MANIFEST' for t in node.targets))
    if not baseline.exists():
        shutil.copytree(ROOT/'ml_knowledge/corpus',baseline)
    for name,spec in manifest['files'].items():
        if file_hash(baseline/name)!=spec['sha256']:
            raise ValueError('v074 corpus source hash mismatch: '+name)
    records={}
    with sqlite3.connect(baseline/'knowledge.sqlite') as db:
        for identifier,meta,text in db.execute('SELECT d.id,d.meta,t.text FROM docs d JOIN texts t ON d.sha=t.sha'):
            obj=json.loads(meta);obj['text']=text;records[identifier]=obj
    before={key:canonical(value) for key,value in records.items()}
    for kind,identifier in changes:
        obj=read_json(ROOT/'ml_knowledge/catalogs'/('methods' if kind=='method' else 'recipes')/(identifier+'.json'))
        key=kind+':'+identifier
        if key in records:
            record=dict(records[key]);record.update(version=str(obj['version']),payload=obj,text=canonical(obj),warnings=[obj.get('constraints','')])
        else:
            record={'id':key,'legacy_id':identifier,'type':kind,'title':obj['title'],'library':'project','version':str(obj['version']),
                'symbols':[],'text':canonical(obj),'payload':obj,'source_url':'project:recipes/'+identifier,
                'source_kind':'project_recommendation','license':'project','conditions':obj['conditions'],
                'warnings':[obj.get('constraints','')],'applicability':'unknown','content_status':'complete',
                'library_versions':[{'library':'project','version':str(obj['version'])}]}
        if kind=='method':
            record['text']='\n\n'.join([obj['title'], 'Observation: '+obj['observations'],
                'Diagnostic: '+obj['diagnostic'], 'Minimum experiment: '+obj['minimum_experiment'],
                'Acceptance: '+obj['acceptance'], 'Contraindications: '+obj['constraints'],
                'Execution: '+obj['implementation_path']+'; verify the concrete adapter.',
                'Cost: '+obj['cost'], 'Evidence status: project hypothesis, not measured improvement.'])
        record['doc_sha256']=digest(record['text']);records[key]=record
    changed=[key for key in before if before[key]!=canonical(records[key])]
    added=sorted(set(records)-set(before))
    if changed!=['method:hard_case_ml11'] or len(added)!=4:
        raise ValueError('Unexpected corpus delta: '+str((changed,added)))
    (frozen/'documents.jsonl').write_text(''.join(canonical(records[key])+'\n' for key in sorted(records)))
    lock=read_json(frozen/'inputs.lock.json');lock.update(documents_sha256=file_hash(frozen/'documents.jsonl'),
        base_release='solution_v074.zip',audit_delta={'changed':changed,'added':added},
        runtime_sources={name:file_hash(ROOT/'ml_knowledge'/name) for name in ['runtime.py','lexical.py','context.py']})
    write_json(frozen/'inputs.lock.json',lock)
    write_json(ROOT/'outputs/audit_plan_v074_20261024/knowledge_delta.json',{'preserved':len(before)-len(changed),'changed':changed,'added':added})
    from ml_knowledge.build import build
    destination=ROOT/'ml_knowledge/corpus_audit_build'
    if destination.exists():
        raise FileExistsError('Prior audit build must be inspected before rebuilding')
    result=build(frozen,destination)
    old=ROOT/'outputs/audit_plan_v074_20261024'/('knowledge_previous_'+read_json(ROOT/'ml_knowledge/corpus/knowledge.lock.json')['corpus'][:12])
    if not old.exists():
        (ROOT/'ml_knowledge/corpus').rename(old)
    else:
        raise FileExistsError('Previous corpus swap already exists')
    destination.rename(ROOT/'ml_knowledge/corpus')
    return result

if __name__=='__main__':
    print(canonical(build_delta()))
