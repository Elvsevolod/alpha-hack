"""Reproduce already-viewed historical control metrics after recipe freezing.

This only audits immutable saved reference models. It never tunes a candidate.
"""
from pathlib import Path
import importlib.util,json,sys
import numpy as np,pandas as pd,joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.pytorch_agent_study import references,load_dataset,correction,metric
from automl_agent.util import atomic_json,file_hash


def run(output,freeze):
    if not Path(freeze).is_file():raise ValueError('An explicit completed selection freeze is required before control verification')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);refs=references()
    for name in ['group_context','residual_calibration']:
        path=ROOT/'research/pytorch_agent_20261026/reference_sources'/(name+'.py')
        spec=importlib.util.spec_from_file_location('automl_agent.'+name,path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    root=Path('/Users/semolik/Developer/alpha-hack/outputs');records=[]
    for kind,folder,names in [('regression','literature_reference_20261025/regression',['regularized_lgb','engineered_cb']),('classification','further_reference_20261025/classification',['current_weighted','hierarchy_balanced'])]:
        d=load_dataset(kind);source=root/folder;split=np.load(source/'splits.npz');rows=split['control'];fit=split['development'];task=d.task;target=task.targets[0];classes=sorted(task.classes[target]) if task.classes else None
        y=pd.Categorical(d.train[target],categories=classes).codes if classes else d.train[target].to_numpy();counts=np.bincount(y[fit],minlength=len(classes)) if classes else None
        predictions=[];hashes={}
        for name in names:
            path=source/(name+'_audit/model.joblib');hashes[str(path)]=file_hash(path);state=joblib.load(path)
            data=refs.engineer(d.train.iloc[rows],state['feature_plan'])[state['columns']].copy()
            for c in state['categorical']:
                data[c]=data[c].fillna('__MISSING__').astype(str)
                if state['spec']['model']=='lightgbm':data[c]=pd.Categorical(data[c],categories=state['vocabulary'][c])
            if state['preprocessor'] is not None:data=state['preprocessor'].transform(data)
            model=state['model']
            if hasattr(model,'models_'):
                for component in model.models_.values():component.set_params(n_jobs=1)
            elif state['spec']['model']!='catboost':model.set_params(n_jobs=1)
            if state['spec']['model']=='catboost':p=model.predict(data,thread_count=1)
            else:p=model.predict_proba(data) if classes else model.predict(data)
            predictions.append(correction(p,counts) if classes else p)
        base=sum(predictions)/len(predictions);np.save(out/(kind+'_uncalibrated.npy'),base)
        if classes:
            selection=json.loads((source/'selection.json').read_text());offset=np.mean(selection['learned_offsets'],axis=0)
            score=metric(y[rows],base,classes,offset);expected=.6565257846316546;recipe=selection
        else:
            path=root/'strong_forward_meta_20261025/frozen_meta.joblib';meta=joblib.load(path);hashes[str(path)]=file_hash(path)
            calibrator=meta['model'];base=calibrator.predict(d.train.iloc[rows],base)
            score=metric(y[rows],base,None);expected=38.83714138257429
            recipe=dict(policy='residual_half',categorical=calibrator.categorical,shrinkage=calibrator.shrinkage,strength=calibrator.strength)
        np.save(out/(kind+'_prediction.npy'),base)
        records.append(dict(dataset=kind,metric=score,expected_historical=expected,reproduced=bool(np.isclose(score,expected,rtol=0,atol=1e-10)),
            source_hashes=hashes,recipe=recipe,rows=len(rows),fit_rows=len(fit),data_hash=d.source_hash,
            control_previously_viewed=True,new_quality_claim=False,prediction_sha256=file_hash(out/(kind+'_prediction.npy'))))
    atomic_json(out/'verification.json',dict(records=records,freeze_sha256=file_hash(Path(freeze)),api_calls=0,control_used_only_for_historical_verification=True))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
