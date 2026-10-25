"""Post-evaluation integrity diagnostics; no fitting, selection or API calls."""
from pathlib import Path
import json,sys
import numpy as np
from sklearn.metrics import f1_score,precision_recall_fscore_support

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.pytorch_agent_study import load_dataset
from automl_agent.util import atomic_json,file_hash


def run():
    source=ROOT/'outputs/realmlp_author_equal_20261026';records=[]
    for task in ['regression','classification','adult','bank','bike','har']:
        dataset=load_dataset(task);target=dataset.task.targets[0]
        classes=sorted(dataset.task.classes.get(target,[]))
        for path in sorted(source.glob('*/'+task+'/*/result.json')):
            directory=path.parent;result=json.loads(path.read_text());p=np.load(directory/'prediction.npy')
            with np.load(directory/'charged_splits.npz') as split:
                fit,evaluate=split['train'],split['evaluation']
            assert not set(fit)&set(evaluate)
            y=dataset.train.iloc[evaluate][target].to_numpy()
            record=dict(task=task,phase=result['phase'],arm=result['arm'],prediction_sha256=file_hash(directory/'prediction.npy'),
                fit_rows=len(fit),evaluation_rows=len(evaluate),finite=bool(np.isfinite(p).all()))
            if dataset.task.group_column:
                groups=dataset.train[dataset.task.group_column].astype(str)
                fit_groups,evaluation_groups=set(groups.iloc[fit]),set(groups.iloc[evaluate])
                assert not fit_groups&evaluation_groups
                record['group_boundary']=dict(fit_units=len(fit_groups),evaluation_units=len(evaluation_groups),overlap=0)
            if dataset.task.time_column:
                from automl_agent.validation import ordered_periods
                periods=ordered_periods(dataset.train[dataset.task.time_column])
                assert periods.iloc[fit].max()<periods.iloc[evaluate].min()
                record['temporal_boundary']=dict(strict_past_to_future=True,
                    fit_periods=int(periods.iloc[fit].nunique()),evaluation_periods=int(periods.iloc[evaluate].nunique()))
            if classes:
                assert p.shape==(len(y),len(classes))
                assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
                np.testing.assert_allclose(p.sum(1),1.,atol=1e-5)
                label=np.asarray(classes)[(np.log(np.maximum(p,1e-15))+np.asarray(result['offsets'])).argmax(1)]
                score=f1_score(y,label,labels=classes,average='macro',zero_division=0)
                precision,recall,f1,support=precision_recall_fscore_support(y,label,labels=classes,zero_division=0)
                record['classes']=[dict(label=str(c),support=int(s),precision=float(pr),recall=float(re),f1=float(f))
                    for c,s,pr,re,f in zip(classes,support,precision,recall,f1)]
                record['accuracy']=float(np.mean(y==label))
            else:
                assert p.shape==y.shape
                score=float(np.sqrt(np.mean((y-p)**2)))
                record['residual_mean']=float(np.mean(y-p))
            np.testing.assert_allclose(score,result['score'],rtol=0,atol=1e-10)
            record.update(recomputed_score=float(score),matches_recorded_metric=True)
            records.append(record)
    atomic_json(source/'prediction_audit.json',dict(records=records,api_calls=0,no_fitting=True,
        no_control_driven_recipe_changes=True,independent_new_quality_test=False))
    print(json.dumps(dict(verified=len(records)),indent=2))


if __name__=='__main__':run()
