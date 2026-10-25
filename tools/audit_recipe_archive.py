"""Read the four new recipes from the exact ZIP, then fit/save/refit offline."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile


def main(archive, output):
    reports = []
    with tempfile.TemporaryDirectory(prefix='recipe-archive-') as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(archive) as package:
            package.extractall(root)
        sys.path = [str(root), *[p for p in sys.path if p and not p.startswith('/workspace')]]
        from agent_bundle import materialize
        runtime = materialize(root)
        sys.path.insert(0, str(runtime))
        import cloudpickle
        import numpy as np
        import pandas as pd
        import automl_agent.pipelines as pipelines
        from automl_agent.contracts import TaskSpec, MetricSpec, OutputSpec, PipelineSpec
        from ml_knowledge_bundle import open_store
        assert Path(pipelines.__file__).is_relative_to(runtime)
        rng = np.random.default_rng(42)
        X = pd.DataFrame({'value': rng.normal(size=300), 'category': np.tile(['red','green','blue'],100)},
            index=['row_'+str(i) for i in range(300)])
        with open_store(root/'references') as store:
            for identifier in ('xgboost_class_offsets','lightgbm_schedule','autogluon_bagging','lightautoml_cpu'):
                record = store.document('recipe:'+identifier)
                spec = PipelineSpec.model_validate(record['payload']['pipeline'])
                classification = identifier == 'xgboost_class_offsets'
                y = pd.DataFrame({'answer': np.where(X.value<-.5,'z',np.where(X.value>.5,'m','a'))
                    if classification else X.value*3+rng.normal(size=len(X))*.1}, index=X.index)
                task = TaskSpec(kind='classification' if classification else 'regression',targets=['answer'],
                    feature_columns=list(X),categorical_columns=['category'],classes={'answer':['z','a','m']} if classification else {},
                    metric=MetricSpec(name='f1' if classification else 'rmse'),
                    output=OutputSpec(columns=['answer'],prediction_type='labels' if classification else 'values'),source_hash='fixture')
                work = root/identifier; work.mkdir()
                context = {'cpu':2,'memory_mb':4000,'strategy':'random','split_id':'test','protocol_hash':'fixture',
                    'task':task.model_dump(),'workdir':str(work),'budget_seconds':100,'deadline':9999999999,
                    'phase_estimates':{'selected_fit_seconds':1,'prediction_seconds':1,'publication_seconds':1}}
                model = pipelines.build_pipeline(spec,task,context);model.fit(X,y,context)
                restored = cloudpickle.loads(cloudpickle.dumps(model))
                before = model.predict(X.iloc[:12],context).values['answer']
                after = restored.predict(X.iloc[:12],context).values['answer']
                assert np.array_equal(before,after) if classification else np.allclose(before,after)
                extra=X.copy();extra.index=['extra_'+str(i) for i in range(len(X))]
                labels=y.copy();labels.index=extra.index
                combined, targets = pd.concat([X,extra]),pd.concat([y,labels])
                if hasattr(restored,'refit'):
                    summary=restored.refit(combined,targets,context)
                else:
                    restored=pipelines.build_pipeline(restored.frozen_spec(),task,context)
                    summary=restored.fit(combined,targets,{**context,'execution_mode':'refit'})
                assert summary.rows==len(combined)
                prediction=restored.predict(extra.iloc[:12],context)
                values=np.asarray(prediction.values['answer'])
                assert np.isin(values,task.classes['answer']).all() if classification else np.isfinite(values).all()
                reports.append({'recipe':record['id'],'corpus':store.version,'fit_save_load_refit':True,
                    'full_refit_rows':summary.rows,'runtime_from_exact_archive':True})
                print(json.dumps(reports[-1]),flush=True)
    output.write_text(json.dumps({'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'real_qwen_requests':0,'passed':len(reports)==4,'recipes':reports},indent=2)+'\n')


if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]))
