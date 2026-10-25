"""One public-task audit: frozen train-CV reference and a stock Qwen agent.

Development data preparation is separate from solving. Withheld labels and
reference recipes are siblings of, not files inside, the agent task folder.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from automl_agent.util import atomic_json,file_hash


def prepare(out):
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    task=out/'task';task.mkdir();evaluation=out/'evaluation';evaluation.mkdir()
    data=load_breast_cancer(as_frame=True);X=data.data.copy()
    y=np.asarray(data.target_names)[data.target]
    fit,test=train_test_split(np.arange(len(X)),test_size=.2,stratify=y,random_state=42)
    train=X.iloc[fit].copy();train.insert(0,'id',fit);train['diagnosis']=y[fit]
    query=X.iloc[test].copy();query.insert(0,'id',test)
    train.to_csv(task/'train.csv',index=False);query.to_csv(task/'test.csv',index=False)
    pd.DataFrame({'id':test,'diagnosis':['benign']*len(test)}).to_csv(task/'sample_submission.csv',index=False)
    pd.DataFrame({'column':list(X.columns),'dtype':'numeric',
        'description':['Continuous nuclear morphology measurement; original WDBC mean, standard-error or worst-value summary.']*len(X.columns)}).to_csv(task/'data_dictionary.csv',index=False)
    (task/'task.md').write_text('''Predict diagnosis (benign or malignant) for independent WDBC observations using the supplied numeric morphology measurements. This is a binary classification benchmark. Primary metric: Macro F1, averaged equally over both declared classes. The id column identifies rows and is not a predictor. There is no declared time or subject-group transfer.

train.csv contains labels; test.csv has no labels. Preserve every test row in its original order. Write submission.csv with exactly the columns id,diagnosis and the class names benign or malignant, following sample_submission.csv. data_dictionary.csv describes the predictors. Source: UCI Breast Cancer Wisconsin (Diagnostic), CC BY 4.0, DOI 10.24432/C5DW2B.
''')
    pd.DataFrame({'id':test,'diagnosis':y[test]}).to_csv(evaluation/'withheld_labels.csv',index=False)
    # Choose the reference by training-only CV, never by external test score.
    pipe=Pipeline([('scale',StandardScaler()),('model',SVC())])
    grid=[{'scale':[StandardScaler()],'model':[SVC(kernel='rbf')],
        'model__C':[.1,1,10,100],'model__gamma':['scale',.001,.01,.1]},
        {'scale':[StandardScaler()],'model':[LogisticRegression(max_iter=2000)],'model__C':[.01,.1,1,10]},
        {'scale':['passthrough'],'model':[ExtraTreesClassifier(n_estimators=300,random_state=42,n_jobs=1)],'model__min_samples_leaf':[1,3]}]
    started=time.monotonic()
    search=GridSearchCV(pipe,grid,scoring='f1_macro',cv=StratifiedKFold(4,shuffle=True,random_state=43),n_jobs=1)
    search.fit(X.iloc[fit],y[fit]);p=search.predict(X.iloc[test])
    pd.DataFrame({'id':test,'diagnosis':p}).to_csv(evaluation/'reference_prediction.csv',index=False)
    atomic_json(evaluation/'reference.json',dict(selection='training-only 4-fold stratified CV',
        best_cv_score=search.best_score_,parameters={k:str(v) for k,v in search.best_params_.items()},
        candidates=len(search.cv_results_['params']),seconds=time.monotonic()-started,
        external_labels_used_for_selection=False,global_optimum_claimed=False))
    atomic_json(out/'manifest.json',dict(dataset='UCI WDBC redistributed by sklearn',
        source='https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic',
        license='CC BY 4.0',split='stratified 80/20 seed 42',train_rows=len(fit),test_rows=len(test),
        metric='macro_f1',classes=['benign','malignant'],seconds=600,max_calls=10,cpu=2,memory_mb=4096,
        isolation='native task-directory boundary; Docker daemon unavailable; no OS container isolation claim',
        source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in [ROOT/'solution.py',*sorted((ROOT/'automl_agent').glob('*.py'))]},
        task_files_sha256={p.name:file_hash(p) for p in task.iterdir()}))
    print('Prepared WDBC:',len(fit),'train,',len(test),'withheld test; reference frozen')


def live(out):
    from dotenv import dotenv_values
    import hashlib
    from datetime import datetime,timezone
    out=Path(out).resolve();env=os.environ.copy()
    for k,v in dotenv_values(ROOT/'.env').items():
        if v is not None:env.setdefault(k,v)
    env['OPENAI_MODEL']='qwen/qwen3.8-27b'
    env['OPENAI_BASE_URL']='https://changellenge.up.railway.app/v1'
    env.pop('OPENAI_EXTRA_BODY',None)
    env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    checker=ROOT/'skills/alpha-api-limits/scripts/check_limits.py'
    before=out/'api_limits_before.json';after=out/'api_limits_after.json'
    if (out/'live_started.json').exists():raise RuntimeError('No blind retry of a previously started live run')
    if before.exists():
        receipt=json.loads(before.read_text())
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(receipt['checked_at'])).total_seconds()
        if age>300 or receipt['key_fingerprint']!=hashlib.sha256(env.get('OPENAI_API_KEY','').encode()).hexdigest()[:16]:
            raise RuntimeError('Before snapshot is stale or belongs to another key')
        if receipt.get('budget_exhausted') or receipt.get('limits',{}).get('blocked'):
            raise RuntimeError('Organizer budget exhausted or blocked')
    else:
        rc=subprocess.run([sys.executable,str(checker),'before','--output',str(before)],env=env).returncode
        if rc:raise RuntimeError('Live run not started: organizer limit status unavailable or blocked')
    manifest=json.loads((out/'manifest.json').read_text())
    runtime=Path(manifest.get('runtime_path',ROOT))
    command=[sys.executable,str(runtime/'solution.py'),str(out/'task/task.md'),
        '--time-limit','600','--max-calls','10','--cpu','2','--memory-mb','4096','--run-id','public_context_audit']
    result=None;timed_out=False;started=time.monotonic();process=None
    atomic_json(out/'live_started.json',dict(started_at=datetime.now(timezone.utc).isoformat(),model=env['OPENAI_MODEL']))
    try:
        with (out/'agent.log').open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:result=process.wait(timeout=720)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        after_rc=subprocess.run([sys.executable,str(checker),'after','--before',str(before),'--output',str(after)],env=env).returncode
        atomic_json(out/'live.json',dict(returncode=result,external_timeout=timed_out,seconds=time.monotonic()-started,
            api_after_confirmed=after_rc in {0,3},model=env['OPENAI_MODEL']))


def analyze(out):
    import sqlite3
    import pandas as pd
    from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
    from automl_agent.util import canonical
    out=Path(out).resolve();root=out/'task/.agent_work/public_context_audit'
    report=json.loads((root/'run_report.json').read_text())
    labels=pd.read_csv(out/'evaluation/withheld_labels.csv')
    metrics={}
    for name,path in [('reference',out/'evaluation/reference_prediction.csv'),('agent',out/'task/submission.csv')]:
        prediction=pd.read_csv(path)
        if list(prediction)!=['id','diagnosis'] or prediction.id.tolist()!=labels.id.tolist():raise ValueError('Output contract differs')
        if not set(prediction.diagnosis)<={'benign','malignant'}:raise ValueError('Undeclared class')
        metrics[name]=dict(macro_f1=float(f1_score(labels.diagnosis,prediction.diagnosis,average='macro')),
            accuracy=float(accuracy_score(labels.diagnosis,prediction.diagnosis)),
            errors=int((labels.diagnosis!=prediction.diagnosis).sum()),
            confusion=confusion_matrix(labels.diagnosis,prediction.diagnosis,labels=['benign','malignant']).tolist())
    with sqlite3.connect('file:'+str((root/'journal.sqlite').resolve())+'?mode=ro',uri=True) as db:
        counts=dict(db.execute('select kind,count(*) from events group by kind'))
        decisions=[]
        for (raw,) in db.execute("select payload from events where kind='qwen_decision' order by seq"):
            c=json.loads(raw)['command'];decisions.append(dict(tool=c['tool'],operation_id=c['operation_id']))
        errors=[]
        for identifier,raw in db.execute("select id,record from operations where json_extract(record,'$.status')='failed'"):
            o=json.loads(raw);r=o.get('result') or {};body=r.get('data',r)
            errors.append(dict(operation_id=identifier,tool=o.get('tool'),error=body.get('error'),status=r.get('status')))
        usage=[json.loads(raw) for (raw,) in db.execute('select usage_json from cc_usage')]
    requests=[]
    for path in sorted((root/'requests').glob('*.json')):
        q=json.loads(path.read_text());parts=[]
        for message in q.get('messages',[]):
            content=message.get('content','');m=dict(role=message['role'],bytes=len(canonical(content).encode()))
            if isinstance(content,str):
                try:
                    obj=json.loads(content)
                    if isinstance(obj,dict):m['components_bytes']={k:len(canonical(v).encode()) for k,v in obj.items()}
                except json.JSONDecodeError:pass
            parts.append(m)
        requests.append(dict(file=path.name,wire_bytes=len(canonical(q).encode()),
            native=bool(q.get('tools')),tools_bytes=len(canonical(q.get('tools',[])).encode()),messages=parts))
    before=json.loads((out/'api_limits_before.json').read_text());after=json.loads((out/'api_limits_after.json').read_text())
    result=dict(metrics=metrics,agent_minus_reference_f1=metrics['agent']['macro_f1']-metrics['reference']['macro_f1'],
        state=report['state'],control=report['control_status'],refit=report['refit_status'],
        calls=report['model_calls'],token_usage=report['token_usage'],events=counts,decisions=decisions,rejections=errors,
        requests=requests,max_server_prompt_tokens=max([u['prompt_tokens'] for u in usage] or [0]),
        model_time_including_queue=report['time_accounting']['model_including_queue_seconds'],
        training_seconds=report['time_accounting']['useful_training_seconds'],
        scope='one live run, public WDBC holdout; not contest quality or causal A/B',
        api=dict(before=before['remaining_budget'],after=after['remaining_budget'],increase=after.get('comparison',{}).get('increase')))
    atomic_json(out/'analysis.json',result)
    print(json.dumps({k:result[k] for k in ['metrics','state','calls','max_server_prompt_tokens','api']},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','live','analyze']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.output)
    elif a.action=='live':live(a.output)
    else:analyze(a.output)
