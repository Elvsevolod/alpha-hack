"""Standard-library test runner for the exact organizer dependency environment."""
from __future__ import annotations

import argparse
import errno
import importlib.metadata
import json
from pathlib import Path
import re
import socket
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from automl_agent.budget import Limits
from automl_agent.contracts import Hypothesis,PipelineSpec,ToolCommand
from automl_agent.controller import Controller
from automl_agent.util import atomic_json,file_hash

ROOT=Path(__file__).resolve().parents[1]


def environment():
    requirement=ROOT/'requirements.txt'
    packages={};failures=[]
    for line in requirement.read_text().splitlines():
        match=re.fullmatch(r'([A-Za-z0-9_.-]+)(?:\[([^]]+)\])?==([^\s#]+)',line.strip())
        if not match:continue
        try:actual=importlib.metadata.version(match[1])
        except importlib.metadata.PackageNotFoundError:actual='absent'
        packages[match[1]]={'required':match[3],'actual':actual,'extras':match[2]}
        if actual!=match[3]:failures.append(match[1])
    interfaces=sorted(name for _,name in socket.if_nameindex())
    routes=Path('/proc/net/route').read_text() if Path('/proc/net/route').is_file() else ''
    probe=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    try:
        # UDP connect asks the local kernel for a route and sends no payload.
        probe.connect(('192.0.2.1',9));route_error=None
    except OSError as error:route_error=error.errno
    finally:probe.close()
    offline=route_error in {errno.ENETUNREACH,errno.EHOSTUNREACH} and not any(line.strip() for line in routes.splitlines()[1:])
    return {'python':sys.version,'python_312':sys.version_info[:2]==(3,12),'packages':packages,
        'requirements_sha256':file_hash(requirement),'version_failures':failures,'network_interfaces':interfaces,
        'network_route_errno':route_error,'ipv4_routes':routes,'network_namespace_offline':offline}


def fixture(root,kind='regression',text=False):
    root.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(73);n=240;nt=30
    values=rng.normal(size=n+nt)
    frame=pd.DataFrame({'id':[f'{i:05d}' for i in range(n+nt)],'x':values,'cat':np.where(values>0,'yes','no'),
        'all_missing':np.full(n+nt,np.nan)})
    frame.loc[::9,'x']=np.nan
    categorical=['cat'];texts=[];time_column=None;series=[];group=None;horizon=None
    targets=['target'];classes={};metric={'name':'rmse'};prediction='values'
    frame['target']=values*2+.1*rng.normal(size=n+nt)
    if kind in {'classification','multioutput_classification','multilabel'}:
        frame['target']=np.where(values>0,'positive','negative');metric={'name':'f1','average':'macro'};prediction='labels'
        if kind=='multioutput_classification':
            frame['other']=np.where(values>.5,'high',np.where(values<-.5,'low','middle'));targets.append('other')
        if kind=='multilabel':
            frame['target']=(values>0).astype(int);frame['other']=np.zeros(n+nt,dtype=int);targets.append('other')
    elif kind=='multioutput_regression':
        frame['other']=values**2;targets.append('other')
    elif kind=='forecasting':
        # A regular panel with two independent series and a held-out horizon.
        n=120;nt=12
        frame=pd.DataFrame([{'id':f'{s}_{t:04d}','time':t,'series':str(s),'x':float(t),'target':float(t+s)}
                            for t in range(66) for s in range(2)])
        categorical=['series'];time_column='time';series=['series'];horizon=3
    elif kind=='ranking':
        frame['query']=np.arange(n+nt)//5;frame['target']=np.tile([0,1,2,3,1],(n+nt)//5)
        group='query';metric={'name':'ndcg','k':3};prediction='scores'
    if text:
        frame['notes']=np.where(np.arange(len(frame))%2,'some warm text','other cold text');texts=['notes']
    train=frame.iloc[:n];test=frame.iloc[n:].drop(columns=targets)
    train.to_csv(root/'train.csv',index=False);test.to_csv(root/'test.csv',index=False)
    sample=pd.DataFrame({'id':test.id})
    for target in targets:sample[target]=0
    sample.to_csv(root/'sample_submission.csv',index=False)
    spec={'kind':kind,'targets':targets,'id_columns':['id'],'feature_columns':[c for c in test if c!='id'],
        'categorical_columns':categorical,'text_columns':texts,'time_column':time_column,'series_columns':series,
        'group_column':group,'horizon':horizon,'metric':metric,'output':{'columns':['id',*targets],'prediction_type':prediction},
        'evidence':{'contract':['task','schema']}}
    (root/'task_spec.json').write_text(json.dumps(spec));(root/'task.md').write_text('Synthetic offline technical fixture; no external training data.\n')
    return root/'task.md'


TORCH_CODE='''import time
import numpy as np
import torch
from automl_agent.api import TrainingSummary,prediction_bundle
from automl_agent.preprocessing import FeatureMatrix
from automl_agent.contracts import PipelineSpec,TaskSpec
class Candidate:
    def fit(self,X,y,context):
        start=time.monotonic();torch.set_num_threads(context['cpu']);torch.manual_seed(context['seed'])
        self.task=TaskSpec.model_validate(context['task']);self.features=FeatureMatrix(PipelineSpec(model='ridge'),self.task)
        matrix=self.features.fit_transform(X,y).toarray().astype(np.float32)
        self.model=torch.nn.Sequential(torch.nn.Linear(matrix.shape[1],8),torch.nn.ReLU(),torch.nn.Linear(8,1))
        optimizer=torch.optim.Adam(self.model.parameters(),lr=.02)
        a=torch.as_tensor(matrix);b=torch.as_tensor(y.to_numpy(dtype=np.float32))
        for epoch in range(20):
            optimizer.zero_grad();loss=torch.mean((self.model(a)-b)**2);loss.backward();optimizer.step()
        self.model.eval()
        return TrainingSummary(rows=len(X),seconds=time.monotonic()-start)
    def predict(self,X,context):
        matrix=self.features.transform(X).toarray().astype(np.float32)
        with torch.no_grad():values=self.model(torch.as_tensor(matrix)).numpy().reshape(-1)
        return prediction_bundle({context['task']['targets'][0]:values},X,context,'values')
def build_pipeline(context):return Candidate()
'''

CLASSICAL_CODE='''from automl_agent.pipelines import DirectPipeline
from automl_agent.contracts import PipelineSpec, TaskSpec
def build_pipeline(context):
    return DirectPipeline(PipelineSpec(model="ridge", seed=context["seed"]), TaskSpec.model_validate(context["task"]))
'''


def case(root,name,kind,pipeline,code=None,text=False):
    path=fixture(root/name,kind,text)
    controller=Controller(path,Limits(seconds=120,cpu=2,memory_mb=5120,max_candidates=2,max_calls=0),run_id='offline')
    try:
        controller.prepare()
        if code is not None:
            # A local scripted critic permits exercising the real custom worker
            # without authorizing any model request in this technical fixture.
            controller.qwen.critique=lambda request, **kwargs: dict(review_id=request['review_id'], verdict='approve',
                evaluation_scope='research_validation', comparator_id=controller.journal.get('baseline_id'),
                evidence_assessment='Fixed offline adapter fixture', comparison_assessment='Compare the measured baseline',
                leakage_assessment='Only fit rows reach this worker', cost_assessment='Bounded local test',
                falsification_assessment='Require finite aligned predictions and frozen refit', revision_requests=[])
        arguments={'pipeline':pipeline.model_dump(mode='json'),'hypothesis':Hypothesis(observation='Offline synthetic fixture',
            expected_effect='Verify supported adapter behavior',changed_component=name,acceptance='Finite aligned outputs, persistence and fixed refit').model_dump(),
            'budget_seconds':60}
        if code is not None:arguments['code']=code
        tool='run_custom_experiment' if code is not None else 'run_automl' if pipeline.engine in {'flaml','autogluon','lightautoml'} else 'tune_model' if pipeline.engine=='optuna' else 'train_model'
        operation=controller.dispatch(ToolCommand(operation_id='candidate',tool=tool,arguments=arguments))
        operation=operation.get('data',operation)
        result=operation.get('result',{})
        controller.journal.put('imputation_skip_reason','Technical adapter fixture; quality comparison is separate')
        controller.freeze('Offline adapter verification complete')
        controller.finalize();report=controller.report()
        published=pd.read_csv(controller.publisher.output,dtype=str,keep_default_na=False)
        expected=pd.read_csv(path.parent/'test.csv',dtype=str,keep_default_na=False)
        passed=result.get('status')=='completed' and result.get('score') is not None and report['state']=='completed' and published.id.tolist()==expected.id.tolist() and report['control_status']=='completed' and report['refit_status']=='completed'
        return {'name':name,'passed':passed,'candidate_status':result.get('status'), 'candidate_error':result.get('error') or operation.get('error'),
            'control':report['control_status'],'refit':report['refit_status'],'execution_policy':report['execution_policy'],
            'publication':report['publication'],'run_report':str(controller.workdir/'run_report.json')}
    finally:controller.close()


def run(output):
    output.mkdir(parents=True,exist_ok=True);env=environment();atomic_json(output/'environment.json',env)
    results=[]
    cases=[('linear','regression',PipelineSpec(model='ridge'),None,False),
        ('categorical','classification',PipelineSpec(model='catboost',representation='native',hyperparameters={'iterations':20}),None,False),
        ('text','classification',PipelineSpec(model='logistic',representation='mixed'),None,True),
        ('multioutput_regression','multioutput_regression',PipelineSpec(model='extra_trees',hyperparameters={'n_estimators':20}),None,False),
        ('multioutput_classification','multioutput_classification',PipelineSpec(model='lightgbm',hyperparameters={'n_estimators':20}),None,False),
        ('multilabel_constant','multilabel',PipelineSpec(model='logistic'),None,False),
        ('panel_forecast','forecasting',PipelineSpec(model='lightgbm',representation='lags',hyperparameters={'n_estimators':20}),None,False),
        ('ranking','ranking',PipelineSpec(model='lgbm_ranker',representation='native',hyperparameters={'n_estimators':20}),None,False),
        ('flaml','regression',PipelineSpec(engine='flaml',model='library_search',learners=['lgbm'],n_trials=2),None,False),
        ('autogluon','classification',PipelineSpec(engine='autogluon',model='library_search',learners=['GBM'],n_trials=2),None,False),
        ('optuna','regression',PipelineSpec(engine='optuna',model='ridge',n_trials=2,search_space={'alpha':{'type':'float','low':.1,'high':2.}}),None,False),
        ('custom_classical','regression',PipelineSpec(engine='custom',model='ridge'),CLASSICAL_CODE,False),
        ('knn','classification',PipelineSpec(model='knn',hyperparameters={'n_neighbors':5}),None,False),
        ('lightautoml','regression',PipelineSpec(engine='lightautoml',model='lightautoml',learners=['lgb']),None,False),
        ('autogluon_explicit','classification',PipelineSpec(engine='autogluon',model='library_search',learners=['GBM'],n_trials=1,
            autogluon_configurations={'GBM':[{'n_estimators':20,'num_leaves':7},{'n_estimators':20,'num_leaves':15}]}),None,False)]
    for args in cases:
        result=case(output,*args);results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in {'execution_policy','publication'}},ensure_ascii=False),flush=True)
        report={'environment':env,'cases':results,'passed':sum(r['passed'] for r in results),'total':len(results),
            'exact_environment':not env['version_failures'] and env['python_312'],'all_passed':all(r['passed'] for r in results),
            'offline':env['network_namespace_offline']}
        atomic_json(output/'REPORT.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('/tmp/offline_acceptance'))
    report=run(parser.parse_args().output)
    raise SystemExit(0 if report['all_passed'] and report['exact_environment'] and report['offline'] else 1)
