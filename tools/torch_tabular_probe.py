"""Offline PyTorch feasibility probe; not a contest or live-agent capability.

Uses task roles and existing splits, never control/test labels. No model hubs,
external packages or pretrained weights. Saves preprocessing and numeric state.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


class InputTransform:
    def __init__(self, columns, categorical, encoding='standard',dual_numeric=False):
        self.columns, self.categorical, self.encoding = list(columns), list(categorical), encoding
        self.dual_numeric=dual_numeric

    def fit(self, frame):
        if not len(frame):
            raise ValueError('Neural preprocessing needs nonempty fit rows')
        self.numeric = [c for c in self.columns if c not in self.categorical]
        self.encoded_categorical=self.categorical+[c for c in self.numeric
            if self.dual_numeric and 1<frame[c].nunique(dropna=True)<=32]
        self.maps = {}
        for c in self.encoded_categorical:
            values = frame[c].dropna().astype(str).value_counts().head(2000).index
            self.maps[c] = {v: i+2 for i, v in enumerate(sorted(values))}
        a = frame[self.numeric].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan)
        self.median = a.median().fillna(0).to_numpy()
        raw = np.where(a.isna(),self.median,a).astype(float)
        self.center, self.scale = raw.mean(axis=0), np.maximum(raw.std(axis=0),1e-6)
        self.knots = [np.unique(np.quantile(raw[:,i],np.linspace(0,1,9))) for i in range(raw.shape[1])]
        self.quantile=None
        if self.encoding=='quantile' and raw.shape[1]:
            from sklearn.preprocessing import QuantileTransformer
            self.quantile=QuantileTransformer(n_quantiles=min(256,len(raw)),output_distribution='normal',random_state=42).fit(raw)
        if self.encoding not in {'standard','piecewise','quantile'}:
            raise ValueError('Unsupported numeric representation')
        return self

    def transform(self, frame):
        a = frame[self.numeric].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
        missing = a.isna().to_numpy()
        raw = np.where(missing,self.median,a).astype(float)
        quantile=getattr(self,'quantile',None)
        values = [quantile.transform(raw) if quantile is not None else np.clip((raw-self.center)/self.scale,-10,10),missing.astype(float)]
        if self.encoding=='piecewise':
            for i,k in enumerate(self.knots):
                if len(k)>1:
                    values.append(np.clip((raw[:,i,None]-k[:-1])/(k[1:]-k[:-1]),0,1))
        num = np.concatenate(values,axis=1).astype('float32')
        cats=[]
        for c in getattr(self,'encoded_categorical',self.categorical):
            code=frame[c].astype(str).map(self.maps[c]).fillna(0).to_numpy(dtype='int64')
            code[frame[c].isna().to_numpy()]=1
            cats.append(code)
        cat=np.column_stack(cats) if cats else np.empty((len(frame),0),dtype='int64')
        return num,cat


def network(numeric, cardinalities, output, width=128, depth=2, heads=1):
    import torch
    from torch import nn

    class BatchLinear(nn.Module):
        def __init__(self, n_in, n_out, first):
            super().__init__()
            self.weight=nn.Parameter(torch.empty(n_in,n_out))
            nn.init.kaiming_uniform_(self.weight.T,a=5**.5)
            self.r=nn.Parameter(torch.randn(heads,n_in) if first else torch.ones(heads,n_in))
            self.s=nn.Parameter(torch.ones(heads,n_out))
            self.bias=nn.Parameter(torch.zeros(heads,n_out))

        def forward(self,x):
            return ((x*self.r)@self.weight)*self.s+self.bias

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            dimensions=[min(16,max(2,round(n**.5))) for n in cardinalities]
            self.embeddings=nn.ModuleList([nn.Embedding(n,d) for n,d in zip(cardinalities,dimensions)])
            layers=[];inputs=numeric+sum(dimensions)
            for i in range(depth):
                layers.extend([BatchLinear(inputs,width,i==0) if heads>1 else nn.Linear(inputs,width),
                    nn.LayerNorm(width),nn.ReLU(),nn.Dropout(.1)])
                inputs=width
            layers.append(BatchLinear(inputs,output,False) if heads>1 else nn.Linear(inputs,output))
            self.body=nn.Sequential(*layers)

        def forward(self, num, cat):
            x=torch.cat([num]+[m(cat[:,i]) for i,m in enumerate(self.embeddings)],dim=1)
            return self.body(x[:,None,:].expand(-1,heads,-1) if heads>1 else x)
    return Net()


class TorchMLP:
    def __init__(self, columns, categorical, classes=None, encoding='standard', epochs=30, seconds=30, seed=42,
                 width=128,depth=2,heads=1,learning_rate=.001,dual_numeric=False):
        if epochs<1 or seconds<=0 or min(width,depth,heads)<1:
            raise ValueError('Positive neural epochs, time and architecture dimensions required')
        self.columns,self.categorical,self.classes=columns,categorical,classes
        self.encoding,self.epochs,self.seconds,self.seed=encoding,epochs,seconds,seed
        self.width,self.depth,self.heads,self.learning_rate=width,depth,heads,learning_rate
        self.dual_numeric=dual_numeric

    def fit(self, X, y, evaluation=None):
        import torch
        from torch.nn import functional as F
        from sklearn.metrics import f1_score
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        rng=np.random.default_rng(self.seed)
        self.transformer=InputTransform(self.columns,self.categorical,self.encoding,self.dual_numeric).fit(X)
        a,b=self.transformer.transform(X)
        self.shape=(a.shape[1],[len(self.transformer.maps[c])+2 for c in self.transformer.encoded_categorical],len(self.classes) if self.classes else 1)
        net=network(*self.shape,width=self.width,depth=self.depth,heads=self.heads)
        optimizer=torch.optim.AdamW(net.parameters(),lr=self.learning_rate,weight_decay=.0003)
        raw=np.asarray(y)
        if self.classes:
            target=pd.Categorical(raw,categories=self.classes).codes
            if (target<0).any():raise ValueError('Undeclared fit label')
            frequency=np.bincount(target,minlength=len(self.classes))
            if (frequency==0).any():raise ValueError('Missing declared fit class')
            weights=torch.tensor(np.sqrt(frequency.max()/frequency),dtype=torch.float32)
            ty=torch.tensor(target,dtype=torch.long)
            self.mean,self.scale=0.,1.
        else:
            self.mean,self.scale=float(raw.mean()),max(float(raw.std()),1e-6)
            ty=torch.tensor((raw-self.mean)/self.scale,dtype=torch.float32)
        tx,tc=torch.from_numpy(a),torch.from_numpy(b)
        started=time.monotonic();best=None;best_score=-float('inf');bad=0
        self.curve=[];self.best_epoch=0
        for epoch in range(self.epochs):
            net.train()
            for start in range(0,len(X),256):
                if start==0:order=rng.permutation(len(X))
                rows=order[start:start+256]
                optimizer.zero_grad(set_to_none=True)
                z=net(tx[rows],tc[rows])
                if self.classes:
                    loss=F.cross_entropy(z.reshape(-1,len(self.classes)),
                        ty[rows,None].expand(-1,self.heads).reshape(-1) if self.heads>1 else ty[rows],weight=weights)
                else:
                    loss=F.mse_loss(z[:,:,0],ty[rows,None].expand(-1,self.heads)) if self.heads>1 else F.mse_loss(z[:,0],ty[rows])
                if not torch.isfinite(loss):raise ValueError('Nonfinite neural loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            self.epochs_run=epoch+1
            if evaluation is not None:
                p=self._infer(net,evaluation[0])
                if self.classes:
                    target=pd.Categorical(evaluation[1],categories=self.classes).codes
                    score=float(f1_score(target,p.argmax(axis=1),labels=np.arange(len(self.classes)),average='macro',zero_division=0))
                else:score=-float(np.sqrt(np.mean((np.asarray(evaluation[1])-p)**2)))
                self.curve.append(dict(epoch=epoch+1,score=score))
                if score>best_score:
                    best_score=score;best={k:v.detach().numpy().copy() for k,v in net.state_dict().items()};self.best_epoch=epoch+1;bad=0
                else:bad+=1
            if time.monotonic()-started>=self.seconds or bad>=7:break
        if best is None:
            best={k:v.detach().numpy().copy() for k,v in net.state_dict().items()};self.best_epoch=self.epochs_run
        self.state=best
        self.seconds_used=time.monotonic()-started
        return self

    def _infer(self, net, X):
        import torch
        if not len(X):
            return np.empty((0,len(self.classes)),dtype='float32') if self.classes else np.empty(0,dtype='float32')
        net.eval();a,b=self.transformer.transform(X);parts=[]
        heads=getattr(self,'heads',1)
        with torch.inference_mode():
            for start in range(0,len(X),512):
                z=net(torch.from_numpy(a[start:start+512]),torch.from_numpy(b[start:start+512]))
                if self.classes:
                    p=torch.softmax(z,dim=-1)
                    if heads>1:p=p.mean(dim=1)
                else:
                    p=(z[:,:,0].mean(dim=1) if heads>1 else z[:,0])*self.scale+self.mean
                parts.append(p.numpy())
        return np.concatenate(parts)

    def predict(self,X):
        import torch
        torch.set_num_threads(1)
        net=network(*self.shape,width=getattr(self,'width',128),depth=getattr(self,'depth',2),heads=getattr(self,'heads',1))
        net.load_state_dict({k:torch.from_numpy(v) for k,v in self.state.items()})
        return self._infer(net,X)


def stopping_strategy(plan):
    return 'exact_group' if plan.evaluation_mode in {'exact_group','low_overlap'} else plan.strategy


def worker(request):
    import joblib
    import resource
    from tools.plan_development_benchmark import tasks
    from automl_agent.protocols import initial_protocol
    from automl_agent.validation import choose_validation,inner_split
    from automl_agent.identity import feature_groups
    from automl_agent.custom import validate_code
    from automl_agent.util import atomic_json,file_hash
    q=json.loads(Path(request).read_text());out=Path(q['directory']);dataset=tasks()[q['kind']]
    task=dataset.task;target=task.targets[0]
    plan=initial_protocol(dataset,choose_validation(dataset,seed=42))
    fit=np.asarray(plan.fit);validation=np.asarray(plan.validation)
    X=dataset.train[task.feature_columns];y=dataset.train[[target]]
    # No sample cap: structural group/time boundaries stay exactly as selected
    # by the current controller, and no control or external-test labels enter.
    strategy=stopping_strategy(plan)
    a,b=inner_split(X.iloc[fit],y.iloc[fit],task,strategy,42)
    if strategy=='exact_group':
        groups=feature_groups(X.iloc[fit][task.feature_columns])
        assert not set(groups[a]) & set(groups[b])
    classes=task.classes.get(target) if task.classes else None
    np.savez(out/'splits.npz',fit=fit,validation=validation,inner_fit=fit[a],inner_stop=fit[b])
    try:validate_code('import torch\ndef build_pipeline(context):\n    return None\n','pipeline')
    except ValueError as error:barrier=str(error)
    else:barrier='not_rejected'
    configs=[dict(name='standard',encoding='standard'),dict(name='piecewise',encoding='piecewise')]
    if q.get('strong'):
        configs=[dict(name='quantile_mlp',encoding='quantile',width=256,depth=3,heads=1,epochs=100,learning_rate=.002),
            dict(name='quantile_batchensemble',encoding='quantile',width=128,depth=3,heads=8,epochs=100,learning_rate=.002),
            dict(name='piecewise_batchensemble',encoding='piecewise',width=128,depth=3,heads=8,epochs=100,learning_rate=.002)]
    if q.get('dual'):
        configs=[dict(name='dual_piecewise_mlp',encoding='piecewise',dual_numeric=True,width=256,depth=3,heads=1,epochs=100,learning_rate=.002),
            dict(name='dual_piecewise_ensemble',encoding='piecewise',dual_numeric=True,width=128,depth=3,heads=4,epochs=100,learning_rate=.002)]
    atomic_json(out/'protocol.json',dict(kind=q['kind'],scope='offline_feasibility_not_agent_run',
        strategy=plan.strategy,inner_strategy=strategy,evaluation_mode=plan.evaluation_mode,fit_rows=len(fit),validation_rows=len(validation),
        inner_fit_rows=len(a),inner_stop_rows=len(b),control_used=False,test_labels_used=False,
        columns=task.feature_columns,categorical=task.categorical_columns,classes=classes,
        source_hash=file_hash(Path(__file__)),split_hash=file_hash(out/'splits.npz'),
        data_hash=dataset.source_hash,agent_barrier=barrier,threads=1,seed=42,
        architectures=configs,epochs_cap=100 if q.get('strong') or q.get('dual') else 30,
        seconds_per_fit=60 if q.get('dual') else 30,
        comparison='Adaptive development pilot; fixed baseline 200-tree LightGBM; not previous reference splits or scores'))
    results=[]
    for config in configs:
        name=config['name'];kwargs={k:v for k,v in config.items() if k!='name'}
        seconds=60 if q.get('dual') else 30
        selected=TorchMLP(task.feature_columns,task.categorical_columns,classes,seconds=seconds,**kwargs)
        selected.fit(X.iloc[fit[a]],y.iloc[fit[a]][target],(X.iloc[fit[b]],y.iloc[fit[b]][target]))
        model=TorchMLP(task.feature_columns,task.categorical_columns,classes,seconds=seconds,**dict(kwargs,epochs=selected.best_epoch))
        model.fit(X.iloc[fit],y.iloc[fit][target]);p=model.predict(X.iloc[validation])
        model_path=out/f'{name}.joblib';joblib.dump(model,model_path)
        np.testing.assert_allclose(joblib.load(model_path).predict(X.iloc[validation]),p,rtol=1e-6)
        np.save(out/f'{name}_prediction.npy',p)
        score=metric(y.iloc[validation][target],p,classes)
        results.append(dict(model=name,score=score,best_epoch=selected.best_epoch,refit_epochs=model.epochs_run,
            refit_truncated=model.epochs_run<selected.best_epoch,inner_seconds=selected.seconds_used,
            refit_seconds=model.seconds_used,fit_count=2,inner_curve=selected.curve,inference_roundtrip=True))
        print(name,score,flush=True)
    from automl_agent.contracts import PipelineSpec
    from automl_agent.pipelines import make_estimator
    frame=X.copy();cat=task.categorical_columns
    for c in cat:frame[c]=pd.Categorical(frame[c].astype(str),categories=sorted(frame.iloc[fit][c].astype(str).unique()))
    model=make_estimator(PipelineSpec(model='lightgbm',hyperparameters={'n_estimators':200,'num_leaves':31,'learning_rate':.05}),bool(classes),{'cpu':1})
    labels=pd.Categorical(y.iloc[fit][target],categories=classes).codes if classes else y.iloc[fit][target]
    kwargs={}
    if classes:
        frequency=np.bincount(labels,minlength=len(classes));kwargs['sample_weight']=np.sqrt(frequency.max()/frequency)[labels]
    started=time.monotonic();model.fit(frame.iloc[fit],labels,**kwargs)
    p=model.predict_proba(frame.iloc[validation]) if classes else model.predict(frame.iloc[validation])
    results.append(dict(model='lightgbm',score=metric(y.iloc[validation][target],p,classes),fit_count=1,seconds=time.monotonic()-started))
    atomic_json(out/'results.json',dict(results=results,max_rss_macos_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        api_calls=0,live_agent_run=False,promotion=False,unknown='external quality, seed stability, container equivalence and autonomous choice untested'))


def metric(y,p,classes):
    from sklearn.metrics import f1_score
    if classes:
        target=pd.Categorical(y,categories=classes).codes
        return float(f1_score(target,p.argmax(axis=1),labels=np.arange(len(classes)),average='macro',zero_division=0))
    return float(np.sqrt(np.mean((np.asarray(y)-p)**2)))


def run(output,strong=False,dual=False):
    from automl_agent.util import atomic_json
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    failed=[]
    for kind in ['regression','classification']:
        directory=out/kind;directory.mkdir();request=directory/'request.json'
        atomic_json(request,dict(kind=kind,directory=str(directory),strong=strong,dual=dual))
        env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','TMPDIR'}}
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        shutil.copyfile(Path(__file__),directory/'source_snapshot.py')
        try:
            with (directory/'worker.log').open('w') as log:
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(request)],
                    env=env,stdout=log,stderr=subprocess.STDOUT,timeout=420 if strong or dual else 180)
            if result.returncode:
                atomic_json(directory/'failure.json',dict(returncode=result.returncode))
                failed.append(kind)
        except subprocess.TimeoutExpired:
            atomic_json(directory/'failure.json',dict(reason='worker_timeout',seconds=420 if strong or dual else 180))
            failed.append(kind)
        print(kind,'done',flush=True)
    if failed:raise RuntimeError('Offline probe failed: '+', '.join(failed))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path);parser.add_argument('--worker',type=Path)
    parser.add_argument('--strong',action='store_true')
    parser.add_argument('--dual',action='store_true')
    args=parser.parse_args()
    if args.worker:
        # Persist an importable class identity, not __main__.TorchMLP.
        from tools.torch_tabular_probe import worker as execute_worker
        execute_worker(args.worker)
    elif args.output:run(args.output,args.strong,args.dual)
    else:parser.error('Provide --output or --worker')
