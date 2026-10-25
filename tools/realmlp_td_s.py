"""Research port of published RealMLP-TD-S, not a production capability.

Paper arXiv:2407.04491v3, Table A.1/A.2; audited against pytabkit
c126ea51187c5080b91f28d352481dbd3b2194b0. Only organizer dependencies.
Explicit adaptation: fit-median numeric imputation before the author recipe;
the official sklearn interface rejects numeric NaNs. No added indicators.
"""
from __future__ import annotations
import math,time
import numpy as np
import pandas as pd


def coslog4(t):
    return .5*(1-math.cos(2*math.pi*math.log2(1+15*t)))


class RealTransform:
    def __init__(self,columns,categorical):self.columns=list(columns);self.categorical=list(categorical)
    def raw(self,X):
        a=X[self.numeric].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
        numeric=np.where(a.isna(),self.median,a).astype('float32')
        arrays=[numeric]
        for c,values in self.vocab.items():
            # Distinct raw statuses remain inspectable; the author's representation
            # deliberately maps both missing and unseen categories to the zero vector.
            code=X[c].astype(str).map({v:i for i,v in enumerate(values)}).fillna(-1).to_numpy(dtype='int64')
            code[X[c].isna().to_numpy()]=-2
            if len(values)<=2:
                arrays.append(np.where(code==0,1.,np.where(code==1,-1.,0.))[:,None])
            else:
                a=np.zeros((len(X),len(values)),dtype='float32');rows=np.flatnonzero(code>=0);a[rows,code[rows]]=1.;arrays.append(a)
        return np.concatenate(arrays,axis=1).astype('float32')
    def fit(self,X):
        if not len(X):raise ValueError('Nonempty fit required')
        self.numeric=[c for c in self.columns if c not in self.categorical]
        self.vocab={c:sorted(X[c].dropna().astype(str).unique().tolist()) for c in self.categorical}
        self.median=X[self.numeric].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).median().fillna(0).to_numpy()
        a=self.raw(X);self.center=np.quantile(a,.5,axis=0).astype('float32')
        delta=np.quantile(a,.75,axis=0)-np.quantile(a,.25,axis=0);spread=np.ptp(a,axis=0)
        denom=np.where(delta!=0,delta,.5*spread)
        self.scale=np.divide(1.,denom,out=np.zeros_like(denom),where=denom!=0).astype('float32')
        return self
    def transform(self,X):
        z=(self.raw(X)-self.center)*self.scale
        return (z/np.sqrt(1+(z/3)**2)).astype('float32')


def make_network(inputs,outputs,classification):
    import torch
    from torch import nn
    class Linear(nn.Module):
        def __init__(self,a,b,last=False):
            super().__init__();self.gain=a**-.5
            self.weight=nn.Parameter(torch.zeros(b,a) if last else torch.randn(b,a))
            self.bias=nn.Parameter(torch.zeros(b) if last else torch.randn(b))
        def forward(self,x):return nn.functional.linear(x,self.weight*self.gain,self.bias)
    class Net(nn.Module):
        def __init__(self):
            super().__init__();self.scale=nn.Parameter(torch.ones(inputs));self.layers=nn.ModuleList()
            for i,(a,b) in enumerate(zip([inputs,256,256,256],[256,256,256,outputs])):self.layers.append(Linear(a,b,i==3))
        def forward(self,x):
            x=x*self.scale
            for i,layer in enumerate(self.layers):
                x=layer(x)
                if i<3:x=nn.functional.selu(x) if classification else nn.functional.mish(x)
            return x
    return Net()


class RealMLPTDS:
    def __init__(self,columns,categorical,classes=None,seed=42,epochs=256,seconds=600):
        self.columns,self.categorical,self.classes=columns,categorical,classes
        self.seed,self.epochs,self.seconds=seed,epochs,seconds
    def fit(self,X,y,evaluation=None):
        began=time.monotonic();deadline=began+self.seconds
        import torch
        from torch.nn import functional as F
        for field in ['state','best_epoch']:
            if hasattr(self,field):delattr(self,field)
        torch.set_num_threads(1);torch.manual_seed(self.seed)
        self.transformer=RealTransform(self.columns,self.categorical).fit(X)
        data=self.transformer.transform(X);self.inputs=data.shape[1];outputs=len(self.classes) if self.classes else 1
        net=make_network(self.inputs,outputs,bool(self.classes))
        groups=[dict(params=[net.scale],factor=6.)]
        groups.extend(dict(params=[layer.weight],factor=1.) for layer in net.layers)
        groups.extend(dict(params=[layer.bias],factor=.1) for layer in net.layers)
        base_lr=.04 if self.classes else .07
        optimizer=torch.optim.Adam(groups,lr=base_lr,betas=(.9,.95),eps=1e-8,weight_decay=0.)
        raw=np.asarray(y)
        if self.classes:
            codes=pd.Categorical(raw,categories=self.classes).codes
            if (codes<0).any() or len(np.unique(codes))!=len(self.classes):raise ValueError('Complete declared fit class coverage required')
            target=torch.tensor(codes,dtype=torch.long);self.mean,self.std=0.,1.
        else:
            self.mean,self.std=float(raw.mean()),max(float(raw.std()),1e-12)
            target=torch.tensor((raw-self.mean)/self.std,dtype=torch.float32)
        tx=torch.from_numpy(data)
        valid=torch.from_numpy(self.transformer.transform(evaluation[0])) if evaluation is not None else None
        actual=pd.Categorical(evaluation[1],categories=self.classes).codes if evaluation is not None and self.classes else np.asarray(evaluation[1]) if evaluation is not None else None
        if self.classes and actual is not None and (actual<0).any():raise ValueError('Unknown stopping class')
        best=float('inf');state=None;self.curve=[];self.epochs_run=0;self.best_epoch=0
        batch_size=min(256,len(X));iterated=(len(X)//batch_size)*batch_size
        for epoch in range(self.epochs):
            net.train();order=torch.randperm(len(X))
            for start in range(0,iterated,batch_size):
                if time.monotonic()>=deadline:raise TimeoutError('Full RealMLP schedule/replay did not fit the assigned budget')
                # The horizon remains 256 when replaying an earlier selected epoch.
                progress=(epoch+start/iterated)/256.
                for group in optimizer.param_groups:group['lr']=base_lr*group['factor']*coslog4(progress)
                rows=order[start:start+batch_size];optimizer.zero_grad(set_to_none=True);z=net(tx[rows])
                loss=F.cross_entropy(z,target[rows],label_smoothing=.1) if self.classes else F.mse_loss(z[:,0],target[rows])
                if not torch.isfinite(loss):raise ValueError('Nonfinite RealMLP loss')
                loss.backward();optimizer.step()
            self.epochs_run=epoch+1
            if evaluation is not None:
                p=self._predict_tensor(net,valid)
                error=float(np.mean(p.argmax(1)!=actual)) if self.classes else float(np.mean((p-actual)**2))
                self.curve.append(dict(epoch=epoch+1,error=error))
                if error<=best:best=error;self.best_epoch=epoch+1;state={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
            if time.monotonic()>=deadline:raise TimeoutError('RealMLP budget includes preprocessing/validation')
        if evaluation is None:
            self.best_epoch=self.epochs_run;state={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
        self.state=state;self.seconds_used=time.monotonic()-began
        return self
    def _predict_tensor(self,net,tensor):
        import torch
        net.eval();parts=[]
        with torch.inference_mode():
            for start in range(0,len(tensor),1024):
                z=net(tensor[start:start+1024]);parts.append((z.softmax(1) if self.classes else z[:,0]*self.std+self.mean).numpy())
        return np.concatenate(parts)
    def predict(self,X):
        import torch
        if not len(X):return np.empty((0,len(self.classes))) if self.classes else np.empty(0)
        torch.set_num_threads(1);net=make_network(self.inputs,len(self.classes) if self.classes else 1,bool(self.classes))
        net.load_state_dict({k:torch.from_numpy(v) for k,v in self.state.items()})
        return self._predict_tensor(net,torch.from_numpy(self.transformer.transform(X)))
