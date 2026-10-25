"""Bounded, randomly initialized tabular nets using organizer dependencies only.

Derived from the 2026-10-25 offline prototype (see provenance manifest).
No external weights, downloads, model hubs or additional packages.
"""
from __future__ import annotations
import time
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
        if self.encoding not in {'standard','piecewise','learned_piecewise','quantile','periodic'}:
            raise ValueError('Unsupported numeric representation')
        return self

    def transform(self, frame):
        a = frame[self.numeric].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
        missing = a.isna().to_numpy()
        raw = np.where(missing,self.median,a).astype(float)
        quantile=getattr(self,'quantile',None)
        values = [quantile.transform(raw) if quantile is not None else np.clip((raw-self.center)/self.scale,-10,10),missing.astype(float)]
        if self.encoding in {'piecewise','learned_piecewise'}:
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


def network(numeric, cardinalities, output, width=128, depth=2, heads=1, periodic=False, residual=False, piece_sizes=(), attention=False):
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

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.block=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.ReLU(),nn.Dropout(.1),nn.Linear(width,width))
        def forward(self,x):
            return x+self.block(x)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention=attention
            if attention:
                d=32
                self.numeric_weight=nn.Parameter(torch.randn(numeric,d)*.02)
                self.numeric_bias=nn.Parameter(torch.zeros(numeric,d))
                self.embeddings=nn.ModuleList([nn.Embedding(n,d) for n in cardinalities])
                self.cls=nn.Parameter(torch.randn(1,1,d)*.02)
                self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,2,64,.1,batch_first=True),1,enable_nested_tensor=False)
                self.output=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,output))
                return
            dimensions=[min(16,max(2,round(n**.5))) for n in cardinalities]
            self.embeddings=nn.ModuleList([nn.Embedding(n,d) for n,d in zip(cardinalities,dimensions)])
            self.piece_sizes=tuple(piece_sizes)
            self.piece_modules=nn.ModuleList([nn.Linear(n,8) for n in piece_sizes])
            numeric_width=numeric-sum(piece_sizes)+8*len(piece_sizes)
            self.frequency=nn.Parameter(torch.randn(numeric,8)*.1) if periodic else None
            layers=[];inputs=(numeric*17 if periodic else numeric_width)+sum(dimensions)
            if residual:
                layers=[nn.Linear(inputs,width),*[Residual() for _ in range(depth)],nn.LayerNorm(width),nn.Linear(width,output)]
                self.body=nn.Sequential(*layers)
                return
            for i in range(depth):
                layers.extend([BatchLinear(inputs,width,i==0) if heads>1 else nn.Linear(inputs,width),
                    nn.LayerNorm(width),nn.ReLU(),nn.Dropout(.1)])
                inputs=width
            layers.append(BatchLinear(inputs,output,False) if heads>1 else nn.Linear(inputs,output))
            self.body=nn.Sequential(*layers)

        def forward(self, num, cat):
            if self.attention:
                numeric=num[:,:,None]*self.numeric_weight[None,:,:]+self.numeric_bias[None,:,:]
                tokens=[self.cls.expand(len(num),-1,-1),numeric]+[m(cat[:,i])[:,None,:] for i,m in enumerate(self.embeddings)]
                return self.output(self.encoder(torch.cat(tokens,dim=1))[:,0,:])
            if self.piece_sizes:
                base=num.shape[1]-sum(self.piece_sizes)
                pieces=torch.split(num[:,base:],self.piece_sizes,dim=1)
                num=torch.cat([num[:,:base]]+[layer(piece).relu() for layer,piece in zip(self.piece_modules,pieces)],dim=1)
            if self.frequency is not None:
                phase=2*torch.pi*num[:,:,None]*self.frequency[None,:,:]
                num=torch.cat([num,phase.sin().flatten(1),phase.cos().flatten(1)],dim=1)
            x=torch.cat([num]+[m(cat[:,i]) for i,m in enumerate(self.embeddings)],dim=1)
            return self.body(x[:,None,:].expand(-1,heads,-1) if heads>1 else x)
    return Net()


class TorchMLP:
    def __init__(self, columns, categorical, classes=None, encoding='standard', epochs=30, seconds=30, seed=42,
                 width=128,depth=2,heads=1,learning_rate=.001,dual_numeric=False,residual=False,attention=False):
        if epochs<1 or seconds<=0 or min(width,depth,heads)<1:
            raise ValueError('Positive neural epochs, time and architecture dimensions required')
        self.columns,self.categorical,self.classes=columns,categorical,classes
        self.encoding,self.epochs,self.seconds,self.seed=encoding,epochs,seconds,seed
        self.width,self.depth,self.heads,self.learning_rate=width,depth,heads,learning_rate
        self.dual_numeric=dual_numeric
        self.residual=residual
        self.attention=attention

    def fit(self, X, y, evaluation=None):
        # A failed subsequent fit must not expose a previous successful state.
        for field in ('state','epochs_run','best_epoch'):
            if hasattr(self,field):delattr(self,field)
        import torch
        from torch.nn import functional as F
        from sklearn.metrics import f1_score
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        rng=np.random.default_rng(self.seed)
        self.transformer=InputTransform(self.columns,self.categorical,self.encoding,self.dual_numeric).fit(X)
        a,b=self.transformer.transform(X)
        self.shape=(a.shape[1],[len(self.transformer.maps[c])+2 for c in self.transformer.encoded_categorical],len(self.classes) if self.classes else 1)
        self.network_config=dict(width=self.width,depth=self.depth,heads=self.heads,periodic=self.encoding=='periodic',residual=self.residual,
            attention=self.attention,piece_sizes=tuple(len(k)-1 for k in self.transformer.knots if len(k)>1) if self.encoding=='learned_piecewise' else ())
        net=network(*self.shape,**self.network_config)
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
        self.curve=[];self.best_epoch=0;self.epochs_run=0
        for epoch in range(self.epochs):
            net.train()
            for start in range(0,len(X),256):
                if time.monotonic()-started>=self.seconds:
                    raise TimeoutError('Neural fit deadline exhausted before a complete epoch schedule')
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
            if bad>=7:break
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
        net=network(*self.shape,**self.network_config)
        net.load_state_dict({k:torch.from_numpy(v) for k,v in self.state.items()})
        return self._infer(net,X)

