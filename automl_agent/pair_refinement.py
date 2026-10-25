"""Optional bounded specialists for class pairs selected on fit calibration.

This is a mass-preserving conditional correction, not full pairwise coupling.
The caller owns split isolation and supplies probabilities in declared order.
"""
from __future__ import annotations

import numpy as np


class PairRefiner:
    def __init__(self,max_pairs=3,min_train_support=30,min_calibration_support=5,strength=.5):
        self.max_pairs=max_pairs
        self.min_train_support=min_train_support
        self.min_calibration_support=min_calibration_support
        self.strength=strength

    def fit(self,X,y,calibration_y,calibration_probability):
        from .contracts import PipelineSpec
        from .pipelines import make_estimator
        if self.max_pairs<1 or not 0<=self.strength<=1:
            raise ValueError('Invalid pair-refinement bounds')
        y=np.asarray(y,dtype=int);cy=np.asarray(calibration_y,dtype=int)
        cp=np.asarray(calibration_probability,dtype=float)
        if cp.ndim!=2 or len(cy)!=len(cp) or not len(cy):raise ValueError('Missing calibration observations')
        self.class_count_=cp.shape[1]
        if min(y.min(),cy.min())<0 or max(y.max(),cy.max())>=self.class_count_:
            raise ValueError('Undeclared class index')
        if not np.isfinite(cp).all() or (cp<0).any() or not np.allclose(cp.sum(axis=1),1):
            raise ValueError('Invalid calibration probabilities')
        counts=np.bincount(y,minlength=self.class_count_)
        support=np.bincount(cy,minlength=self.class_count_)
        predicted=cp.argmax(axis=1)
        matrix=np.bincount(cy*self.class_count_+predicted,minlength=self.class_count_**2).reshape(self.class_count_,self.class_count_)
        candidates=[]
        for a in range(self.class_count_):
            for b in range(a+1,self.class_count_):
                if min(counts[a],counts[b])<self.min_train_support or min(support[a],support[b])<self.min_calibration_support:continue
                confusion=matrix[a,b]/support[a]+matrix[b,a]/support[b]
                if confusion>0:candidates.append((float(confusion),a,b))
        candidates.sort(key=lambda z:(-z[0],z[1],z[2]))
        self.pairs_=[]
        for confusion,a,b in candidates[:self.max_pairs]:
            rows=np.flatnonzero((y==a)|(y==b))
            spec=PipelineSpec(model='lightgbm',seed=42,hyperparameters=dict(n_estimators=250,
                num_leaves=15,learning_rate=.04,reg_lambda=10,min_child_samples=20))
            model=make_estimator(spec,True,{'cpu':2})
            model.fit(X.iloc[rows] if hasattr(X,'iloc') else X[rows],(y[rows]==b).astype(int))
            self.pairs_.append(dict(a=a,b=b,model=model,confusion=confusion,
                factors=np.sqrt(counts.max()/counts[[a,b]])))
        return self

    def predict_proba(self,X,base_probability):
        p=np.array(base_probability,dtype=float,copy=True)
        if p.shape!=(len(X),self.class_count_) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):
            raise ValueError('Invalid base probabilities or class order')
        for pair in self.pairs_:
            a,b=pair['a'],pair['b'];q=pair['model'].predict_proba(X)[:,1]
            f=pair['factors'];q=q*f[1]/((1-q)*f[0]+q*f[1])
            mass=p[:,a]+p[:,b]
            p[:,a]=(1-self.strength)*p[:,a]+self.strength*mass*(1-q)
            p[:,b]=(1-self.strength)*p[:,b]+self.strength*mass*q
        return p
