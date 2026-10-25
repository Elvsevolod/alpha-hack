"""Optional shrinkage of group residuals learned from honest fit-only forecasts.

The caller must supply out-of-fit predictions, never in-sample residuals or
protected evaluation labels. No class, dataset or time-column names are fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ResidualCalibration:
    def __init__(self,categorical,shrinkage=100,strength=.5):
        self.categorical=list(categorical)
        self.shrinkage=shrinkage
        self.strength=strength

    def fit(self,X,y,prediction):
        from .group_context import GroupContext
        actual=np.asarray(y);predicted=np.asarray(prediction)
        if actual.ndim!=1 or predicted.shape!=actual.shape:
            raise ValueError('Aligned one-dimensional out-of-fit predictions required')
        error=actual-predicted
        if not len(error) or len(X)!=len(error) or error.ndim!=1 or not np.isfinite(error).all():
            raise ValueError('Finite aligned out-of-fit residuals required')
        if self.shrinkage<=0 or not 0<=self.strength<=1:raise ValueError('Invalid calibration bounds')
        self.global_=float(error.mean())
        self.tables_={}
        for c in self.categorical:
            keys=GroupContext.keys(X[c])
            table=pd.DataFrame({'key':keys.to_numpy(),'error':error}).groupby('key').error.agg(['mean','count'])
            self.tables_[c]=(table['mean']-self.global_)*table['count']/(table['count']+self.shrinkage)
        return self

    def predict(self,X,prediction):
        from .group_context import GroupContext
        prediction=np.asarray(prediction)
        if prediction.shape!=(len(X),) or not np.isfinite(prediction).all():
            raise ValueError('Finite aligned query predictions required')
        adjustments=[GroupContext.keys(X[c]).map(table).fillna(0).to_numpy() for c,table in self.tables_.items()]
        group=np.mean(adjustments,axis=0) if adjustments else np.zeros(len(X))
        return prediction+self.strength*(self.global_+group)
