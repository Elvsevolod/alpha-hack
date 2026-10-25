"""Verified narrow neural route; the common executor owns process/RSS limits."""
from __future__ import annotations
import math
import time
import pandas as pd

from .api import prediction_bundle
from .contracts import TrainingSummary
from .preprocessing import selected_features, FrameTransform
from .torch_tabular import TorchMLP
from .validation import inner_split

PARAMETERS = {'encoding', 'width', 'depth', 'heads', 'epochs', 'learning_rate', 'dual_numeric', 'residual', 'attention'}


def validate(spec, task):
    p = spec.hyperparameters
    if set(p) - PARAMETERS:
        raise ValueError('Unsupported torch_tabular parameters: ' + str(sorted(set(p)-PARAMETERS)))
    for key, default, low, high in [('width',128,8,512),('depth',2,1,4),('heads',1,1,8),('epochs',50,1,200)]:
        value=p.get(key,default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError('Invalid bounded neural parameter: '+key)
    lr=p.get('learning_rate',.001)
    if type(lr) not in (int,float) or not math.isfinite(lr) or not 1e-5 <= lr <= .05:
        raise ValueError('Invalid neural learning_rate')
    if p.get('encoding','quantile') not in {'standard','quantile','piecewise','learned_piecewise','periodic'}:
        raise ValueError('Invalid neural encoding')
    if type(p.get('attention',False)) is not bool or type(p.get('residual',False)) is not bool or type(p.get('dual_numeric',False)) is not bool:
        raise ValueError('attention/residual/dual_numeric must be boolean')
    if p.get('residual',False) and p.get('heads',1)!=1:
        raise ValueError('Residual shared-weight combination is unverified')
    if task.metric.name != ('f1' if task.classes else 'rmse') or task.classes and task.metric.average!='macro':
        raise ValueError('torch_tabular epoch selection is verified for RMSE or Macro F1 only')
    if p.get('attention',False) and (p.get('heads',1)!=1 or p.get('residual',False) or p.get('encoding','quantile')!='quantile' or set(p)&{'width','depth'}):
        raise ValueError('Attention profile uses fixed width 32, one layer, one output head and quantile encoding')
    if (spec.engine != 'direct' or task.kind not in {'regression','classification'} or len(task.targets)!=1
            or task.text_columns or spec.representation not in {'tabular','native'}
            or any(op.name not in {'math','datetime','log1p','aggregate'} for op in spec.transformations) or spec.resampling!='none' or spec.shift_strategy!='none'
            or spec.prediction_policy!='default' or spec.postprocess_fit_policy!='none'
            or spec.fit_diagnostics!='none' or spec.target_transform!='none' or spec.log_inverse!='naive'
            or spec.training_objective or spec.early_stopping_rounds or spec.checkpoint_counts
            or spec.refit_iteration_policy!='fixed' or spec.class_offsets is not None or spec.learned_decision
            or spec.learned_inverse or spec.imputation.model_dump()!=type(spec.imputation)().model_dump()):
        raise ValueError('torch_tabular supports single-target tabular default preprocessing/loss/output only')
    if any(type(v) is not int or not 1<=v<=200 for v in spec.iteration_schedule.values()) or (spec.frozen_iterations is not None and spec.frozen_iterations>200):
        raise ValueError('Invalid frozen neural epochs')
    if set(spec.iteration_schedule)-set(task.targets):
        raise ValueError('Neural epoch schedule contains an unknown target')


class TorchPipeline:
    def __init__(self, spec, task):
        self.spec,self.task=spec,task

    def fit(self,X,y,context):
        from .compatibility import require_compatible
        require_compatible(self.spec,self.task,context.get('strategy'))
        if context.get('sample_weights') is not None:
            raise ValueError('Neural domain sample weighting is unverified')
        for field in ('model','epochs','summary'):
            if hasattr(self,field):delattr(self,field)
        started=time.monotonic();target=self.task.targets[0]
        columns=selected_features(self.spec,self.task)
        original_columns=columns
        self.frame_transform=FrameTransform(self.spec.transformations)
        self.frame_transform.fit(X[columns])
        transformed=self.frame_transform.transform(X[columns])
        columns=list(transformed)
        categorical=[c for c in self.task.categorical_columns if c in columns]
        categorical += [c for c in columns if c not in original_columns and not pd.api.types.is_numeric_dtype(transformed[c])]
        classes=self.task.classes.get(target)
        params=dict(self.spec.hyperparameters)
        params.setdefault('encoding','quantile')
        params.setdefault('learning_rate',.002)
        epochs=self.spec.iteration_schedule.get(target,self.spec.frozen_iterations)
        auxiliary=0;auxiliary_seconds=0.;selector_startup=0.;selection=None
        if context.get('execution_mode')=='refit' and not epochs:
            raise ValueError('Neural refit requires frozen epochs')
        def model(epoch_count,seconds):
            return TorchMLP(columns,categorical,classes,seconds=seconds,seed=self.spec.seed,
                **{**params,'epochs':epoch_count})
        budget=float(context['budget_seconds'])
        if not epochs:
            # Exact-X boundaries also apply to a low-overlap outer protocol.
            strategy=context.get('inner_strategy',context['strategy'])
            if context.get('evaluation_mode') in {'exact_group','low_overlap'}:
                strategy='exact_group'
            a,b=inner_split(X,y,self.task,strategy,self.spec.seed)
            selector=model(params.get('epochs',50),max(.01,budget*.45))
            selector.fit(transformed.iloc[a],y.iloc[a][target],(transformed.iloc[b],y.iloc[b][target]))
            selector_startup=selector.startup_seconds
            epochs=selector.best_epoch;auxiliary=1;auxiliary_seconds=time.monotonic()-started
            selection={'strategy':strategy,'fit_rows':len(a),'stop_rows':len(b),'selected_epochs':epochs,
                'actual_selection_epochs':selector.epochs_run,'curve':selector.curve,'scope':'strictly_inside_current_fit'}
        remaining=budget-(time.monotonic()-started)
        if remaining<=.01:raise TimeoutError('Neural selection exhausted refit budget')
        self.model=model(epochs,remaining)
        self.original_columns=original_columns
        self.model.fit(transformed,y[target])
        if self.model.epochs_run!=epochs:
            raise TimeoutError('Incomplete neural frozen epoch refit')
        self.epochs=epochs
        self.summary=TrainingSummary(rows=len(X),seconds=time.monotonic()-started,auxiliary_fits=auxiliary,
            selected_parameters={'model':'torch_tabular','hyperparameters':params,'epochs':epochs},
            learning_curve={target:selection} if selection else None,
            imputation={'numeric':'median','categorical':'missing_category','indicators':True,'training_scope':'fit_only_no_target','auxiliary_fits':0},
            preprocessing={'fit_only':True,'numeric':params.get('encoding','quantile'),'stateless_frame_transformations':[op.name for op in self.spec.transformations],'categorical':'learned_embeddings',
                'unknown_and_missing_distinct':True,'columns':columns},
            budget={'selected_fit_actual_seconds':max(0.,time.monotonic()-started-auxiliary_seconds-self.model.startup_seconds),
                'selected_fit_rows':len(X),'auxiliary_fit_actual_seconds':max(0.,auxiliary_seconds-selector_startup),
                'initialization_actual_seconds':selector_startup+self.model.startup_seconds,
                'worker_initialization_seconds':max(selector_startup,self.model.startup_seconds),
                'iteration_schedule':{target:epochs},'complete_epoch_refit':True,'cpu_threads':1,
                'independent_oof_auxiliary_fit_seconds':max(0.,auxiliary_seconds-selector_startup)})
        return self.summary

    def predict(self,X,context):
        p=self.model.predict(self.frame_transform.transform(X[self.original_columns]))
        return prediction_bundle({self.task.targets[0]:p},X,context,classes=self.task.classes)

    def frozen_spec(self):
        return self.spec.model_copy(update={'iteration_schedule':{self.task.targets[0]:self.epochs}})
