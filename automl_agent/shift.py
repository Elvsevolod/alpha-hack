"""Opt-in transductive hypotheses using features and fit-only labels."""
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score

from .research_evaluation import feature_groups
from .util import digest


def importance_weights(X, future, task, cpu, seed):
    started = time.monotonic()
    columns = task.feature_columns
    all_X = pd.concat([X[columns], future[columns]], ignore_index=True)
    domain = np.r_[np.zeros(len(X)), np.ones(len(future))]
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(all_X[c]) and c not in task.categorical_columns]
    categorical = [c for c in columns if c not in numeric]
    for column in categorical:
        all_X[column] = all_X[column].fillna('__missing__').astype(str)
    groups = feature_groups(all_X)
    if len(np.unique(groups)) < 3:
        raise ValueError('Importance weighting requires at least three exact-feature groups')
    probability = np.full(len(all_X), np.nan)
    for a,b in GroupKFold(3).split(all_X, domain, groups):
        if len(np.unique(domain[a])) < 2:
            raise ValueError('Domain training fold lacks train/test support')
        transformer = ColumnTransformer([('numeric',SimpleImputer(keep_empty_features=True),numeric),
            ('categorical',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),categorical)])
        model = make_pipeline(transformer, RandomForestClassifier(n_estimators=80,max_depth=5,min_samples_leaf=10,n_jobs=cpu,random_state=seed))
        model.fit(all_X.iloc[a],domain[a]);probability[b]=model.predict_proba(all_X.iloc[b])[:,1]
    p=np.clip(probability[:len(X)],.05,.95)
    raw=p/(1-p)*len(X)/max(1,len(future))
    weights=np.clip(raw,.2,5.);weights/=weights.mean()
    effective = float(weights.sum() ** 2 / (weights @ weights))
    if not np.isfinite(weights).all() or (weights <= 0).any() or effective < max(3, .1 * len(X)):
        raise ValueError('Importance weights have insufficient effective sample size or invalid support')
    evidence={'scope':'feature_only_domain_oof','domain_auc':float(roc_auc_score(domain,probability)),
        'clipping':[.2,5.], 'clipped_fraction':float(np.mean((raw<.2)|(raw>5))),
        'effective_rows':effective, 'fit_rows':len(X),
        'normalized_weight_quantiles': np.quantile(weights, [0, .1, .5, .9, 1]).tolist(),
        'test_rows':len(future), 'task_labels_used':False, 'auxiliary_fits':3,
        'repeatable_fit_seconds':time.monotonic()-started,
        'assumption':'covariate_shift_and_support_overlap; P(y|X) invariance is unverified'}
    return pd.Series(weights,index=X.index),evidence


def pseudo_indices(probabilities, X, classes, admitted, cap):
    """Confidence-ranked class-balanced caps, invariant to input file order."""
    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    labels = np.asarray(classes)[probabilities.argmax(axis=1)]
    queues = []
    keys = feature_groups(X)
    for label in classes:
        if label not in admitted:
            continue
        eligible = np.flatnonzero((labels == label) & (confidence >= .95)).tolist()
        queues.append(sorted(eligible, key=lambda i: (-confidence[i], str(keys[i]), str(X.index[i]))))
    selected = []
    for rank in range(max(map(len, queues), default=0)):
        for queue in queues:
            if rank < len(queue) and len(selected) < cap:
                selected.append(queue[rank])
        if len(selected) >= cap:
            break
    return np.asarray(selected, dtype=int)


class ShiftPipeline:
    def __init__(self,spec,task):self.spec,self.task=spec,task

    def fit(self,X,y,context):
        from .pipelines import DirectPipeline
        from .validation import pseudo_label_split
        from .evaluation import score_predictions
        future=context.get('unlabelled_features')
        if future is None or future.empty:
            raise ValueError('Opt-in shift intervention requires staged test features, never test labels')
        started=time.monotonic()
        auxiliary_seconds, auxiliary_fits = 0., 0
        base=self.spec.model_copy(update={'shift_strategy':'none'})
        self.pipeline=DirectPipeline(base,self.task)
        def fit_auxiliary(frame, actual, weights=None, fresh=False):
            nonlocal auxiliary_seconds, auxiliary_fits
            fit_started = time.monotonic()
            recipe = type(base).model_validate(base.research_recipe()) if fresh else base
            model = DirectPipeline(recipe, self.task)
            local = {**context, 'fit_row_keys': frame.index.tolist(),
                'budget_seconds': max(.1, context['budget_seconds'] - (time.monotonic()-started))}
            if fresh:
                for name in ('sample_weights', 'schedule_selection', 'schedule_validation', 'optuna_trial', 'rolling_folds'):
                    local.pop(name, None)
                local['execution_mode'] = 'train'
            if weights is not None:
                local['sample_weights'] = weights
            summary = model.fit(frame, actual, local)
            auxiliary_seconds += time.monotonic()-fit_started
            auxiliary_fits += 1 + summary.auxiliary_fits
            return model
        if self.spec.shift_strategy=='importance_weighting':
            weights,self.evidence=importance_weights(X,future,self.task,context['cpu'],self.spec.seed)
            auxiliary_seconds += self.evidence['repeatable_fit_seconds']
            auxiliary_fits += self.evidence['auxiliary_fits']
            self.evidence['per_class_effective_rows'] = {}
            for target, classes in self.task.classes.items():
                self.evidence['per_class_effective_rows'][target] = {}
                for label in classes:
                    values = weights[y[target] == label].to_numpy()
                    ess = float(values.sum() ** 2 / (values @ values)) if len(values) else 0.
                    self.evidence['per_class_effective_rows'][target][str(label)] = ess
                    if len(values) and ess < .1 * len(values):
                        raise ValueError('Importance weights collapse class support')
            summary=self.pipeline.fit(X,y,{**context,'sample_weights':weights})
        else:
            if self.task.kind!='classification':
                raise ValueError('Verified pseudo-label hypothesis requires single-output classification')
            target=self.task.targets[0]
            # Separate teacher training, pseudo pool, and untouched student gate.
            teacher_rows, pseudo_rows, b = pseudo_label_split(X, y, self.task, context['strategy'], self.spec.seed)
            teacher=fit_auxiliary(X.iloc[teacher_rows],y.iloc[teacher_rows],fresh=True)
            bundle=teacher.predict(X.iloc[pseudo_rows],{**context,'split_id':'pseudo_simulation'})
            probabilities=np.asarray(bundle.raw_probabilities.get(target,bundle.values[target]))
            selected=probabilities.max(axis=1)>=.95
            labels=np.asarray(self.task.classes[target])[probabilities.argmax(axis=1)]
            truth=y.iloc[pseudo_rows][target].to_numpy()
            precision=float(np.mean(labels[selected]==truth[selected])) if selected.any() else 0.
            false_positive_by_class={str(label):int(np.sum(selected&(labels==label)&(truth!=label))) for label in self.task.classes[target]}
            per_class = {str(label): {'support': int(np.sum(selected & (labels==label))),
                'precision': float(np.mean(truth[selected & (labels==label)]==label)) if np.any(selected & (labels==label)) else None}
                for label in self.task.classes[target]}
            admitted_classes = [label for label in self.task.classes[target]
                if per_class[str(label)]['support'] >= 5 and per_class[str(label)]['precision'] >= .95]
            self.evidence={'scope':'fit_inner_transfer_simulation','coverage':float(selected.mean()),'precision':precision,
                'false_positives_by_class':false_positive_by_class,'per_class':per_class, 'admitted_classes':admitted_classes,
                'confidence_threshold':.95,'admitted':precision>=.95 and int(selected.sum())>=10 and bool(admitted_classes)}
            if not self.evidence['admitted']:
                raise ValueError('Pseudo-label transfer simulation failed precision/support gate: '+str(self.evidence))
            selected = pseudo_indices(probabilities, X.iloc[pseudo_rows], self.task.classes[target], admitted_classes,
                max(1, len(teacher_rows)//5))
            if not len(selected):
                raise ValueError('Pseudo-label simulation selected no supported rows')
            augmented = pd.concat([X.iloc[teacher_rows], X.iloc[pseudo_rows[selected]]])
            simulated_y = pd.concat([y.iloc[teacher_rows], pd.DataFrame({target: labels[selected]}, index=X.index[pseudo_rows[selected]])])
            simulated_weights = pd.Series(np.r_[np.ones(len(teacher_rows)), np.full(len(selected), .2)], index=augmented.index)
            student = fit_auxiliary(augmented, simulated_y, simulated_weights, fresh=True)
            teacher_bundle = teacher.predict(X.iloc[b], {**context, 'split_id': 'student_gate'})
            student_bundle = student.predict(X.iloc[b], {**context, 'split_id': 'student_gate'})
            teacher_score = score_predictions(self.task, y.iloc[b], teacher_bundle.values, teacher_bundle.prediction_type)
            student_score = score_predictions(self.task, y.iloc[b], student_bundle.values, student_bundle.prediction_type)
            gain = (student_score-teacher_score) * (1 if self.task.metric.maximize else -1)
            self.evidence.update(student_gate={'teacher_score':teacher_score, 'student_score':student_score,
                'gain':gain, 'fit_rows':len(teacher_rows), 'pseudo_pool_rows':len(pseudo_rows), 'evaluation_rows':len(b),
                'evaluation_labels_used_for_training':False, 'strategy':context['strategy'],
                'target_domain_precision_known':False, 'learned_state_policy':'fresh_inside_gate_training_rows'})
            if gain <= 1e-12:
                raise ValueError('Pseudo-label student did not improve on the untouched fit-inner transfer gate: '+str(self.evidence))
            teacher=fit_auxiliary(X,y)
            bundle=teacher.predict(future,{**context,'split_id':'unlabelled_test'})
            probabilities=np.asarray(bundle.raw_probabilities.get(target,bundle.values[target]))
            selected=pseudo_indices(probabilities, future, self.task.classes[target], admitted_classes, max(1,len(X)//5))
            if not len(selected):
                raise ValueError('No supported high-confidence target-domain pseudo rows')
            pseudo=future.iloc[selected].copy();pseudo.index=['pseudo_'+digest(str(key)) for key in pseudo.index]
            labels=np.asarray(self.task.classes[target])[probabilities[selected].argmax(axis=1)]
            combined=pd.concat([X,pseudo]);targets=pd.concat([y,pd.DataFrame({target:labels},index=pseudo.index)])
            weights=pd.Series(np.r_[np.ones(len(X)),np.full(len(pseudo),.2)],index=combined.index)
            summary=self.pipeline.fit(combined,targets,{**context,'sample_weights':weights})
            summary.rows=len(X)
            self.evidence.update(pseudo_rows=len(pseudo),pseudo_fraction_cap=.2,pseudo_weight=.2)
            self.evidence['pseudo_support'] = {str(label):int(np.sum(labels==label)) for label in self.task.classes[target]}
        summary.preprocessing={**(summary.preprocessing or {}),'shift_intervention':self.evidence}
        summary.auxiliary_fits += auxiliary_fits
        summary.budget = {**(summary.budget or {}),
            'selected_fit_rows':len(X),
            'auxiliary_fit_actual_seconds':(summary.budget or {}).get('auxiliary_fit_actual_seconds',0)+auxiliary_seconds,
            'repeatable_auxiliary_fit_seconds':auxiliary_seconds, 'repeatable_auxiliary_fits':auxiliary_fits,
            'wrapper_policy':'teacher/domain and student gate repeat on every permitted fit population'}
        summary.seconds=time.monotonic()-started
        self.summary=summary
        return summary

    def predict(self,X,context):return self.pipeline.predict(X,context)
    def frozen_spec(self):
        frozen=self.pipeline.frozen_spec()
        return frozen.model_copy(update={'shift_strategy':self.spec.shift_strategy})
