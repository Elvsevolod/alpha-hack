"""Tie JSON class-weight keys to confirmed labels, never to class positions."""
from decimal import Decimal,InvalidOperation
from numbers import Real
import math


def resolve_class_weight(spec,task,target,observed=None):
    labels=task.classes.get(target,[])
    if not labels:
        raise ValueError('class_weight requires confirmed classification labels')
    weights=spec.hyperparameters['class_weight']
    if isinstance(weights,list):
        if (spec.model not in {'random_forest','extra_trees'}
                or task.kind not in {'multioutput_classification','multilabel'}
                or len(weights)!=len(task.targets) or any(not isinstance(w,dict) for w in weights)):
            raise ValueError('class_weight list requires one dictionary per forest output')
        weights=weights[task.targets.index(target)]
    if weights is None:return None
    if isinstance(weights,str):
        allowed={'balanced','balanced_subsample'} if spec.model in {'random_forest','extra_trees'} else {'balanced'}
        if weights not in allowed:raise ValueError('Unsupported class_weight preset for this model')
        return weights
    if not isinstance(weights,dict):
        raise ValueError('class_weight must be a label dictionary, balanced preset or None')
    result={}
    for key,weight in weights.items():
        # Numeric labels survive JSON only as string keys. Decimal avoids
        # collapsing distinct large integer labels through float rounding.
        number=None
        if isinstance(key,(str,Real)) and not isinstance(key,bool):
            try:
                candidate=Decimal(str(key))
                if candidate.is_finite():number=candidate
            except InvalidOperation:pass
        matches=[label for label in labels if
            (isinstance(label,str) and isinstance(key,str) and key==label)
            or (isinstance(label,Real) and number is not None and Decimal(str(label))==number)]
        if len(matches)!=1 or matches[0] in result:
            raise ValueError('Invalid or ambiguous class_weight label mapping; use confirmed TaskSpec classes, not positional indices')
        try:
            valid=not isinstance(weight,bool) and isinstance(weight,Real) and math.isfinite(float(weight)) and weight>=0
        except OverflowError:valid=False
        if not valid:
            raise ValueError('class_weight values must be finite nonnegative numbers')
        result[matches[0]]=weight
    present=labels if observed is None else set(observed)
    if not any(result.get(label,1)>0 for label in present):
        raise ValueError('class_weight must leave positive weight on an observed class')
    # A declared rare class may be absent from this particular inner fit.
    # Its weight has no samples to act on and must not reach a local class map.
    return {label:weight for label,weight in result.items() if label in present}


def resolve_catboost_class_weights(spec,task,target,observed=None):
    """CatBoost infers its class names from dictionary keys, including their types."""
    weights=spec.hyperparameters['class_weights']
    if not task.classes.get(target):
        raise ValueError('class_weights requires confirmed classification labels')
    if weights is not None and spec.hyperparameters.get('auto_class_weights') not in (None,'None'):
        raise ValueError('class_weights cannot be combined with auto_class_weights')
    if not isinstance(weights,dict):
        # Native positional lists retain their existing CatBoost semantics.
        return weights
    surrogate=spec.model_copy(update={'hyperparameters':{'class_weight':weights}})
    try:
        resolved=resolve_class_weight(surrogate,task,target)
        if set(resolved)!=set(task.classes[target]):
            raise ValueError('class_weight dictionary must cover every confirmed class label')
        return resolve_class_weight(surrogate,task,target,observed)
    except ValueError as error:
        # The shared resolver emits only fixed public diagnostics, never keys.
        raise ValueError(str(error).replace('class_weight','class_weights')) from None
