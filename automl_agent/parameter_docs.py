"""Read parameter sections from verified, frozen documentation; never import/fit an API."""
import re
import ast

from .util import digest

VERSION = 1


def parameter_section(record, name=None):
    text = record['text']
    heading = re.search(r'(?m)^Parameters\s*\n[-=]{3,}\s*\n', text)
    if not heading:
        return signature_parameters(record,name)
    start = heading.end()
    following = re.search(r'(?m)^[A-Za-z][A-Za-z ]+\n[-=]{3,}\s*(?:\n|$)', text[start:])
    end = start + following.start() if following else len(text)
    declarations = list(re.finditer(r'(?m)^([A-Za-z_]\w*(?:,\s*[A-Za-z_]\w*)*)\s*:\s*[^\n]+', text[start:end]))
    names = [m.group(1) for m in declarations]
    if name:
        found = next((i for i,m in enumerate(declarations) if name in [p.strip() for p in m.group(1).split(',')]), None)
        if found is None:
            return {'status':'unknown_parameter','available_names':names}
        original = start
        start = original + declarations[found].start()
        end = original + declarations[found+1].start() if found+1 < len(declarations) else end
    signature = next((line for line in text[:heading.start()].splitlines()
        if re.match(r'^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\(', line)), None)
    return {'status':'available','start':start,'end':end,'available_names':names,'signature':signature,'detail_status':'described'}


def signature_parameters(record, name=None):
    """Some frozen class docs contain only a signature. Parse it as data, never execute."""
    match = re.search(r'(?m)^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\([^\n]+',record['text'])
    if not match:
        return None
    signature = match.group(0)
    try:
        tree = ast.parse('def _'+signature[signature.index('('):]+':\n    pass\n')
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
            return None
        arguments = tree.body[0].args
    except SyntaxError:
        return None
    positional = [*arguments.posonlyargs,*arguments.args]
    defaults = [None]*(len(positional)-len(arguments.defaults))+arguments.defaults
    rows=[]
    for argument, default in [*zip(positional,defaults),*zip(arguments.kwonlyargs,arguments.kw_defaults)]:
        rows.append({'name':argument.arg,'has_default':default is not None,
            'default_text':ast.unparse(default) if default is not None else None,
            'annotation_text':ast.unparse(argument.annotation) if argument.annotation else None})
    names=[row['name'] for row in rows]
    if name and name not in names:
        return {'status':'unknown_parameter','available_names':names}
    return {'status':'available','start':match.start(),'end':match.end(),'available_names':names,
        'signature':signature,'detail_status':'signature_only',
        'declarations':[row for row in rows if name is None or row['name']==name],
        'forwards_kwargs':arguments.kwarg is not None}


def adapter_contract(record, names, eda=False):
    """Library documentation is broader than the ready-adapter contract."""
    from .pipelines import PARAMETERS, CONTROLLED_PARAMETERS
    symbol = record.get('title','').strip()
    short = symbol.rsplit('.',1)[-1]
    if not short or short[0].islower():  # fit/predict arguments are not hyperparameters
        return None
    stem = re.sub(r'(Classifier|Regressor)$','',short).casefold()
    alias = {'logisticregression':'logistic','lgbm':'lightgbm','xgb':'xgboost','lgbmranker':'lgbm_ranker'}
    model = alias.get(stem) or next((m for m in PARAMETERS if m.replace('_','')==stem), None)
    if model is None:
        return None
    controlled = set(CONTROLLED_PARAMETERS.get(model,set()))
    allowed = PARAMETERS[model] - controlled
    mode_allowed = True
    if eda:
        from .eda.policy import DIRECT_MODELS, RESOURCE_PARAMETERS
        controlled |= RESOURCE_PARAMETERS
        allowed -= RESOURCE_PARAMETERS
        mode_allowed = model in DIRECT_MODELS
    return {'model':model,'mode_allowed':mode_allowed,'allowed_hyperparameters':sorted(allowed),
        'controller_controlled_parameters':sorted(set(names)&controlled),
        'library_only_parameters':sorted(set(names)-allowed-controlled),
        'limitation':'Library defaults/descriptions are documentation, not recommended values or permission to bypass the adapter. Training compatibility still requires verification.'}
