"""Fit-only OOF blend fitting; external validation remains the quality estimate."""
import numpy as np
import time
from sklearn.linear_model import LogisticRegression, Ridge

from .evaluation import score_predictions
from .util import digest, read_json


class OOFCacheError(ValueError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(detail['error'])


def cached_oof(c, identifier, spec, rows, directory):
    """A previous attempt never opens a second fit, including corrupt results."""
    from .util import file_hash, redact, utf8_short
    manifest_path, request_path = directory / 'worker_result.json', directory / 'request.json'
    if not manifest_path.is_file() and not request_path.is_file():
        return None
    identity = {'member_id': identifier, 'identity': directory.name, 'pipeline_hash': spec.pipeline.hash,
        'data_hash': c.dataset.source_hash, 'protocol_hash': c.plan.hash, 'task_hash': c.task.hash,
        'rows_hash': digest(rows), 'library_versions': spec.library_versions}
    unknown = {**identity, 'status': 'outcome_unknown', 'error_code': 'oof_outcome_unknown',
        'error': 'OOF operation outcome unknown; inspect the existing attempt before retrying'}
    try:
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        request = read_json(request_path)
        context = request['context']
        expected_keys = [c.dataset.row_keys[i] for i in rows]
        valid = (request['mode'] == 'oof' and digest(request['pipeline']) == spec.pipeline.hash
            and digest(request['task']) == c.task.hash and context['data_hash'] == c.dataset.source_hash
            and context['protocol_hash'] == c.plan.hash and context['oof_seed'] == 42
            and context['library_versions'] == spec.library_versions
            and read_json(directory / 'fit_X.json')['index'] == expected_keys
            and read_json(directory / 'fit_y.json')['index'] == expected_keys)
        if manifest.get('status') == 'completed':
            path = directory / 'oof_predictions.json'
            if (not valid or manifest.get('request_hash',digest(request)) != digest(request)
                    or file_hash(path) != manifest['hashes'][path.name]):
                raise ValueError('OOF cached identity or artifact changed')
            return read_json(path)
        from pathlib import Path
        terminal_status = manifest.get('status')
        terminal = (valid and terminal_status in {'error', 'import_error', 'memory_limit', 'timeout', 'insufficient_pilot_support'}
            and manifest.get('diagnostic_code') == terminal_status
            and isinstance(manifest.get('error'), str) and bool(manifest['error'])
            and ('request_hash' not in manifest or manifest['request_hash'] == digest(request))
            and Path(manifest.get('diagnostic_artifact', '')).resolve() == manifest_path.resolve())
    except (OSError, ValueError, KeyError, TypeError):
        raise OOFCacheError(unknown) from None
    if not terminal:
        raise OOFCacheError(unknown)
    detail = {**identity, 'status': 'technical_failure', 'error_code': 'oof_known_terminal_error',
        'terminal_status': terminal_status, 'error': utf8_short(redact(manifest['error']), 700),
        'terminal_identity_basis': 'request_digest' if 'request_hash' in manifest else 'validated_request_and_owned_artifact',
        'diagnostic_ref': str(manifest_path.relative_to(c.workdir))}
    # Publish only the bounded worker error and identities, never raw traceback,
    # staged targets or partial predictions. The exact read route stays usable.
    if getattr(c, 'context_memory', None) is not None:
        from context_compressor import Source
        from .util import canonical
        source = Source('oof_error:' + directory.name, digest(detail), canonical(detail),
            'registered_feedback', 'external_data', origin='oof_terminal_error')
        c.context_memory.store.put_source(source)
        detail['diagnostic_read'] = c.context_memory.store.route(source.reference)
    raise OOFCacheError(detail)


def fit_blend(task, reports, y, strategy, seed):
    if strategy not in {"weighted_oof", "stacked_oof"}:
        raise ValueError("Learned blend requires an explicit supported strategy")
    if any(len(report.get('folds', [])) < 2 for report in reports):
        raise ValueError('insufficient_oof_support: learned weights/meta-models require at least two fitted folds')
    if any(report.get("prediction_role", "strict_oof") != "strict_oof" for report in reports):
        raise ValueError("Meta-training requires strict OOF, never search validation predictions")
    if len({report['folds_hash'] for report in reports}) != 1:
        raise ValueError('OOF blend members must use identical external folds')
    key_sets = [set(report['row_keys']) for report in reports]
    common = key_sets[0].intersection(*key_sets[1:]) if len(key_sets) > 1 else key_sets[0]
    keys = [key for key in reports[0]['row_keys'] if key in common]
    if len(keys) < 10:
        raise ValueError('Insufficient common OOF rows for a regularized blend')
    values = []
    for report in reports:
        positions = {key: i for i, key in enumerate(report['row_keys'])}
        if report['classes'] != task.classes:
            raise ValueError('OOF class order differs from the contract')
        values.append({target: np.asarray(report['values'][target])[[positions[key] for key in keys]] for target in task.targets})
    actual = y.loc[keys]
    if strategy == 'weighted_oof':
        count = len(reports)
        rng = np.random.default_rng(seed)
        candidates = [np.ones(count)/count, *np.eye(count), *rng.dirichlet(np.ones(count)*2, size=64)]
        def objective(weights):
            prediction = {target: np.average([member[target] for member in values], axis=0, weights=weights) for target in task.targets}
            value = score_predictions(task, actual, prediction, 'probabilities' if task.classes else 'values')
            return (-value if task.metric.maximize else value) + .001 * float(np.sum((weights-1/count)**2))
        weights = min(candidates, key=objective).tolist()
        meta = {}
    else:
        weights, meta = None, {}
        for target in task.targets:
            matrix = np.column_stack([np.asarray(value[target]).reshape(len(keys), -1) for value in values])
            model = LogisticRegression(C=.1, max_iter=500, random_state=seed) if task.classes else Ridge(alpha=10.)
            model.fit(matrix, actual[target])
            meta[target] = {'coef': model.coef_.tolist(), 'intercept': np.asarray(model.intercept_).tolist(),
                'classes': model.classes_.tolist() if task.classes else []}
    evidence = {'scope': 'fit_only_oof_meta_training', 'row_keys_hash': digest(keys), 'rows': len(keys),
        'folds_hash': reports[0]['folds_hash'], 'unbiased_meta_score_claimed': False,
        'quality_measurement': 'untouched_research_validation_and_closed_control'}
    return weights, meta, evidence


def fit_final_layer(task, reports, y, spec):
    from .decision import fit_decision
    if task.kind != "classification" or task.output.prediction_type != "labels":
        raise ValueError("Final ensemble offsets require single-output label classification")
    if any(len(report.get('folds', [])) < 2 for report in reports):
        raise ValueError('insufficient_oof_support: learned final decision requires at least two fitted folds')
    key_sets = [set(report["row_keys"]) for report in reports]
    common = key_sets[0].intersection(*key_sets[1:]) if len(key_sets) > 1 else key_sets[0]
    keys = [key for key in reports[0]["row_keys"] if key in common]
    if len({report["folds_hash"] for report in reports}) != 1 or any(report["classes"] != task.classes for report in reports):
        raise ValueError("Final decision requires common OOF folds and class order")
    values = []
    for report in reports:
        index = {key: i for i, key in enumerate(report["row_keys"])}
        values.append({target: np.asarray(report["values"][target])[[index[key] for key in keys]] for target in task.targets})
    combined = combine(task, values, spec)
    target = task.targets[0]
    state = fit_decision(task, y.loc[keys], combined[target], keys, reports[0]["folds_hash"])
    state["scope"] = "fit_only_oof_joint_composition_training"
    return spec.model_copy(update={"learned_decision": {target: state}})


def combine(task, members, spec):
    result = {}
    for target in task.targets:
        if target in spec.ensemble_meta:
            meta = spec.ensemble_meta[target]
            matrix = np.column_stack([np.asarray(value[target]).reshape(len(value[target]), -1) for value in members])
            score = matrix @ np.asarray(meta['coef']).T + np.asarray(meta['intercept'])
            if task.classes:
                if score.shape[1] == 1:
                    p = 1/(1+np.exp(-np.clip(score[:, 0], -700, 700)))
                    probabilities = np.column_stack([1-p, p])
                else:
                    score -= score.max(axis=1, keepdims=True)
                    probabilities = np.exp(score); probabilities /= probabilities.sum(axis=1, keepdims=True)
                value = np.zeros((len(matrix), len(task.classes[target])))
                for i, label in enumerate(meta['classes']):
                    value[:, task.classes[target].index(label)] = probabilities[:, i]
                result[target] = value
            else:
                result[target] = score.reshape(-1)
        else:
            result[target] = np.average([member[target] for member in members], axis=0, weights=spec.ensemble_weights)
    return result


def oof_directory(c, spec, rows, phase="research"):
    identity = digest({'pipeline': spec.pipeline.hash, 'data': c.dataset.source_hash, 'protocol': c.plan.hash,
        'task': c.task.hash, 'rows': rows, 'fold_seed': 42, 'phase': phase,
        'schema': c.task.feature_columns, 'versions': spec.library_versions, 'artifact_kind': 'strict_oof_probabilities_v3'})
    return c.workdir / 'oof' / identity


def oof_forecast(c, identifiers, rows=None):
    """Count every fresh fold fit; cached predictions need no new training."""
    from .research_evaluation import oof_plan
    from .completion import candidate_measured_cost
    purpose = 'research' if rows is None else 'stability'
    rows = c.plan.fit if rows is None else rows
    X = c.dataset.features(rows)
    y = c.dataset.train.iloc[rows][c.task.targets]
    strategy = 'exact_group' if c.plan.evaluation_mode in {'exact_group', 'low_overlap'} else c.plan.strategy
    fold_counts = {}
    costs, cached = {}, []
    for identifier in identifiers:
        spec, result = c.lookup(identifier)
        directory = oof_directory(c, spec, rows, purpose)
        report = cached_oof(c, identifier, spec, rows, directory)
        if report is not None:
            if len(report.get('folds', [])) < 2:
                raise ValueError('insufficient_oof_support: cached learned composition requires at least two fitted folds')
            fold_counts[identifier] = len(report.get('folds', []))
            cached.append(identifier)
            costs[identifier] = 0.
            continue
        parts, _ = oof_plan(spec.pipeline, c.task, X, y, strategy, 42)
        if len(parts) < 2:
            raise ValueError('insufficient_oof_support: learned composition requires at least two fitted folds')
        fold_counts[identifier] = len(parts)
        if spec.pipeline.engine == 'ensemble' or spec.pipeline.engine == 'custom' and not spec.pipeline.custom_oof:
            raise ValueError('OOF members require built-in independent pipelines')
        costs[identifier] = sum(candidate_measured_cost(result.training_summary, result.seconds,
            len(a), len(c.plan.fit), spec.pipeline) for a, _, _, _ in parts)
        if spec.pipeline.model=='torch_tabular':
            auxiliary=float((result.training_summary.get('budget') or {}).get('independent_oof_auxiliary_fit_seconds',0))
            costs[identifier]+=sum(auxiliary*len(a)/max(1,len(c.plan.fit)) for a,_,_,_ in parts)
    total = sum(costs.values())
    return {'seconds': total + max(1., total * .05), 'member_seconds': costs, 'folds': max(fold_counts.values(), default=0),
        'member_fold_counts': fold_counts,
        'cached_members': cached, 'basis': 'measured_all_fold_fits_plus_meta_overhead', 'estimate_is_guarantee': False}


def member_oof(controller, identifier, seconds, rows=None):
    c = controller
    purpose = 'research' if rows is None else 'stability'
    rows = c.plan.fit if rows is None else rows
    spec, result = c.lookup(identifier)
    if spec.pipeline.engine == 'ensemble' or spec.pipeline.engine == 'custom' and not spec.pipeline.custom_oof:
        raise ValueError('OOF members require a built-in pipeline with independently rebuilt preprocessing')
    directory = oof_directory(c, spec, rows, purpose)
    cached = cached_oof(c, identifier, spec, rows, directory)
    if cached is not None:
        return cached
    path = directory / 'oof_predictions.json'
    request = c.stage(directory, spec, rows, {}, mode='oof')
    request['context']['oof_seed'] = 42
    request['context']['minimum_oof_folds'] = 2
    started = time.monotonic()
    outcome = c.executor.run(directory, request, seconds)
    c.journal.event('oof_member_execution', {'member_id': identifier, 'purpose': purpose,
        'seconds': time.monotonic() - started, 'status': outcome['status']})
    if outcome['status'] != 'completed':
        # A verified terminal manifest is immediately available on first error,
        # as well as on any later cache read with a new outer operation ID.
        cached_oof(c, identifier, spec, rows, directory)
        raise ValueError('OOF member did not complete: ' + outcome['status'])
    return read_json(path)
