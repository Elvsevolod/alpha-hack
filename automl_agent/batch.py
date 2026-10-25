"""One model decision, several independently journaled bounded fits."""
import time
from .contracts import TrainArgs, ToolCommand
from .compatibility import require_compatible
from .tool_results import payload
from .util import digest


def execute_batch(c, args, operation_id):
    candidates = [TrainArgs.model_validate(item) for item in args.candidates]
    resolved = []
    for original in candidates:
        candidate = original.model_copy(deep=True)
        if candidate.apply_recipe:
            from .recipe_patch import resolve
            candidate.pipeline, _ = resolve(c, candidate.apply_recipe)
            original.pipeline = candidate.pipeline.model_copy(deep=True)
        from .recipe_patch import inherit_parent
        inherit_parent(c, candidate)
        if candidate.pipeline.engine not in {'direct', 'optuna', 'flaml', 'autogluon', 'lightautoml'}:
            raise ValueError('Batch requires built-in independent training candidates')
        require_compatible(candidate.pipeline, c.task, c.plan.strategy, c.capabilities)
        c.validate_evidence(candidate.hypothesis.evidence_refs)
        resolved.append(candidate)
    if len({candidate.pipeline.hash for candidate in resolved}) != len(candidates):
        raise ValueError('Batch hypotheses must have different pipeline identities')
    started = time.monotonic()
    allocation = min(args.budget_seconds, max(0., c.budget.remaining - c.budget.final_reserve(c.finalist_costs())))
    shared_deadline = time.time() + allocation
    from .admission import admit
    for index, candidate in enumerate(candidates):
        candidate.budget_seconds = min(candidate.budget_seconds, allocation / len(candidates))
        child_id = 'batch_' + digest({'parent': operation_id, 'index': index})[:24]
        if candidate.pipeline.engine == 'direct':
            admit(c, ToolCommand(operation_id=child_id, tool='train_model', arguments=candidate.model_dump(mode='json', exclude_unset=True)), shared_deadline)
    results = []
    for index, candidate in enumerate(candidates):
        available = allocation - (time.monotonic() - started)
        if available < .2 or c.state != 'searching':
            results.append({'index': index, 'status': 'not_started_shared_budget'})
            continue
        candidate.budget_seconds = min(candidate.budget_seconds, available / (len(candidates)-index))
        child_id = 'batch_' + digest({'parent': operation_id, 'index': index})[:24]
        tool = 'train_model' if candidate.pipeline.engine == 'direct' else 'tune_model' if candidate.pipeline.engine == 'optuna' else 'run_automl'
        response = payload(c.dispatch(ToolCommand(operation_id=child_id, tool=tool, arguments=candidate.model_dump(mode='json', exclude_unset=True))))
        measured = response.get('result') or {}
        experiment = response.get('experiment') or {}
        results.append({'index': index, 'operation_id': child_id, 'experiment_id': experiment.get('experiment_id'),
            'status': measured.get('status', response.get('status')), 'score': measured.get('score'),
            'seconds': measured.get('seconds'), 'error': response.get('error')})
        c.journal.put('batch_' + operation_id, {'results': results, 'shared_budget_seconds': allocation})
    return {'status': 'success', 'results': results, 'seconds': time.monotonic()-started,
        'shared_budget_seconds': allocation, 'execution': 'sequential_one_worker', 'retry_policy': 'reuse_child_operation_ids'}
