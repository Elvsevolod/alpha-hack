"""Budget-aware decision context; facts and restrictions, never dataset recipes."""
from ..util import digest


def action_window(generations, tools):
    """Reserve the final available generation for a model-written conclusion."""
    closing = generations <= 1
    return {
        'phase': 'conclude' if closing else 'investigate',
        'analysis_generations_remaining': max(0, generations - 1),
        'final_generation_reserved': True,
        'instruction': ('Call finish now using measured facts, unknowns and a feasible next experiment.'
            if closing else 'Choose the highest-information affordable question, or finish early.'),
    }, [t for t in tools if not closing or t['function']['name'] == 'finish']


def _changes(before, after, path=''):
    changes = []
    for key in sorted(set(before) | set(after)):
        name = f'{path}.{key}' if path else key
        a, b = before.get(key), after.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            changes.extend(_changes(a, b, name))
        elif a != b:
            changes.append({'component': name, 'before': a, 'after': b})
    return changes


def experiment_context(history):
    """Keep actual fit results available even after their receipts leave recent_results."""
    experiments = []
    reads = []
    for item in history:
        command, result = item['command'], item['result']
        args = command.get('arguments', {})
        data = result.get('data', result)
        report = data.get('report', data)
        if not isinstance(report, dict):
            continue
        if command['tool'] in {'get_dataset_report', 'read_knowledge', 'search_knowledge'}:
            reads.append({'operation_id': command['operation_id'], 'tool': command['tool'],
                'query_hash': digest({k:v for k,v in args.items() if k != 'operation_id'}),
                'arguments': args, 'status': result.get('status')})
        if report.get('score') is None or not isinstance(report.get('pipeline'), dict):
            continue
        row = {k:report.get(k) for k in ('metric', 'score', 'fit_number', 'fit_keys_hash',
            'evaluation_keys_hash', 'evaluation_rows', 'evaluation_periods', 'support_status', 'support_limitations', 'pipeline', 'temporal_comparison', 'inner_feature_transfer')}
        row.update(operation_id=command['operation_id'], evidence_ref=result.get('evidence_ref'),
            score_scope='fit_inner_holdout', outer_validation_scored=False, control_scored=False)
        if experiments:
            baseline = experiments[0]
            matched = all(row.get(k) is not None and row[k] == baseline.get(k)
                for k in ('metric', 'fit_keys_hash', 'evaluation_keys_hash'))
            changes = _changes(baseline['pipeline'], row['pipeline'])
            row['comparison'] = {'baseline_operation': baseline['operation_id'], 'matched_rows_and_metric': matched,
                'changes': changes, 'score_delta': row['score'] - baseline['score'] if matched else None,
                'causal_attribution': 'not_established',
                'limitation': 'One inner split; multiple changed components cannot isolate an individual effect.'}
        experiments.append(row)
    return {'experiments': experiments, 'previous_reads': reads,
        'interpretation_rule': 'A returned score is a measurement, not proof of stable transfer or a causal explanation.'}


def generation_inputs(messages, tools, allowed, budget):
    """Replan at the actual request boundary, including inherited transport fallback."""
    import json
    import copy
    from .policy import constrain_advertised_pipeline
    from ..util import canonical
    if not allowed:
        return messages, tools, allowed
    try:
        context = json.loads(messages[-1]['content'])
    except (ValueError, TypeError):
        return messages, tools, allowed
    if not isinstance(context, dict) or 'action_window' not in context:
        return messages, tools, allowed
    allowed = copy.deepcopy(allowed)
    for tool in allowed:
        constrain_advertised_pipeline(tool['function'].get('parameters', {}).get('$defs', {}))
    constrain_advertised_pipeline(context.get('$defs', {}))
    remaining = budget.limits.max_calls - budget.counter('llm_calls')
    context['action_window'], allowed = action_window(remaining, allowed)
    previous = context.get('remaining', {}).get('generations')
    context.setdefault('remaining', {})['generations'] = remaining
    if previous != remaining:
        context['transport_repair'] = 'The previous response did not produce an executed command. Re-evaluate the remaining budget; do not assume an experiment happened.'
    if not tools and 'tool_argument_schemas' in context:
        names = [t['function']['name'] for t in allowed]
        context['tool_argument_schemas'] = {k:v for k,v in context['tool_argument_schemas'].items() if k in names}
        context['command_schema']['properties']['tool']['enum'] = names
    messages = [*messages[:-1], {**messages[-1], 'content': canonical(context)}]
    return messages, allowed if tools else tools, allowed
