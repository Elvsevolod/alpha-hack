"""Task-local branch identity and bounded, evidence-only experience projections."""
from __future__ import annotations

import re

from .util import canonical, digest, utf8_short


def enabled(controller):
    return getattr(controller, 'prompt_contract', {}).get('version', 3) >= 3


def branch_axes(pipeline):
    """One structural identity; fitted state and ordinary HPO values are excluded."""
    p = pipeline
    from .pipelines import effective_parameters
    parameters = effective_parameters(p) if p.engine in {'direct', 'optuna'} else p.hyperparameters
    return {
        'engine': p.engine, 'family': p.model, 'representation': p.representation,
        'features': digest([sorted(p.include_features) if p.include_features is not None else None,
            sorted(p.exclude_features)]),
        'transformations': digest([op.model_dump(mode='json') for op in p.transformations]),
        'imputation': p.imputation.hash,
        'objective': digest([p.training_objective, p.target_transform, p.log_inverse, p.resampling,
            {key: parameters[key] for key in ('objective', 'loss_function', 'loss', 'class_weight',
                'scale_pos_weight', 'auto_class_weights') if key in parameters}]),
        'decision': digest([p.prediction_policy, p.postprocess_fit_policy, p.decision_method]),
        'transfer': p.shift_strategy,
        'composition': digest([p.ensemble_strategy if p.engine == 'ensemble' else 'single', p.members,
            p.autogluon_profile if p.engine == 'autogluon' else None, p.learners]),
        'custom': p.custom_semantic_hash or p.custom_hash,
        'numeric_representation': digest([p.neighbor_scale, p.text_analyzer, p.max_text_features]),
    }


def branch_key(pipeline):
    return digest(branch_axes(pipeline))


def project(controller, reviews):
    """Rank only comparable public reviews; do not read artifacts or worker values."""
    c = controller
    rows = [row for row in reviews if row['spec'].task_hash == c.task.hash
        and row['spec'].data_version == c.dataset.source_hash
        and row['spec'].validation_hash == c.plan.hash
        and row['spec'].library_versions == c.capabilities.versions]
    note = c.context_memory.store.active_note() or {}
    by_id = {row['spec'].experiment_id: row for row in rows}
    by_review = {row['id']: row for row in rows}
    selected = next((by_id.get(i) or by_review.get(i) for i in note.get('related_ids', [])
        if i in by_id or i in by_review), None)
    selection = 'research_note' if selected else 'latest_comparable'
    if selected is None:
        selected = rows[-1] if rows else None
    words = set(re.findall(r'\w{3,}', note.get('question', '').casefold()))

    def rank(row):
        s = row['spec']
        text = (s.hypothesis.observation + ' ' + s.hypothesis.changed_component).casefold()
        return (branch_key(s.pipeline) == branch_key(selected['spec'].pipeline),
            len(words & set(re.findall(r'\w{3,}', text))), row['sequence'], s.experiment_id)

    def card(row):
        s, review = row['spec'], row['review']
        return {'experiment_id': s.experiment_id, 'branch_id': branch_key(s.pipeline),
            'parent_id': s.parent_id, 'comparator_id': review['comparison']['experiment_id'],
            'outcome': review['outcome'], 'score': review['measurement']['score'],
            'scope': review['measurement']['scope'],
            'tested_design': {'source': 'agent_assumption',
                'changed_component': utf8_short(s.hypothesis.changed_component, 48)},
            'read': {'tool': 'read_run_records', 'arguments': {'ids': [row['id']], 'max_chars': 12000}}}

    past = sorted((row for row in rows if row is not selected), key=rank, reverse=True)[:2] if selected else []
    result = {'version': 1, 'selection_source': selection, 'current': card(selected) if selected else None,
        'previous': [card(row) for row in past],
        'limitation': 'Measured local outcomes do not establish hypothesis acceptance or transferable gains.'}
    # The full projection is a source; inline facts must stay within core limits.
    assert len(canonical(result).encode()) < 12000
    return result
