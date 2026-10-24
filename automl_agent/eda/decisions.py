"""Compact auditable decision context, derived from original tool receipts.

This is a structural assessment, never an automatic judge of whether a claim is true.
It does not select analyses or prescribe transformations.
"""
from ..util import digest, utf8_short

POLICIES = {
    'free': '',
    'evidence': '''Use decision_ledger to keep your current question grounded in measured results.
For an experiment, state the observation, an alternative explanation, the smallest useful test,
and its acceptance condition in the existing hypothesis fields. Copy exact evidence_refs from
receipts when available. Failed or incomplete analyses leave uncertainty. Read details through
existing history tools if needed. Do not treat the ledger as validation of your interpretation.
''',
    'alternatives': '''Before selecting the next tool, compare two plausible next questions by
expected information and remaining time/fits. Explain the choice briefly in knowledge_gap,
hypothesis.observation or finish.reason, whichever the chosen action supports. Use measured
evidence; do not add a mandatory analysis, an extra model call or an obligatory training.
''',
}


def decision_ledger(history, profile_reference, protocol_hash):
    entries = []
    known = {}
    for item in history:
        command, result = item['command'], item['result']
        arguments = command.get('arguments', {})
        data = result.get('data', result)
        report = data.get('report', {})
        if not isinstance(report, dict):
            report = {}
        provenance = result.get('provenance', {})
        hypothesis = arguments.get('hypothesis') or {}
        refs = hypothesis.get('evidence_refs', [])
        # Identity and success are separate. Receipt existence cannot prove claim support.
        links = []
        for ref in refs:
            previous = known.get((ref.get('id'), ref.get('version')))
            links.append({'reference': ref, 'prior_receipt': previous is not None,
                'result_available': previous['available'] if previous else None,
                'scope_valid': previous['scope_valid'] if previous else None,
                'claim_support': 'unassessed'})
        scope_valid = provenance.get('protocol_hash') == protocol_hash if provenance.get('protocol_hash') else None
        available = result.get('status') == 'success' and data.get('analysis_status','available') in {'available','success','completed'}
        entry = {'operation_id':command['operation_id'], 'tool':command['tool'],
            'question':utf8_short(arguments.get('knowledge_gap') or hypothesis.get('observation') or arguments.get('reason') or '',1000),
            'result_hash':digest(result), 'evidence_ref':result.get('evidence_ref'),
            'status':result.get('status'), 'analysis_status':data.get('analysis_status',report.get('status')),
            'delivery_complete':result.get('complete'), 'scope':arguments.get('scope'),
            'scope_valid':scope_valid, 'result_available':available, 'evidence_links':links,
            'expected_effect':utf8_short(hypothesis.get('expected_effect',''),600),
            'acceptance':utf8_short(hypothesis.get('acceptance',''),600),
            'claim_support':'unassessed'}
        # Values are copied, not judged or compared across unmatched splits.
        for key in ('score','metric','fit_number','support_status','fit_keys_hash','evaluation_keys_hash'):
            if key in data or key in report:
                entry[key] = data[key] if key in data else report[key]
        entries.append(entry)
        ref = result.get('evidence_ref')
        if ref:
            known[(ref.get('id'),ref.get('version'))] = {'available':available,'scope_valid':scope_valid}
    return {'version':1, 'base_profile':profile_reference, 'protocol_hash':protocol_hash,
        'entries':entries, 'limitation':'Receipt linkage and status only; scientific interpretation and claim support require review.'}


def decision_assessment(ledger, finished):
    experiments = [e for e in ledger.get('entries',[]) if e['tool']=='train_model']
    return {'status':'unassessed', 'execution_finished':bool(finished),
        'experiments':len(experiments),
        'experiments_with_explicit_refs':sum(bool(e['evidence_links']) for e in experiments),
        'references_unresolved_in_receipt_history':sum(not link['prior_receipt'] for e in experiments for link in e['evidence_links']),
        'claim_support':'unassessed',
        'limitation':'Completion, valid references and mocked decisions do not establish reasoning quality or predictive improvement.'}
