"""Portable EDA reports: measured receipts separate from unverified agent prose."""
from ..util import atomic_bytes, atomic_json
from ..reporting import verification_dimensions
from .planning import experiment_context


def write_reports(directory, summary, history, evaluation):
    facts = experiment_context(history)
    report = {'version': 1, 'verification_dimensions':verification_dimensions(), 'execution': summary, 'evaluation_contract': evaluation,
        'experiments': facts['experiments'], 'claim_support': 'unassessed',
        'scope': 'fit-only exploration; no external labels, control, full refit or submission'}
    atomic_json(directory / 'evaluation_report.json', report)
    lines = ['# EDA evaluation report', '', f"Execution: **{summary['state']}**. Stop: {summary['stop_reason']}.", '',
        f"Generations: {summary['model_generations']}; fit attempts: {summary['model_fits']}.", '',
        '## Measured experiments', '', '| Operation | Model | Metric | Score | Evaluation rows | Periods |',
        '|---|---|---|---:|---:|---:|']
    for item in facts['experiments']:
        lines.append(f"| {item['operation_id']} | {item['pipeline']['model']} | {item['metric']} | {item['score']:.6g} | {item['evaluation_rows']} | {item['evaluation_periods'] or '—'} |")
    lines += ['', '## Evaluation and limitations', '', report['scope'] + '.', '']
    for item in facts['experiments']:
        lines.append(f"- {item['operation_id']}: support={item['support_status']}; split and component comparison in evaluation_report.json.")
        for limitation in item.get('support_limitations') or []:
            lines.append(f'  - {limitation}')
        comparison = item.get('temporal_comparison')
        if comparison:
            lines.append(f"  - Paired comparison: {comparison['verdict']}; delta interval={comparison.get('delta_interval')}; meaningful gain floor={comparison.get('minimum_absolute_gain')}.")
    lines += ['', '## Agent conclusion — interpretation requires review', '',
        (summary.get('finish') or {}).get('reason', 'No agent conclusion was completed. Unknowns remain.'), '',
        'Execution completion is not validation of the explanation or a guarantee of external predictive quality.', '']
    atomic_bytes(directory / 'REPORT.md', '\n'.join(lines).encode())
