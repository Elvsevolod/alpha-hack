"""Reporting keeps execution facts separate from scientific claims."""
from .util import atomic_bytes


def verification_dimensions():
    return {'constraints':'Runtime receipts and resource checks; not a new submission certificate.',
        'decision_quality':'unassessed; completing a run does not verify the model explanation.',
        'predictive_quality':'Task-local measurements only; no claim of competition or external superiority.'}


def write_full_summary(directory, report):
    lines=['# Research run report','',f"Execution: **{report['state']}**.",
        f"Control: {report['control_status']}; refit: {report['refit_status']}.",
        f"Model calls: {report['model_calls']}; research candidates: {report['research_candidates']}.",'',
        '## Separate verification dimensions','']
    lines += [f'- {key}: {value}' for key,value in report['verification_dimensions'].items()]
    lines += ['', '## Reproduction', '', f"Task hash: {report['task_hash']}.",
        f"Run: {report['run_id']}. Exact measurements, configuration, resources and publication are in run_report.json.",
        '', '## Stop reason — not an independently verified scientific explanation','',
        report.get('research_stop_reason') or report.get('failure') or 'No model-written stop reason recorded.', '']
    atomic_bytes(directory/'REPORT.md','\n'.join(lines).encode())
