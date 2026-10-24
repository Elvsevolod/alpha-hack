import json

from automl_agent.budget import Limits
from automl_agent.controller import Controller
from conftest import write_task


def test_full_run_report_is_honest_and_optional_prose_failure_keeps_csv(tmp_path, monkeypatch):
    c=Controller(write_task(tmp_path/'task'),Limits(seconds=120,cpu=1),run_id='reports')
    try:
        result=c.run(no_llm=True)
        assert result['state']=='completed'
        assert result['readable_report_status']=='available'
        assert (c.workdir/'REPORT.md').exists()
        assert 'unassessed' in result['verification_dimensions']['decision_quality']
        before=c.publisher.output.read_bytes()
        def unavailable(*args):raise OSError('controlled optional report failure')
        monkeypatch.setattr('automl_agent.reporting.write_full_summary',unavailable)
        result=c.report()
        assert result['readable_report_status']=='unavailable'
        assert c.publisher.output.read_bytes()==before
        assert json.loads((c.workdir/'run_report.json').read_text())['state']=='completed'
    finally:c.close()
