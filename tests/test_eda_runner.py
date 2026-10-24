"""Standalone EDA must not activate the research controller or overwrite submissions."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from automl_agent.data import ContractUnresolved
from automl_agent.eda.runner import run
from conftest import write_task


def test_offline_eda_uses_existing_services_without_models_or_api(tmp_path, monkeypatch):
    task = write_task(tmp_path / 'task')
    def forbidden(*args, **kwargs):
        pytest.fail('Standalone EDA activated training or API')
    monkeypatch.setattr('automl_agent.controller.Controller.__init__', forbidden)
    monkeypatch.setattr('automl_agent.qwen.QwenClient.__init__', forbidden)
    monkeypatch.setattr(httpx.Client, 'send', forbidden)
    before = {p.name: p.read_bytes() for p in task.parent.iterdir() if p.is_file()}
    output = tmp_path / 'report'
    result = run(task, output, columns=['value', 'answer'])
    assert result['state'] == 'completed'
    assert result['api_calls'] == result['model_fits'] == 0
    assert not result['submission_written']
    assert json.loads((output / 'correlations.json').read_text())['status'] == 'available'
    profile = json.loads((output / 'profile.json').read_text())
    assert profile['scope'] == 'fit' and profile['targets']['answer']
    assert not list(output.rglob('submission.csv'))
    assert before == {p.name: p.read_bytes() for p in task.parent.iterdir() if p.is_file()}
    with sqlite3.connect(output / 'journal.sqlite') as db:
        assert db.execute('SELECT count(*) FROM experiments').fetchone()[0] == 0


def test_external_contract_and_missing_contract_are_explicit(tmp_path):
    task = write_task(tmp_path / 'task')
    source = task.parent / 'task_spec.json'
    external = tmp_path / 'explicit.json'
    external.write_bytes(source.read_bytes())
    source.unlink()
    with pytest.raises(ContractUnresolved):
        run(task, tmp_path / 'missing')
    assert not (tmp_path / 'missing').exists()
    assert run(task, tmp_path / 'external', contract_path=external)['state'] == 'completed'


def test_runner_rejects_overwriting_or_polluting_source_tree(tmp_path):
    task = write_task(tmp_path / 'task')
    for output in [task.parent, tmp_path, task.parent / 'new_report']:
        with pytest.raises(ValueError):
            run(task, output)
    assert run(task, task.parent / 'outputs' / 'safe')['state'] == 'completed'


def test_real_cli_outputs_summary_without_credentials(tmp_path):
    task = write_task(tmp_path / 'task')
    output = tmp_path / 'cli'
    result = subprocess.run([sys.executable, '-m', 'automl_agent.eda', str(task), '--output', str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary['api_calls'] == summary['model_fits'] == 0
    assert summary['correlations_status'] == 'not_requested'
    assert (output / 'REPORT.md').is_file()
