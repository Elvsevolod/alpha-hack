import json
import sqlite3
from types import SimpleNamespace

import pytest

from automl_agent.contracts import PipelineSpec
from automl_agent.research import research_options
from test_research_learning import trial
from test_tool_api_v2 import agent


@pytest.fixture
def controller(tabular, monkeypatch):
    from automl_agent import directions
    _, _, task = tabular
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE events(seq INTEGER PRIMARY KEY,kind TEXT,payload TEXT)')
    records, operations = [], {}
    monkeypatch.setattr(directions, 'direction_map', lambda _: dict(directions=[], affordable_unattempted=[],
        technical_failure_is_poor_quality=False, no_mandatory_model_count=True))
    c = SimpleNamespace(task=task, records=lambda: records,
        plan=SimpleNamespace(hash='split', strategy='random', evaluation_mode='single_holdout', observations=[]),
        journal=SimpleNamespace(db=db, operation=lambda key: operations.get(key)))
    yield c, records, operations
    db.close()


def branch(c, name='feature_subset'):
    return next(row for row in research_options(c)['branches'] if row['branch']==name)


@pytest.mark.parametrize('selection', [dict(include_features=['number', 'second']), dict(exclude_features=['empty'])])
def test_effective_nonempty_subset_has_its_own_measured_direction(controller, selection):
    c, records, _ = controller
    assert branch(c)['status']=='no_admitted_attempt'
    s, r = trial('subset', 1, pipeline=PipelineSpec(model='ridge', **selection))
    records.append((s, r))
    row = branch(c)
    assert row['status']=='measured' and row['measured_candidates']==1
    assert 'optional' in row['diagnostic'] and 'include_features/exclude_features' in row['diagnostic']
    assert 'pipeline' not in row  # No invented executable list of column names.


def test_explicit_full_feature_list_does_not_claim_a_subset_test(controller):
    c, records, _ = controller
    records.append(trial('full', 1, pipeline=PipelineSpec(model='ridge', include_features=c.task.feature_columns)))
    assert branch(c)['status']=='no_admitted_attempt'


@pytest.mark.parametrize('status,scope,expected', [('timeout','full','technical_failure'),
    ('outcome_unknown','full','technical_failure'), ('completed','pilot','no_admitted_attempt')])
def test_technical_or_pilot_outcome_is_not_a_measured_subset_result(controller, status, scope, expected):
    c, records, _ = controller
    s, r = trial('subset', 1, pipeline=PipelineSpec(model='ridge', exclude_features=['empty']))
    records.append((s, r.model_copy(update=dict(status=status,scope=scope,score=1 if status=='completed' else None))))
    row = branch(c)
    assert row['status']==expected and row['measured_candidates']==0


def test_old_protocol_measurement_does_not_cover_current_search(controller):
    c, records, _ = controller
    records.append(trial('old', 1, pipeline=PipelineSpec(engine='optuna', model='ridge'),validation_hash='old_split'))
    assert branch(c, 'optuna')['status']=='no_admitted_attempt'
    records.append(trial('current', 2, pipeline=PipelineSpec(engine='optuna', model='ridge')))
    assert branch(c, 'optuna')['measured_candidates']==1


def test_old_protocol_proposal_is_not_a_current_proposal(controller):
    c, _, operations = controller
    command=dict(operation_id='old_proposal',tool='tune_model',arguments=dict(pipeline=dict(engine='optuna',model='ridge')))
    c.journal.db.execute('INSERT INTO events(kind,payload) VALUES (?,?)',('qwen_decision',json.dumps(dict(command=command))))
    operations['old_proposal']=SimpleNamespace(protocol_hash='old_split',status='succeeded')
    assert branch(c, 'optuna')['status']=='no_admitted_attempt'


def test_single_predictor_marks_subset_unavailable_without_inventing_empty_features(controller):
    c, _, _ = controller
    c.task=c.task.model_copy(update=dict(feature_columns=['number'],categorical_columns=[]))
    row=branch(c)
    assert row['status']=='not_applicable_to_current_contract' and row['prerequisites']


def test_subset_coverage_survives_public_reads_and_all_compaction_modes(agent):
    from test_layered_prompt_context import read_all
    c=agent
    before=c.publisher.output.read_bytes()
    snapshot=c.context_memory.snapshot(c.system_prompt)
    for mode in ['native','json']:
        for stage in range(4):
            prepared=c.context_memory.compressor.prepare(snapshot,mode,budget=262144,minimum_stage=stage)
            core=json.loads(json.loads(prepared.wire_json)['messages'][1]['content'])['core']
            summary=json.loads(read_all(c,core['current_work']['research_options']))
            row=next(row for row in summary['branches'] if row['branch']=='feature_subset')
            assert row['status']=='no_admitted_attempt'
            assert len(json.dumps(summary,ensure_ascii=False,separators=(',',':')).encode())<=3072
    assert not c.journal.get('control_opened') and c.budget.counter('candidate_count')==0
    assert c.publisher.output.read_bytes()==before
