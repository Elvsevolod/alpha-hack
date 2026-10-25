"""Regression coverage for the premature control-preparation timeout."""
import time

import pytest

from automl_agent.budget import Limits
from automl_agent.completion import allocation
from automl_agent.controller import Controller
from conftest import write_task


@pytest.mark.parametrize('graph,own,available,expected',[
    ({'a':20,'b':30},{'a':20},80,50),
    ({'a':20,'b':0,'blend':2,'c':30,'other':3},{'a':20,'b':0,'blend':2},100,67),
    ({'a':20,'b':30},{'a':20},10,0),
])
def test_allocation_uses_slack_and_reserves_unique_other_nodes(monkeypatch,graph,own,available,expected):
    def estimate(controller,identifiers,*args):
        nodes=own if identifiers==['selected'] else graph
        return {'seconds':sum(nodes.values()),'node_seconds':nodes}
    monkeypatch.setattr('automl_agent.completion.estimate',estimate)
    assert allocation(None,['selected','other'],'selected',[], 'control',available)==expected


def test_observed_slowdown_reduces_pool_and_preserves_refit(tmp_path,monkeypatch):
    c=Controller(write_task(tmp_path),Limits(seconds=180,cpu=1))
    try:
        c.prepare();c.freeze('Updated estimates after preparing the first node')
        saved=c.journal.get('finalization_plan');saved['finalists']=['baseline','leader','third']
        c.journal.put('finalization_plan',saved);c.journal.put('leader_id','leader')
        # Cost-only phantom candidates still need a comparable research score
        # under the quality/cost policy. They must never reach preparation.
        lookup = c.lookup
        def fixture_lookup(identifier):
            if identifier in {'leader','third'}:
                spec, measured = lookup('baseline')
                return spec.model_copy(update={'experiment_id':identifier}), measured.model_copy(
                    update={'experiment_id':identifier,'score':measured.score+1})
            return lookup(identifier)
        monkeypatch.setattr(c, 'lookup', fixture_lookup)
        c.budget.deadline=time.time()+100
        def estimate(controller,identifiers,rows=None,phase=None):
            ready=c.journal.get('prepared_finalists',{}) if phase=='control' else {}
            nodes={identifier:0 if identifier in ready else 10 for identifier in identifiers}
            return {'seconds':sum(nodes.values()),'node_seconds':nodes,'disk_bytes':0}
        original=c._prepare_finalist;calls=[]
        def prepare(identifier,rows,part,seconds):
            calls.append((identifier,part,seconds))
            outcome=original(identifier,rows,part,seconds)
            # Leave room for the 10-second refit and C1 publication/safety
            # margins, but not either additional 10-second preparation.
            if part=='control':c.budget.deadline=time.time()+25
            return outcome
        monkeypatch.setattr('automl_agent.completion.estimate',estimate)
        monkeypatch.setattr(c,'_prepare_finalist',prepare)
        monkeypatch.setattr(c,'finalist_costs',lambda: [])
        c.finalize()
        saved=c.journal.get('finalization_plan')
        assert c.journal.get('control_status')=='completed', c.journal.get('finalization_error')
        assert [r['id'] for r in saved['excluded_before_control']]==['third','leader']
        assert saved['finalists']==c.journal.get('control_opened')['finalists']==['baseline']
        assert [(name,part) for name,part,_ in calls]==[('baseline','control'),('baseline','refit')]
        assert 60<calls[0][2]<64  # Slack retained after protecting C1 margins; old proportional grant was ~29s.
        assert c.journal.get('refit_status')=='completed'
    finally:c.close()


def test_opened_control_pool_is_immutable_when_budget_shrinks(tmp_path,monkeypatch):
    c=Controller(write_task(tmp_path),Limits(seconds=180,cpu=1))
    try:
        c.prepare();c.freeze('Durable control opening cannot be changed on recovery')
        saved=c.journal.get('finalization_plan');saved['finalists']=['baseline','leader']
        c.journal.put('finalization_plan',saved)
        c.journal.transition('control','fixture')
        c.journal.put('control_opened',{'finalists':['baseline','leader']})
        c.budget.deadline=time.time()+5
        monkeypatch.setattr('automl_agent.completion.estimate',lambda *args: {'seconds':100,'node_seconds':{'baseline':50,'leader':50},'disk_bytes':0})
        monkeypatch.setattr(c,'_prepare_finalist',lambda *args:pytest.fail('No worker can start without refit reserve'))
        c.finalize()
        assert c.journal.get('finalization_plan')['finalists']==['baseline','leader']
        assert not c.journal.get('finalization_plan').get('excluded_before_control')
        assert c.journal.get('control_status')=='control_incomplete'
    finally:c.close()


def test_last_preparation_cannot_spend_full_refit_reserve(tmp_path,monkeypatch):
    c=Controller(write_task(tmp_path),Limits(seconds=180,cpu=1))
    try:
        c.prepare();c.freeze('Last preparation consumes the available slack')
        original=c._prepare_finalist
        def prepare(*args):
            outcome=original(*args)
            c.budget.deadline=time.time()+.5
            return outcome
        monkeypatch.setattr(c,'_prepare_finalist',prepare)
        c.finalize()
        assert not c.journal.get('control_opened')
        assert c.journal.get('control_status')=='control_preparation_failed'
        assert c.journal.get('refit_status') is None
    finally:c.close()


def test_disk_reclamation_precedes_pool_exclusion(tmp_path,monkeypatch):
    from types import SimpleNamespace
    c=Controller(write_task(tmp_path),Limits(seconds=180,cpu=1))
    try:
        c.prepare();c.freeze('Reclaim reproducible staging before refusing baseline')
        free=[1024**3];reclaimed=[]
        original_estimate=__import__('automl_agent.completion',fromlist=['estimate']).estimate
        def estimate(*args,**kwargs):
            graph=original_estimate(*args,**kwargs);graph['disk_bytes']=1024**3
            return graph
        def reclaim(controller,required):
            reclaimed.append(required);free[0]=4*1024**3
        monkeypatch.setattr('automl_agent.completion.estimate',estimate)
        monkeypatch.setattr('automl_agent.controller.shutil.disk_usage',lambda path:SimpleNamespace(free=free[0]))
        monkeypatch.setattr('automl_agent.cleanup.reclaim',reclaim)
        c.finalize()
        assert reclaimed[0]==int(2.5*1024**3)
        assert c.journal.get('control_status')==c.journal.get('refit_status')=='completed'
    finally:c.close()
