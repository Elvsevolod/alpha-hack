from automl_agent.eda.decisions import decision_ledger, decision_assessment
from automl_agent.util import digest


def test_ledger_keeps_receipt_scope_failure_and_claim_support_separate():
    ref={'id':'observation-a','kind':'observation','version':'v1'}
    first={'command':{'operation_id':'analysis','tool':'get_dataset_report','arguments':{
        'scope':'fit','report':'correlations','knowledge_gap':'Is association supported?'}},
        'result':{'status':'technical_failure','evidence_ref':ref,
            'provenance':{'protocol_hash':'old-split'},'data':{'analysis_status':'technical_failure'}}}
    second={'command':{'operation_id':'experiment','tool':'train_model','arguments':{'hypothesis':{
        'observation':'A measured relation may help','expected_effect':'lower error','acceptance':'matched inner split',
        'evidence_refs':[ref]}}},'result':{'status':'success','data':{'score':.5,'metric':'rmse'}}}
    ledger=decision_ledger([first,second],{'source_id':'profile:p'},'current-split')
    item=ledger['entries'][1]
    assert item['result_hash']==digest(second['result'])
    assert item['score']==.5
    assert item['evidence_links']==[{'reference':ref,'prior_receipt':True,'result_available':False,
        'scope_valid':False,'claim_support':'unassessed'}]
    assert decision_assessment(ledger,True)['status']=='unassessed'
    assert len(decision_ledger([first,second],None,'current-split')['entries'])==2
