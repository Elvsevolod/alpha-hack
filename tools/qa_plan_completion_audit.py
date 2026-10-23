"""Requirement-level audit. Implementation evidence is not proof of live quality."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'План_исправлений_по_QA_организаторов.md'
OUT=ROOT/'outputs/qa_plan_implementation_A'

PROOF={
 'А1':['tests/test_qa_delivery_A.py','B1_protocol_regression.log','automl_agent/controller.py:complete_operation/recover_operations'],
 'А2':['tests/test_qa_diagnostic_protocol.py','B1_protocol_regression.log','sqlite_guest_final/REPORT.json','tools/diagnostic_store.py:receive/claim/finish_job'],
 'А3':['tests/test_qa_diagnostic_protocol.py','independent_signal/REPORT.json','supervisor_sqlite_fix_02/REPORT.json','tools/diagnostic_3h.py:Guard/watchdog_main'],
 'А4':['tests/test_qa_delivery_A.py','tests/test_qa_qwen_unknown_outcome.py','unknown_outcome_final.log','unknown_outcome_exact_v033/REPORT.json'],
 'Б1':['tests/test_qa_B1_source_semantics.py','B1_semantics_fixed.log','B1_semantics_final.log','B_exact_v032_semantics/REPORT.json'],
 'Б2':['tests/test_qa_behavior_B.py:test_finish_with_one_result_early_and_irreversible','tests/test_final_boundaries.py','B_final_temporal_integration.log','automl_agent/controller.py:dispatch/report'],
 'Б3':['tests/test_qa_behavior_B.py:test_class_metrics_use_full_confusion_and_no_counts/test_ten_class_macro_f1_never_exposes_nine_to_reconstruct_small_tenth','B_final_temporal_integration.log','automl_agent/evaluation.py:class_metrics'],
 'Б4':['tests/test_qa_behavior_B.py:test_control_uses_exact_best_and_is_order_independent/test_ensemble_windows_retrain_all_dependencies_and_optional_failure_is_limited','B_final_temporal_integration.log','automl_agent/finalists.py','B_exact_v031_acceptance/REPORT.json']}


def run():
 text=PLAN.read_text();rows=[];section='';sub='';counter={}
 for line_number,line in enumerate(text.splitlines(),1):
  if line.startswith('## '):section=line[3:];sub=''
  if line.startswith('### '):sub=line[4:]
  if not (line.startswith('- ') or line.startswith('**Приёмка:**')):continue
  if section.startswith('0.'):continue
  requirement=line.removeprefix('- ').removeprefix('**Приёмка:** ')
  group=next((key for key in PROOF if sub.startswith(key+'.')),None)
  status='evidence_reviewed_effective_scope';evidence=[];limitation=''
  if group:
   status='implemented_offline_tested';evidence=PROOF[group]
   limitation='Synthetic/offline evidence; general live behavior and quality are not inferred'
  elif section.startswith('4.'):
   status='implemented_or_measured_defaults_retained';evidence=['SECTION4_STATUS.json','docs/qa_section4.md','final_v038_exact/exact/REPORT.json']
   limitation='MAPE/balanced accuracy and explicit numeric clustering integrated; decision-rule/pseudo-label pilots measured without sufficient gain to change defaults. No hidden-quality claim'
  elif sub=='Реальные серии и сравнение':
   if ('После офлайн-приёмки' in requirement or 'Для Б повторить' in requirement or 'самостоятельной интерпретации' in requirement):
    status='real_A_retry_accepted_B_regression_failed';evidence=['STATE.json','live_A028_retry90_run06/REPORT.json','live_B033_retry90_run02/REPORT.json','final_v038_control_replay/replay/REPORT.json']
    limitation='A retry passed both tasks. B classification passed; regression preparation timed out. Final v038 fixes allocation and passes offline replay, but was not live rerun'
   elif 'Каждая новая платная' in requirement:
    status='authorized_series_finished_no_further_run';evidence=['STATE.json','comparison_short_retry_90_90_AUTHORIZED.json','live_A028_retry90_run06/api_limits_after.json','live_B033_retry90_run02/api_limits_after.json'];limitation='Observed aggregate42.72924177210933 of200; provider lag has no proven upper bound. No automatic paid retry'
   elif 'Успешная приёмка пары' in requirement:
    status='actual_A_accepted_B_pair_failed';evidence=['B_exact_v033_semantics/REPORT.json','watchdog_exact_A028_fixed/REPORT.json','watchdog_exact_B033_fixed/REPORT.json'];limitation='Actual A passed both tasks; actual B regression did not complete control/refit. Later synthetic final v038 checks do not rewrite that result'
   elif 'Б обоснованный ранний' in requirement:
    status='implemented_offline_tested';evidence=['tests/test_qa_behavior_B.py','B_exact_v032_semantics/REPORT.json']
   elif 'До оценки результатов' in requirement:
    status='preregistered_no_valid_real_comparison';evidence=['comparison_preregistration.json','comparison_short_authorized_200.json','SHORT_COMPARISON_RETRY_90.json','short_comparison_missing_control_scores.log'];limitation='B regression did not reach full accepted control/refit; short comparison_valid=false, no quality rollback decision inferred'
   else:
    status='local_evidence_only';evidence=['B_exact_v032_semantics/CAPACITY.json','comparison_preregistration.json'];limitation='No 7 CPU / 28000 MiB production-resource or private-quality claim'
  elif sub=='Обязательные офлайн-проверки':
   if 'Полная трёхчасовая репетиция' in requirement:
    status='original_failed_repeat_superseded_by_user';evidence=['rehearsal_3h_02/REPORT.json','sqlite_guest_final/REPORT.json','STATE.json'];limitation='Do not repeat 3h by explicit user instruction. Original full-duration success remains unproved, not a remaining authorized repetition'
   else:
    status='offline_evidence_with_recorded_revision_deltas';evidence=['ACCEPTANCE_v031.json','B1_protocol_regression.log','B1_semantics_final.log','unknown_outcome_final.log','B_final_temporal_integration.log'];limitation='See each acceptance artifact for exact archive, case assertions and inherited evidence; not a live-Qwen result'
  elif sub=='Документы и поставка':
   status='implemented_local_delivery';evidence=['REPORT.md','FINAL_v038_MANIFEST.json','../../dist/solution_v038.manifest.json']
   if 'При регрессии откатывать' in requirement:
    status='rollback_policy_prepared_unexercised';evidence=['comparison_preregistration.json','B_blocks_v031/MANIFEST.json','semantic_and_unknown_v033.manifest.json'];limitation='No new real A/B regression has been measured, so no quality rollback has occurred'
   elif 'Отправка на платформу' in requirement:
    status='separate_action_not_authorized';evidence=['STATE.json'];limitation='No submission performed'
  else:
   status='source_basis_retained';evidence=['REPORT.md','../../План_исправлений_по_QA_организаторов.md'];limitation='Background/basis is not implementation proof'
  counter[group or section]=counter.get(group or section,0)+1
  rows.append({'id':(group or section.split('.')[0])+'.'+str(counter[group or section]),'plan_line':line_number,
   'section':section,'subsection':sub,'requirement':requirement,'status':status,'evidence':evidence,'scope_limit':limitation})
 result={'schema_version':1,'plan':str(PLAN),'plan_sha256':hashlib.sha256(text.encode()).hexdigest(),'candidate':'dist/solution_v038.zip',
  'requirements':rows,'completion_proved':False,'goal_complete':False,
  'outstanding':['Successful real B pair and valid A/B quality comparison not established; latest request prepares final ZIP without another paid run','Original three-hour acceptance failed and repeat explicitly waived; no false success claim'],
  'effective_user_overrides':['No repeat of the three-hour offline rehearsal','Proceed to B before live acceptance of A'],
  'real_A_series_completed':True,'real_A_series_accepted':True,'real_B_series_started':True,'real_B_series_accepted':False,'latest_archive_live_tested':False,'independent_audit_closed':True,
  'no_automatic_paid_retry':True}
 (OUT/'PLAN_COMPLETION_AUDIT_v038.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('Audited',len(rows),'explicit clauses; full objective remains incomplete')

if __name__=='__main__':run()
