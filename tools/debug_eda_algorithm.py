"""Controlled offline debugging: scripted HTTP decisions, real bounded ML fits.

This is not an evaluation of a live LLM's ability to choose analyses. Every role
proposal is supplied from the reviewer's contract. No external labels are read.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from automl_agent.eda.agent import EDAAgent
from automl_agent.util import atomic_json


def measured(item):
    value=item['result'].get('data',item['result'])
    return value.get('report') or value


def execute(root):
    root=root.resolve();output=root/'controlled_debug'
    if output.exists():raise ValueError('Controlled debug already exists; no automatic repeat of fits')
    output.mkdir()
    summaries={}
    for name in ['adult','bank','bike','har']:
        reference=json.loads((root/'sealed'/name/'reference_contract.json').read_text())
        def respond(request):
            assert request.method=='POST' # Intercepted in-memory; never sent to network.
            body=json.loads(request.content);context=json.loads(body['messages'][1]['content'])
            if 'request_id' in context:
                spec={**reference,'contract_version':2,'source_hash':context['source_hash'],
                    'file_roles':{r:r for r in ['train.csv','test.csv','sample_submission.csv','data_dictionary.csv']},
                    'feature_columns':'all_nonservice_inputs'}
                proposal={'request_id':context['request_id'],'source_hash':context['source_hash'],
                    'status':'proposed','task_spec':spec,
                    'evidence':{k:['task','inventory','description_csv:data_dictionary.csv'] for k in spec}}
                message={'content':json.dumps(proposal)}
            else:
                history={item['command']['operation_id']:item for item in context['recent_results']}
                if name=='bike' and 'links' not in history:
                    tool='get_dataset_report';operation='links';args={'report':'correlations',
                        'columns':['season','temp','atemp','hum','windspeed','cnt'],'include_target':True,
                        'knowledge_gap':'Measure redundant weather features; categorical season must be excluded from numeric correlations.'}
                elif 'diagnosis' not in history:
                    tool='get_dataset_report';operation='diagnosis';args={'report':'errors','include_target':True,
                        'knowledge_gap':'Measure the fixed inner baseline before a single controlled hypothesis.'}
                elif 'score' not in measured(history['diagnosis']):
                    tool='finish';operation='finish';args={'reason':'Controlled offline scenario: baseline unavailable; no effect claim. Next inspect worker failure and retained fit allowance.'}
                elif 'hypothesis' not in history:
                    baseline=measured(history['diagnosis'])
                    pipeline={'model':'lightgbm','representation':'native','seed':42,
                        'hyperparameters':{'n_estimators':300,'learning_rate':.05,'num_leaves':31}}
                    if name=='bike':
                        pipeline['target_transform']='log1p';change='target_transform';effect='Test compression of rental-count tail; original-scale RMSE may improve or worsen.'
                    else:
                        pipeline['hyperparameters']['class_weight']='balanced';change='class_weight';effect='Test whether higher weight for rarer classes improves Macro F1; precision may fall.'
                    tool='train_model';operation='hypothesis';args={'pipeline':pipeline,'budget_seconds':60,
                        'hypothesis':{'observation':f"Controlled baseline {baseline['metric']}={baseline['score']}; fit target overview={context['fit_target_overview']}",
                            'expected_effect':effect,'changed_component':change,
                            'acceptance':'Improve the primary score on exactly the same fit-only inner split; treat one split as limited evidence.'}}
                else:
                    baseline=measured(history['diagnosis']);candidate=measured(history['hypothesis'])
                    tool='finish';operation='finish'
                    args={'reason':f"Controlled offline scenario, not a live LLM decision. Baseline {baseline.get('score')}; candidate {candidate.get('score')}. "
                        'Unknown: transfer beyond this inner split. Next repeat the promising hypothesis on representative independent periods/groups before claiming an improvement. '
                        f"Protocol warnings: {context['protocol_assessment']['warnings']}"}
                message={'tool_calls':[{'id':operation,'function':{'name':tool,'arguments':json.dumps({'operation_id':operation,**args})}}]}
            return httpx.Response(200,json={'choices':[{'message':message}],
                'usage':{'prompt_tokens':100,'completion_tokens':100}})
        agent=EDAAgent(root/'tasks'/name/'task.md',output/name,output/'MOCK_cost_ledger.json',
            seconds=600,client=httpx.Client(transport=httpx.MockTransport(respond)))
        agent.qwen.key='offline-fixture'
        agent.qwen.provider_mode='http_mock'
        try:summary=agent.autonomous()
        finally:agent.close()
        history=json.loads((output/name/'trajectory.json').read_text()) if (output/name/'trajectory.json').exists() else []
        by_id={item['command']['operation_id']:item for item in history}
        baseline=measured(by_id['diagnosis']) if 'diagnosis' in by_id else {}
        candidate=measured(by_id['hypothesis']) if 'hypothesis' in by_id else {}
        first,second=baseline.get('score'),candidate.get('score')
        same_split=all(baseline.get(k)==candidate.get(k) and baseline.get(k) is not None for k in ['fit_keys_hash','evaluation_keys_hash']) if second is not None else None
        result={'execution':summary,'baseline_score':first,'candidate_score':second,
            'improvement':(first-second if name=='bike' else second-first) if first is not None and second is not None else None,
            'same_inner_rows':same_split,'metric':baseline.get('metric'),'evaluation_rows':baseline.get('evaluation_rows'),
            'hypothesis':'log1p target' if name=='bike' else 'balanced class weights',
            'real_ml_fits':summary['model_fits'],'real_llm_generations':0,'autonomous_decision_quality':'not_evaluated_scripted_decisions',
            'simulated_costs_are_not_actual_spending':True}
        summaries[name]=result
        atomic_json(output/'results.json',summaries)
        print(json.dumps({'task':name,**result},ensure_ascii=False),flush=True)
    lines=['# Контрольные опыты для отладки алгоритма','',
        '**Реальные обучения; решения заданы HTTP-сценарием, живой LLM не использован. API-расход — 0.**',
        'Эталонные роли заданы проверяющим. Метки внешнего теста не читались. Каждая задача использует не более двух fit на одном и том же внутреннем разбиении.', '',
        '| Задача | Метрика | Опора | Изменение | Результат | Улучшение | Внутренняя оценка, строк | Fit |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for name,item in summaries.items():
        def fmt(value):return f'{value:.6f}' if isinstance(value,(float,int)) else 'не получен'
        lines.append(f"| {name} | {item['metric']} | {fmt(item['baseline_score'])} | {item['hypothesis']} | {fmt(item['candidate_score'])} | {fmt(item['improvement'])} | {item['evaluation_rows']} | {item['real_ml_fits']} |")
    lines += ['', 'Положительное улучшение означает лучший результат на этом внутреннем срезе. Это не внешний тест и не статистическое подтверждение устойчивости. Особенно ограничен Bike: один день / 24 строки. Для HAR оценивается перенос на другие группы внутри доступного train.', '',
        'Журналы каждой задачи: events.jsonl, provider/request_*.json, provider/response_*.json, trajectory.json, analyses/fit_*/diagnosis.json. Все обращения к модели в этих журналах имитированы; MOCK_cost_ledger.json содержит искусственный usage и не относится к денежным расходам.', '']
    (output/'REPORT.md').write_text('\n'.join(lines))
    return summaries

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    execute(args.root)
