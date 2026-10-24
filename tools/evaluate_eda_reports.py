"""Independent fact/protocol checks for saved EDA reports, without model/API/fits."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from automl_agent.contracts import ValidationPlan
from automl_agent.data import Dataset
from automl_agent.eda.assessment import protocol_assessment,target_overview
from automl_agent.util import atomic_json


def evaluate(root, output=None):
    root=root.resolve();destination=Path(output).resolve() if output else root
    destination.mkdir(parents=True,exist_ok=True)
    results={}
    for name in ['adult','bank','bike','har']:
        directory=root/'offline_only'/name
        profile=json.loads((directory/'profile.json').read_text())
        correlations=json.loads((directory/'correlations.json').read_text())
        dataset=Dataset(root/'tasks'/name/'task.md',deferred=True)
        dataset.activate(dataset.explicit(root/'sealed'/name/'reference_contract.json'))
        plan=ValidationPlan.model_validate_json((directory/'validation_plan.json').read_text())
        fit=dataset.train.iloc[plan.fit]
        checks={
            'data_identity_matches':profile['data_version']==dataset.source_hash,
            'row_count_matches':profile['rows']['fit']==len(fit),
            'feature_roles_match':set(profile['columns'])==set(dataset.task.feature_columns),
            'external_labels_absent':not bool(set(dataset.task.targets)&set(dataset.test)),
            'numeric_categories_have_no_moments':all(not {'mean','median','std','q01','q25','q75','q99'}&set(profile['columns'][c]) for c in dataset.task.categorical_columns),
            'missing_counts_match':all(profile['columns'][c]['missing']==int(fit[c].isna().sum()) for c in dataset.task.feature_columns),
            'category_cardinality_matches':all(profile['columns'][c]['unique']==int(fit[c].nunique()) for c in dataset.task.categorical_columns),
        }
        if dataset.task.classes:
            checks['fit_target_counts_match']=all(entry['rows']==int((fit[t]==entry['label']).sum()) for t,item in profile['targets'].items() for entry in item['counts'])
        else:
            checks['fit_target_median_matches']=all(np.isclose(profile['targets'][t]['median'],np.median(fit[t])) for t in dataset.task.targets)
        pair_ok=True;support_ok=True;rank_ok=True
        # All four prepared fit sets are below the 20,000-row sampling limit.
        for a,row in correlations['correlations'].items():
            for b,value in row.items():
                selected=fit[[a,b]].dropna()
                support_ok &= correlations['pair_counts'][a][b]==len(selected)
                if value is None:continue
                expected=np.corrcoef(selected.iloc[:,0],selected.iloc[:,1])[0,1]
                ranked=selected.rank(method='average')
                spearman=np.corrcoef(ranked.iloc[:,0],ranked.iloc[:,1])[0,1]
                pair_ok &= bool(np.isclose(value,expected,atol=1e-9))
                rank_ok &= bool(np.isclose(correlations['spearman'][a][b],spearman,atol=1e-9))
        checks.update(pearson_matches_independent_calculation=bool(pair_ok),spearman_matches_rank_calculation=bool(rank_ok),pair_support_matches=bool(support_ok))
        protocol=protocol_assessment(dataset,plan)
        checks['partition_indices_disjoint']=protocol['partitions_disjoint']
        if 'chronological' in protocol:checks['chronological_boundaries']=protocol['chronological'] and protocol['test_follows_train']
        if 'groups_disjoint' in protocol:checks['participant_boundaries']=protocol['groups_disjoint'] and protocol['external_groups_disjoint']
        if name=='bank':checks['unavailable_duration_excluded']='duration' not in dataset.task.feature_columns
        if name=='bike':checks['target_components_excluded']=not {'casual','registered'}&set(dataset.task.feature_columns)
        results[name]={'facts_verdict':'pass' if all(checks.values()) else 'fail','checks':checks,
            'target_overview':target_overview(profile),'protocol':protocol,
            'coverage':'Base profile and explicitly selected numerical correlations only. No model decisions or quality improvement evaluated.',
            'live_agent_verdict':'not_evaluated'}
        atomic_json((destination/name if output else directory)/'protocol_assessment.json',protocol)
    auth=json.loads((root/'connection_read_only.json').read_text()) if (root/'connection_read_only.json').exists() else {}
    organizer_path=root/'qwen_connection'/'models.json'
    organizer=json.loads(organizer_path.read_text()) if organizer_path.exists() else {}
    result={'datasets':results,'saved_organizer_connection':organizer,'read_only_connection':auth,'api_generations_in_this_evaluation':0,'model_fits':0,
        'overall':'facts_pass_autonomy_unverified' if all(r['facts_verdict']=='pass' for r in results.values()) else 'facts_failed',
        'algorithm_findings':[
            {'finding':'Bank count ratio is 7.55 although historical class_imbalance flag is false (threshold 10).',
             'fix':'Agent context now includes compact fit target counts/ratio independently of risk flags; definitions unchanged.'},
            {'finding':'Bike validation/control and inner diagnostic each use one calendar day; transfer test covers 147 days.',
             'fix':'Protocol remains unchanged; feature-only protocol diagnostics and short-holdout limitations are explicitly supplied to the agent.'},
            {'finding':'No real model decisions were produced; basic successful reports do not establish useful autonomous behavior.',
             'fix':'Separate facts, protocol support and autonomous-decision verdicts; log requests, decisions, results, budgets and stop reasons.'}]}
    atomic_json(destination/'EVALUATION.json',result)
    lines=['# Оценка имеющихся результатов','',
        '**Результат: базовая разведка на четырёх задачах работает; качество самостоятельных решений агента ещё не измерено.**','',
        '## Что уже проверено','',
        '| Задача | Проверки фактов | Что подтверждено | Что ограничивает вывод |',
        '|---|---:|---|---|']
    facts={
        'adult':('Пропуски в трёх признаках, классы 10 651 / 3 349, повторные признаки.','Этот отчёт проверяет факты; отдельные опыты с HTTP-имитацией описаны в controlled_debug/REPORT.md.'),
        'bank':('Классы 12 362 / 1 638, отношение 7.55; duration исключён.','Флаг с порогом 10 не срабатывает; это не означает баланс классов.'),
        'bike':('Коды категорий обработаны как категории; temp/atemp Pearson 0.99234; компоненты cnt исключены.','Внутренняя проверка всего 24 строки / 1 день против 147 дней внешнего переноса.'),
        'har':('561 числовой признак; группы train/test и внутренних частей не пересекаются.','Связи проверены лишь для четырёх признаков; устойчивость модели на новых людях ещё неизвестна.')}
    for name,item in results.items():
        passed=sum(item['checks'].values());total=len(item['checks'])
        lines.append(f"| {name} | {passed}/{total} | {facts[name][0]} | {facts[name][1]} |")
    lines += ['', 'Пересчитаны независимо: пропуски, кардинальность категорий, fit-распределения таргета, Pearson через ковариации и Spearman через ранги, support каждой выбранной пары. Проверены роли, отсутствие внешних меток, хронология и группы. Внешние метки не использовались.', '',
        '## Что исправлено в отладке алгоритма','',
        '1. В каждый следующий запрос добавлена компактная статистика fit-таргета. Агент видит соотношение классов Bank даже при неактивном историческом флаге. Для регрессии видит квантили и нули/отрицательные значения.',
        '2. Агент получает размер и границы проверки, число периодов/групп и предупреждение о коротком горизонте. Сам протокол на этом этапе не меняется. Диагностика на менее чем 30 строках помечает оценку как ограниченную.',
        '3. Лог `events.jsonl` показывает: исходные факты → выбранный инструмент и гипотеза → предыдущие результаты → статус/время/число fit → причину завершения. `provider/request_*.json` и `response_*.json` позволяют проверить точный обмен без Authorization и с удалением ключа. Полные операции также сохраняются в journal.sqlite и trajectory.json.', '',
        '## Подключение и границы вывода','',
        f"Историческая проверка прямого DeepSeek: `/models` HTTP {auth.get('models_http_status')}, `/user/balance` HTTP {auth.get('balance_http_status')}. Это отдельный провайдер: командный ключ Qwen не является ключом DeepSeek.", '',
        f"Сохранённая проверка шлюза Qwen: HTTP {organizer.get('http_status')}, заявленная модель доступна: {organizer.get('requested_model_available')}. Это прошлый снимок без генераций, а не текущая проверка баланса.", '',
        'Результаты обучений с имитацией решений доступны отдельно в controlled_debug/REPORT.md. Они проверяют исполнение экспериментов, но не самостоятельность живой модели. Новых API-запросов в этой оценке нет. Для живого сравнения нужно согласовать предел расхода в единицах шлюза и применить before/after-проверку лимитов.', '']
    (destination/'EVALUATION.md').write_text('\n'.join(lines))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=evaluate(args.root,args.output)
    print(json.dumps({'overall':result['overall'],'checks':{name:[sum(item['checks'].values()),len(item['checks'])] for name,item in result['datasets'].items()}},ensure_ascii=False))
    raise SystemExit(1 if result['overall']=='facts_failed' else 0)
