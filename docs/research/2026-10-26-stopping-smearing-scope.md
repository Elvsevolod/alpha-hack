# Неиспользуемая коррекция внутри native early stopping

## Подтверждённая причина отказа

В [замороженном regression run](2026-10-26-autonomy-regression-result.md) DirectPipeline с log1p + OOF-smearing + early stopping отказал на истории из трёх периодов. Отказ был реальным, но не означал, что независимую финальную коррекцию на текущем fit обучить невозможно.

`DirectPipeline.fit` создаёт inner helper для native stopping, затем отдельно учит финальный inverse на всём разрешённом fit. Inner helper раньше наследовал log_inverse=oof_smearing и учил дополнительную коррекцию на меньшей временной истории. `training_schedule.fit_with_schedule.primary` эту коррекцию не получает/не применяет: на регрессии он сравнивает expm1(actual) и expm1(prediction), то есть naive original-scale values. Поэтому inner inverse был лишним обучением и источником лишнего fold-support требования.

В воспроизведении на четырёх периодах до исправления два `_fit_inverse`: stopping inner на135 строках и root на180. На трёх периодах stopping inner имеет лишь два периода и один forward OOF fold, из-за чего root до своего корректного двухфолдового inverse вообще не доходит. До исправления набор **2 failed / 2 passed**. Тесты фиксируют как нежелательный вложенный fit, так и ложный отказ; двухпериодный настоящий дефицит истории и explicit final-output checkpoints сохранены как отдельные границы.

## Исправление и семантика метрики

Только raw native stopping-helper получает log_inverse=naive и пустой learned_inverse. Число итераций выбирается тем же прежним callback, с теми же fit-inner данными и naive original-scale RMSE. Финальная OOF-коррекция остаётся самостоятельной процедурой полного fit, с прежним требованием минимум двух независимо fitted folds, ограничениями времени, fresh preprocessing и без внешних labels.

В learning curve явно сохраняется stopping_log_inverse=naive, final_log_inverse и surrogate_differs_from_final_decision=True при smearing. Прежде regression curve не отражала это расхождение. Не утверждается, что stopping теперь оптимизирует финальные calibrated predictions: он по-прежнему surrogate. На тестовых данных выбранные iterations и вся naive stopping curve совпадают с парным naive-рецептом.

Explicit checkpoint selection, где измеряется конечный output каждого кандидата, продолжает учить свой inverse внутри checkpoint fit. Эту действительно нужную вложенность не удаляли. История из двух периодов по-прежнему не удовлетворяет двухфолдовому final inverse. Frozen refit сохраняет factor/state и не переобучает коррекцию на открытом control.

## Проверки

- Связанный набор: **15 passed**, 11,29 s — новые случаи, nested learning, recipe/protocol guards.
- С schedule execution identity: **20 passed**, 31,54 s. Наборы перекрываются и не суммируются.
- Настоящий Controller/worker: три fit-периода → log1p/smearing/stopping candidate → frozen control/refit → CSV25 строк с точными ID и конечными values. Finalist закреплён технической fixture; автономный выбор или качество этим тестом не доказываются. Full refit180 строк сохраняет тот же inverse в manifest и сериализованной модели.
- Точный offline replay первоначальных performance_1 fit_X/fit_y, request pipeline/task: **22 054 строки, три периода**, после исправления обучение завершено, два независимых inverse folds, выбран521 stopping rounds. Использованы только исходные fit-входы; новых control/validation quality scores нет. Fresh technical allocation90 s и wall5,46 s не являются сравнением скорости с исходным запуском. Первая версия direct replay harness не добавила task, который обычно инъецирует worker, и остановилась; после восстановления полного контекста replay успешен, runtime для harness не ослаблялся.

[Ledger](../../research/stopping_smearing_scope_20261026/summary.json) содержит XML/source/input hashes и точный replay. Raw XML и replay JSON — ignored outputs/stopping_smearing_scope_20261026. Один первоначальный pytest command ссылался на отсутствующее имя файла и не запускал тесты; он не включается в успешные counts.

Связь с ТЗ — полный учёт auxiliary fits и времени (с.7), learning только внутри train и корректные scope результата (с.17); первичные ссылки собраны в [аудите требований](2026-10-26-agent-path-audit.md). Новых зависимостей/нейросетей/production/ZIP нет. Исправление применено **после** live run; дополнительных real API calls0. Улучшение RMSE нового smearing-рецепта, перенос и самостоятельное использование исправления Qwen остаются непроверенными. Общая цель активна.
