# HGB snapshot: стоимость и соответствие trial score

Это ресурсная проверка исправления из `5495572`, не новый quality benchmark. Baseline — экспорт `daf24c1f09ad3e4b151116b234f408a971088a52`, candidate — `61944302c3789bd3ff8f8e4f95f9a2863efe4100`. Отличающиеся исходники перечислены в `freeze.json`; в этих сценариях активен HGB-путь. Все package source SHA и измерительный скрипт оставались неизменными во время серии.

Фиксированный seed 2901; порядок на каждом workload: baseline → candidate → candidate → baseline. Два повтора одного workload измеряют разброс времени, а не независимый перенос качества. Три заранее заданных сценария: 50 000 × 32 regression, такая же regression с противоположным сигналом на inner holdout, 20 000 × 24 classification с десятью классами. Числовые признаки, 2% пропусков. Full refit использует на 25% больше строк; prediction — 1000 строк. Все данные синтетические.

Каждая пара включает два свежих интерпретатора: one-trial Optuna с выбором итераций и selected full fit, затем frozen full refit без нового поиска. Каждый процесс делает predict и pickle save/load с проверкой идентичности прогнозов. Parent wall включает запуск Python, imports, генерацию данных, обучение, inference, pickle и завершение. CPU seconds берутся из `getrusage`; peak RSS — максимум двух процессов, а не сумма. Потоки OMP/OpenBLAS/MKL/NumExpr ограничены одним. Budget 90 s и watchdog 120 s одинаковы для обеих сторон. Поле memory_mb=4096 передаётся learner, но этот probe не устанавливает OS RSS limit: отчёт содержит наблюдённый пик.

В этой серии **12 пар / 24 процесса**, все завершились. macOS, Python 3.12.7, scikit-learn 1.5.2. Перед серией проверена текущая загрузка: существующие ML-контейнеры показывали 0% CPU, единственный чужой Python относился к backend. Чужие процессы/контейнеры не менялись; полной изоляции хоста это не доказывает.

| Workload | Baseline, весь wall, с | Candidate, весь wall, с | Candidate peak RSS, MiB | Candidate свежий refit, с |
|---|---:|---:|---:|---:|
| Regression | 7.50–7.86 | 7.54–7.56 | 470.3–482.4 | 2.72–2.73 |
| Opposing regression | 3.97–3.98 | 3.96–3.97 | 455.6–501.1 | 1.51–1.55 |
| Ten classes | 6.72–6.77 | 6.71–6.72 | 277.2–287.3 | 2.29–2.30 |

Это диапазоны двух повторов, не доверительные интервалы и не доказательство ускорения/нулевой цены копирования. Время в этой небольшой серии не показывает заметного роста. Пик памяти колеблется; в opposing regression максимальный наблюдённый candidate RSS на 44.4 MiB выше максимального baseline. Нельзя приписать всю эту разницу только snapshot. Проверка не охватывает очень большие матрицы, тысячи итераций или всю конкурсную среду 8CPU/32GB/3h.

Независимый `tools/check_hgb_snapshot_cost.py` проверил hashes, полное число исходов, одинаковые frozen-рецепты и последовательности native fit (строки, запрошенные/обученные раунды) у всех повторов и обеих сторон. Ни одного дополнительного fit из-за восстановления snapshot не добавилось. Затем он заново построил inner fit по frozen-рецепту и отдельно пересчитал RMSE/Macro F1:

| Workload | Baseline trial score | Candidate trial score | Независимый frozen replay |
|---|---:|---:|---:|
| Regression | 0.808753 | 0.808753 | 0.808753 RMSE |
| Opposing regression | 4.244068 | 3.614434 | 3.614434 RMSE |
| Ten classes | 0.570854 | 0.577939 | 0.577939 Macro F1 |

Различие — исправление того, **какое состояние оценивается**, а не доказанный выигрыш модели на внешнем контроле: frozen-рецепты и выбранные раунды одинаковы (80, 10, 20). Старый trial score в двух случаях относился к последнему stale estimator, а не выбранному snapshot. Все candidate scores совпали с независимым replay до 1e-10. Target labels replay-evaluation строк не передавались в fit.

Один начальный запуск остановился из-за отсутствующего обязательного `task` в контексте нового измерительного скрипта. Он не содержит завершённых пар и не включён в таблицу; метаданные сохранены в `first_attempt.json`. Исправлена fixture скрипта, а не production worker. Полная серия после этого заморожена заново.

## Воспроизведение

Восстановите `automl_agent/` для обоих pinned commits через `git archive <commit> automl_agent` в два новых локальных каталога. Используйте существующую среду Python 3.12 / sklearn 1.5.2, не меняя зависимости проекта. Путь `/path/to/...` ниже — placeholder.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /path/to/python tools/measure_hgb_snapshot_cost.py --baseline /path/to/baseline-export --candidate /path/to/candidate-export --output outputs/new-hgb-cost-series
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /path/to/python -I tools/check_hgb_snapshot_cost.py --directory research/hgb_snapshot_cost_20261026 --source-root /path/to/candidate-export --baseline-source /path/to/baseline-export
```

`freeze.json`, `runs.json`, `state.json`, `summary.json`, `independent_checks.json` и frozen JSON-рецепты сохранены здесь. Обученные модели и process logs остаются в ignored outputs; в архив решения они не входят. `manifest.json` содержит SHA свидетельств и обоих скриптов.

Реальных API calls — **0**. Этот probe не включает controller, LLM/семантический разбор, реальный CSV I/O и закрытый контроль. Самостоятельность Qwen, итоговый RMSE/F1 организаторов и точная стоимость полного агента ещё не подтверждены. Общий goal остаётся активным. Production DL не расширяется; предыдущий отрицательный вывод RealMLP не изменяется этой ресурсной проверкой.
