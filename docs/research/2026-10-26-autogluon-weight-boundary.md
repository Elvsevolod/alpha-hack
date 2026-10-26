# Веса классов во вложенном AutoGluon: граница адаптера

## Подтверждённая проблема

Проверка продолжает требование автономного исследования: объявленный рецепт должен исполняться с тем же смыслом, иначе поиск сравнивает другое вмешательство. Общий allowlist прямых моделей нельзя считать контрактом нативных моделей AutoGluon.

В `3b37cd9` функция autogluon_portfolio проверяла каждую явную конфигурацию через validate_parameters прямой модели и передавала результат библиотеке. Однако AutoGluon 1.3.1 использует native LightGBM и кодирует метки до CatBoost. Синтетический офлайн probe использовал настоящий AutoGluon 1.3.1 / LightGBM 4.6.0 / CatBoost 1.2.8, 180 строк, 133 против 47 наблюдений двух классов, один CPU. Реальный LabelCleaner при positive_class=2 дал карту `10→0, 2→1`.

| Конфигурация из нашего portfolio | Наблюдение настоящего backend |
|---|---|
| GBM без class_weight | Обучение успешно |
| GBM class_weight="balanced" | Обучение успешно; предсказания побитно совпали с вариантом без весов, max abs difference 0 |
| GBM class_weight={"2":8,"10":1} | TypeError: Unknown type of parameter:class_weight, got:dict |
| CAT class_weights={"2":8,"10":1} | CatBoostError: Unknown class label: "0" после кодирования |

Побитное совпадение относится к этому probe; вывод о неподдерживаемом sklearn-параметре дополнительно основан на пути передачи params в native learner. Это не сравнение качества на задачах организаторов и не ресурсный benchmark. До исправления новые проверки: **3 failed / 1 passed**.

## Исправление

AutoGluon GBM class_weight и CAT class_weights-словари теперь отклоняются до fit с явным указанием direct/Optuna. Публичная схема объявляет это ограничение. Нативный CAT auto_class_weights и обычный bounded tree portfolio сохраняются. Другие существующие positional/native настройки этим этапом не сертифицируются. Версия реестра совместимости изменена с 3 на 4 для обновления кэшей справки.

Это исправление ложного обещания поддержки, **не реализация weighted AutoGluon**. Для последней нужен отдельный адаптер: веса строк для native LightGBM, точное соответствие внешних/внутренних меток для CAT, поддержка различающихся конфигураций и целей, проверки bagging/stacking, сохранения и fresh full refit. Передача исходного словаря или предположение о порядке меток этим требованиям не отвечает. Автоматического изменения рецепта нет; модель должна явно выбрать другой поддержанный рецепт.

## Проверки и границы

Офлайн HTTP-сценарии в native/JSON: отвергнутый run_automl → доставленная command_validation с direct/Optuna → явный train_model LightGBM → настоящий worker с проверенным class_weight="balanced" → finish/control/full refit CSV. Неверная операция не зарегистрирована, каждый сценарий использует ровно три scripted requests и сохраняет транспорт. Победитель control может быть baseline; этот тест подтверждает веса research worker и корректную финализацию, не обязательно refit взвешенного кандидата-победителя. Отдельные прежние проверки подтверждают веса в frozen direct/Optuna refit.

Наборы пересекаются, числа не суммируются:

- финальный набор boundary/recovery + reference names/memory после обновления реестра: **11 passed**;
- weight boundary + прежние direct/CatBoost weight checks: **54 passed**;
- recovery + boundary + source contract: **18 passed**;
- AutoGluon selection mocks + boundary: **7 passed**;
- default/explicit portfolio и реальный AutoGluon frozen refit на всех разрешённых строках: **5 passed, 21 deselected**. Два seeds здесь — технические проверки, не независимые подтверждения качества.

Первый новый test fixture ошибся в имени функции compatibility; следующие варианты — в tool (`train_model` вместо `run_automl`) и смешал operation_id с созданным experiment_id. Исходные отрицательные XML сохранены; это ошибки тестовых сценариев, не дополнительные product defects.

AutoGluon изначально отсутствовал в venv. Четыре разрешённых пакета версии 1.3.1 загружены с --no-deps в ignored outputs, venv/requirements не менялись. Для исходного AutoGluon try_import_catboost понадобился pkg_resources: в тот же изолированный каталог добавлен setuptools 75.8.2. Сначала наблюдался ImportError из-за окружения; итоговая ошибка CAT воспроизведена после устранения этой тестовой проблемы. Это частичная native development среда Python 3.12, не полная конкурсная Linux-среда.

Реальных Qwen/organizer API-запросов нет; snapshots alpha-api-limits для офлайн-серии не требуются. Production PyTorch/ZIP, данные и обученные веса поставки не менялись. Организаторские RMSE/Macro F1 и самостоятельный выбор Qwen этим исправлением не улучшены доказанным образом. Общая задача активна.

## Источники и повторение

Context7 использован для сверки общего контракта, но его выдача относится к master. Для выводов использованы реальные wheel 1.3.1 и закреплённые исходники: [native LGBModel](https://github.com/autogluon/autogluon/blob/1.3.1/tabular/src/autogluon/tabular/models/lgb/lgb_model.py), [CatBoostModel](https://github.com/autogluon/autogluon/blob/1.3.1/tabular/src/autogluon/tabular/models/catboost/catboost_model.py), [кодирование меток](https://github.com/autogluon/autogluon/blob/1.3.1/tabular/src/autogluon/tabular/learner/default_learner.py), [class_labels_internal_map](https://github.com/autogluon/autogluon/blob/1.3.1/tabular/src/autogluon/tabular/predictor/predictor.py).

[Свидетельство и SHA](../../research/autogluon_weight_audit_20261026/summary.json), исходный probe и наблюдения рядом. Для повторения отрицательного probe использовать automl_agent из `3b37cd9` в отдельном checkout, с теми же версиями пакетов и PYTHONPATH; на исправленных исходниках ошибочные рецепты уже отклоняются до backend. Probe не скачивает данные и не вызывает Qwen. Raw XML и локальные модели остаются в ignored outputs/autogluon_weight_audit_20261026.
