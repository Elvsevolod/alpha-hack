# Совместный runtime: исправления ML и контекст исследовательского цикла

Общая ML-цель остаётся активной. Предыдущие этапы проверяли качество CPU-рецептов, корректность исполнителя и его стоимость. Для проверки самостоятельного выбора сначала нужен единый исполняемый кандидат: наша ветка оставалась на старом system prompt, тогда как в соседней уже были исправлены известные потери feedback. Требования автономного исследования, одного кода для неизвестных задач, закрытого control и общего бюджета сопоставлены с ТЗ в `2026-10-26-agent-path-audit.md`.

## Состав и происхождение

В текущую экспериментальную ветку **выборочно** перенесены изменения из immutable `53304a0` (`feat/strong-reference-search`, относительно `1c463e9`): versioned prompt v6, decision frame, task-local память сопоставимых опытов, optional guides, сохранение verifier feedback после известного усечения, bounded parameter repair, XGBoost aliases и поддерживающие проверки. Перенесены исходник проектного Qwen guide и его generated module/catalog metadata; SOURCE_HASHES проверены. Это использование уже существующей работы с указанием происхождения, не новая разработка всех этих механизмов и не merge всей ветки.

Не импортировались исследовательские CSV, residual/group/pair calibrator modules, чужие журналы или submission bundles. Восемь ранее исправленных ML/DL-модулей byte-identical к `ee7ce01`: class weights, training schedules, HPO, completion identity, executor provenance, preprocessing и оба Torch-модуля. AST всего DirectPipeline тоже идентичен предыдущему; изменения pipelines.py — публичный тип ошибки и нормализация aliases из соседней ветки. Новых действий нет: **17**. Production DL не расширяется.

## Новый доказанный обрыв: настоящий command-validation путь

Соседняя проверка repair начиналась с прямого `Controller.dispatch` ошибочной команды. Новый составной тест начинает с ответа Qwen через настоящий HTTP client/parser/acceptance transaction. Он выявил три отличающихся звена:

1. `ContextMemory.validate` сначала вызывал общий compatibility aggregator, который превращал `UnsupportedParameters` в обычный ValueError. До dispatch команда уже не доходила.
2. Обработчик `Controller.run` для отклонённого ответа модели не добавлял parameter repair, хотя direct-dispatch обработчик его добавлял.
3. Даже после сохранения repair в metadata короткая snapshot-диагностика переносила только status/error/read_routes и теряла новую подсказку.

Оба транспорта завершали fixture штатным baseline CSV, но **не обучали исправленную гипотезу**. Новый тест отверг такой ложный успех: native и JSON дали 2 failures. После исправления первых двух звеньев оставались те же 2 failures — metadata ещё не означала доставку модели.

Теперь публичный тип ошибки сохраняется до compatibility aggregation; run-handler прикрепляет bounded repair; snapshot защищает его в current_work. Полная подсказка занимает не более 2048 bytes; существующий общий лимит current_work **4096 bytes не увеличен**. При тесном core сохраняется точный публичный маршрут к полному сообщению, включая fallback внутри прежней короткой диагностики. Никакой отклонённый ответ не объявляется выполненной операцией или обученной моделью; настройки автоматически не подменяются.

Адресный итог — **4 passed**: native/JSON цикл неправильное имя → фактически доставленные допустимые имена → CatBoost с числовыми JSON-весами → отдельный worker → full refit CSV на 25 строк; ещё два случая проверяют все четыре стадии сжатия в обоих транспортах, включая переполненный гипотезой core и точный read fallback. Private value sentinel не попадает в следующий запрос; отклонённая команда не появляется как выполненная операция. Исправление в mock зависит от действительно полученной подсказки. Это техническая политика имитации, не независимое решение настоящего Qwen.

## Готовность среды и проверки

Первый сборочный запуск тестов остановился из-за не перенесённого supporting `tools/context_structure_audit.py`; он добавлен из того же snapshot. Затем 78 tests прошли, четыре проверки справки упали с `integrity_error`: SQLite в этом worktree оказался **134-byte LFS pointer**, а не документной базой. Это дефект подготовки dev-среды, не неправильный hash guard. Через `git lfs checkout` восстановлен существующий локальный объект без сети: **281751552 bytes**, SHA `06ca0504692d9f00696f50bec19e413f9d0e59dbad136eaf3aa57d25932302db`, совпадает с lock. Git-версия корпуса не менялась; ReferenceStore.verify с requirements прошёл.

После восстановления **77 passed** в наборе public reads, parameter repair, semantic feedback, finish/control/refit и API contracts. До нового исправления реального цикла связанный ML/DL набор дал **110 passed / 2 failed** (оба новых сквозных кейса); эти отказы сохранены. На окончательном исправлении legacy/unknown/truncation/permission guards дали **24 passed**. После окончательного исправления весь составной ML/DL набор повторён: **114 passed / 0 failed**. Он включает старые проверки class weights, CatBoost, HGB schedules, completion/cache identity, HPO error delivery и опционального Torch-пути. Пересекающиеся счётчики не суммируются.

Generated runtime guide связан с обновлённым source hash. Скомпилированный корпус сохранён в прежней версии `491827b5…`: у пяти human-practice cards сравнение нашло только различия source-hash metadata и compiler-added supporting_sections; текст, условия и ограничения совпадают. Это не повод переписывать неподвижный корпус. Новые утверждения о фактически доставленных карточках не выводятся из изменения source catalogs.

Список выбранных путей, происхождение, SHA исходников, corpus readiness и XML checks — `research/agent_path_audit_20261026/context_runtime_integration.json`. Оригинальные отрицательные XML остаются в ignored outputs.

Реальных organizer model calls — **0**; лимиты не запрашивались, остаток неизвестен. alpha-api-limits понадобится перед фактической живой серией и после её завершения/отмены. Пока проведены offline и HTTP mock проверки; новое RMSE/Macro F1 организаторов, независимый transfer, стоимость полного агента и самостоятельность Qwen не доказаны. Результаты старого A/B нельзя численно переносить на этот совмещённый runtime. Main, конкурсный ZIP, frozen исследования и чужие процессы не менялись. Общий goal не завершён.
