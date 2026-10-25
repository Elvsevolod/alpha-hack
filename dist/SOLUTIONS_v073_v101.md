# Сохранённые решения v073–v101

В этой серии 29 неизменённых номерных ZIP. Последняя собранная версия — **v101**:
M1–M8 реализованы поверх v096, A25/A27 и temporal OOF сохранены, все 28 A-пунктов перепроверены.
Исходники в репозитории побайтово сверены со всеми 80 файлами runtime внутри v101.

Архивы поставляются через Git LFS. После клонирования выполните `git lfs pull`.
Для новой плоской сборки используйте `tools/build_flat_submission.py` с
`--knowledge-archive dist/solution_v101.zip` и новым каталогом `--output`.

## Проверки v101

- Общая серия: 1372 passed, 2 skipped, 1 failed. Единственный сбой — запрет macOS на
  остановку группы процессов при подготовке теста бюджета 60/5/5. Неизменённый тест
  отдельно прошёл на хосте и из точного ZIP в Linux; оба пропуска также повторены в Linux.
  Это отдельные проверки, а не одна непрерывная зелёная серия.
- Добавлены 92 тестовых сценария: 78 для активных изменений и 14 для сохранённой
  отрицательной K−1 гипотезы. Финальная целевая серия — 93 passed.
- В закреплённом Docker-образе без сети проверены 88 технических сценариев точного ZIP,
  включая native fits, инструментированные и структурные проверки; совпали все 37 зависимостей.
- Четыре чистых CLI-запуска с локальным HTTP mock завершили control/refit и проверку CSV.
- M4: основная K−1 гипотеза отклонена; отдельная All-K альтернатива прошла фиксированный
  acceptance и независимое синтетическое подтверждение. Параметры после подтверждения не подбирались.
- Реальный Qwen и платформа в этой серии не запускались. Преимущество полного агента
  на публичных или скрытых задачах не установлено.

Полные логи, временные модели, данные и аналитические пакеты оставлены локально.
Оригинальные JSON-метаданные рядом с архивами сохранены; их абсолютные локальные пути
обозначают происхождение проверки и могут быть недоступны после клонирования.

## Версии

Промежуточные и не прошедшие приёмку версии сохранены для истории. В частности,
v073, v075, v082, v083, v097, v099 и v100 не следует выбирать вместо финальной v101.
Наличие ZIP и корректного манифеста само по себе не означает приёмку качества.

| Версия | Байт | SHA-256 | Сохранённый статус / проверка |
| --- | ---: | --- | --- |
| [v073](solution_v073.zip) | 95071082 | `c8a0c81237a1352be66f24daf88133cbe18c5982e3ecf95a934aafc87c958291` | saved_version_no_separate_acceptance_record |
| [v074](solution_v074.zip) | 95071414 | `6e2200131519e02c1feac344dd1f90bb1b4f0b684aab2fa3d95068809798ff56` | [solution_v074.acceptance.json](solution_v074.acceptance.json) |
| [v075](solution_v075.zip) | 95111267 | `880574dbb6083a6411f3f93d6bc9ed499f66a95f5e4449e26a1a94a0dec9a73f` | [solution_v075.acceptance.json](solution_v075.acceptance.json) |
| [v076](solution_v076.zip) | 95111293 | `7b127b25cecbd6dcfba59e767c01d92d65832de80e1cf61ac3071d215b94fa47` | [solution_v076.acceptance.json](solution_v076.acceptance.json) |
| [v077](solution_v077.zip) | 71231263 | `091cb4b865ffe31141ea1dba62248905c9b3a2eb9700bcee3159115fa2b7c85c` | saved_version_no_separate_acceptance_record |
| [v078](solution_v078.zip) | 70660457 | `3b4ba864cc458dd816c9c0feb3550178ff9d27f35837a521c6a1cecbb2f2a2e8` | saved_version_no_separate_acceptance_record |
| [v079](solution_v079.zip) | 70663066 | `5b676b154a1b50f79f9876a841842fe5633f902329d012ebb0a5e6a13b5bc107` | [solution_v079.classical_acceptance.json](solution_v079.classical_acceptance.json) |
| [v080](solution_v080.zip) | 70664151 | `5e71d51df3e05c64a1a2d2033f49b8a6c948d824ab793f67c3ae4d0c35404ab6` | [solution_v080.interface_acceptance.json](solution_v080.interface_acceptance.json) |
| [v081](solution_v081.zip) | 70670250 | `148c65c16dc3d54ee3a12e37fb48d577fdf33442a8654fcd348a338b9b3e8af2` | [solution_v081.postrun_acceptance.json](solution_v081.postrun_acceptance.json) |
| [v082](solution_v082.zip) | 70697985 | `1ce825e76cb6211537fc2a87f14b58936183a703ae0e6a656a2ef0cbd39a4ef7` | [solution_v082.local_validation.json](solution_v082.local_validation.json) |
| [v083](solution_v083.zip) | 70698043 | `3c3b5053ee08c6be55eebe0ce1749defaa7b9d4198b4cf9c790ad25628717d86` | [solution_v083.local_validation.json](solution_v083.local_validation.json) |
| [v084](solution_v084.zip) | 70698151 | `be5e38876d15a310af3d84d96780dcabb46cce586678598096a2568947d52b86` | [solution_v084.local_acceptance.json](solution_v084.local_acceptance.json) |
| [v085](solution_v085.zip) | 70700536 | `d6169dcc26f0a2b6271d825bc4f04a125d65979b9d21ccaeb0a6dbd60c6561d9` | saved_version_no_separate_acceptance_record |
| [v086](solution_v086.zip) | 70700621 | `3f1d98379f6364187b822a6b0f0d99d96838c3085edd6e2309365c0ae94cc113` | [solution_v086.local_acceptance.json](solution_v086.local_acceptance.json) |
| [v087](solution_v087.zip) | 70711500 | `35fa04ee1d47d3793ea594255f160594c70226922e12d888de160a82b3be5bd9` | [solution_v087.local_acceptance.json](solution_v087.local_acceptance.json) |
| [v088](solution_v088.zip) | 70713935 | `1074af2d939d4f64ff21c3b39346758cb365ff8ac312992e8ce3ad5ba390ca8e` | saved_version_no_separate_acceptance_record |
| [v089](solution_v089.zip) | 70716085 | `9a827176229dcdec41cc88714896290c06030dae2b368dc263cfcf99bb807e53` | saved_version_no_separate_acceptance_record |
| [v090](solution_v090.zip) | 70716350 | `76822c84b7d2dc2009063218a6928a18805c6dd323f2261c7f233148b7f88213` | saved_version_no_separate_acceptance_record |
| [v091](solution_v091.zip) | 70716383 | `db1a2d3ac7762d2f5756d7ba72d72eb1d9be19f123a187272d8eee18e4c7f009` | saved_version_no_separate_acceptance_record |
| [v092](solution_v092.zip) | 70726737 | `b4018869540ca6374c1b90ddc28d983256a8926f86effe6f31fae477453e88f6` | saved_version_no_separate_acceptance_record |
| [v093](solution_v093.zip) | 70726806 | `047cc3a05ab7bcc3cf61e3df2365cb733cccf7052647659bd3947dc95c3fbe03` | [solution_v093.audit_acceptance.json](solution_v093.audit_acceptance.json) |
| [v094](solution_v094.zip) | 70727414 | `50fd0dc79781927393a5000e07c9516a4d0fdd2de3c79e1b25b63249e5437b90` | saved_version_no_separate_acceptance_record |
| [v095](solution_v095.zip) | 70727494 | `63b0615cf1544311661a3ec11c154596cac6b67863e51fcbf44d8ac5c2b21dbe` | saved_version_no_separate_acceptance_record |
| [v096](solution_v096.zip) | 70727569 | `7049767e5a1fb40b19a3eb27c7776b0d94329df68f2f63f184b4125417af9ce6` | [solution_v096.audit_acceptance.json](solution_v096.audit_acceptance.json) |
| [v097](solution_v097.zip) | 70732266 | `410b62da80ff9f21fb3125dc5d556a4df223942ce27d88846c131f08a31d5627` | [solution_v097.audit_acceptance.json](solution_v097.audit_acceptance.json) |
| [v098](solution_v098.zip) | 70732325 | `c47ceea7e5bc7da8b17fce305b35a3c8ca462b366ee7ce7cc520569e1bdc3cd6` | [solution_v098.audit_acceptance.json](solution_v098.audit_acceptance.json) |
| [v099](solution_v099.zip) | 70733048 | `52e5b2c3c83088965b56511ae45c18c0ee917cffa053faa41a2983f4d23ecf5f` | [solution_v099.audit_acceptance.json](solution_v099.audit_acceptance.json) |
| [v100](solution_v100.zip) | 70733357 | `50a4465067f38eff1eff0b8043988c501b830f7bd1ea9b63c20fa63cfce0176e` | [solution_v100.audit_acceptance.json](solution_v100.audit_acceptance.json) |
| [v101](solution_v101.zip) | 70733306 | `f283ebf58c35d789d1fb09de6598364f99da4e626e7d246cddf2b418ef7535aa` | [solution_v101.audit_acceptance.json](solution_v101.audit_acceptance.json) |
