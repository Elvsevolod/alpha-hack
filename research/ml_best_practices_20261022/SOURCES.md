# Источники исследования ML: 22 октября 2026 года

Использованы первичные источники: официальная документация и авторские публикации. Ниже указана граница доказательства: описание API подтверждает возможность вызова, но не преимущество качества. Рекомендации в справочнике — проектные гипотезы, которые нужно проверять на fit и неизменном внешнем протоколе.

Для scikit-learn открыта именно документация **1.5.2**, для LightGBM — **4.6.0**, AutoGluon — **1.3.1**, Optuna — **4.2.1**; исходники XGBoost и PyTorch закреплены тегами **v2.1.4** и **v2.5.1**. Context7 использован для предварительной проверки. Его выдача `stable` содержит более новые API: например, `GroupKFold(shuffle=...)` появился после 1.5.2; у `TargetEncoder` в 1.5.2 `cv` — целое число. Поэтому примеры из текущей выдачи Context7 нельзя переносить в стенд без проверки точной версии.

У CatBoost веб-страницы не закреплены версией: сигнатуры сверять с локальным корпусом 1.2.8. У imbalanced-learn открылись страницы 0.14.2; они используются **только для методологических ограничений**, а API для исполнения следует брать из локального корпуса 0.12.4. Несколько запросов к старым страницам imbalanced-learn, странице PyTorch 2.5 и странице XGBoost release_2.1 завершились ошибками; эти неудачные обращения не считаются подтверждёнными источниками. Для PyTorch/XGBoost найдены официальные исходники точных тегов.

## S01 — Утечки и согласованная предобработка

[scikit-learn 1.5.2: Common pitfalls](https://scikit-learn.org/1.5/common_pitfalls.html). Подтверждает необходимость обучать преобразования только на обучающей части. Не доказывает независимость строк конкретного датасета.

## S02 — Выбор схемы валидации

[scikit-learn 1.5.2: Cross-validation](https://scikit-learn.org/1.5/modules/cross_validation.html). Случайная, стратифицированная, групповая и временная проверки отвечают на разные вопросы переноса.

## S03 — Семантика метрик

[scikit-learn 1.5.2: Model evaluation](https://scikit-learn.org/1.5/modules/model_evaluation.html). Направление оптимизации, усреднение и тип предсказания должны соответствовать заданию.

## S04 — Калибровка

[scikit-learn 1.5.2: Probability calibration](https://scikit-learn.org/1.5/modules/calibration.html). Для калибровки нужны предсказания на независимых от обучения модели наблюдениях. Калибровка не гарантирует роста F1.

## S05 — Порог решения

[scikit-learn 1.5.2: Classification threshold](https://scikit-learn.org/1.5/modules/classification_threshold.html). Различает обучение вероятностей и правило принятия решения; предупреждает о настройке порога на данных обучения модели.

## S06 — Кодирование таргетом

[TargetEncoder 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.TargetEncoder.html). `fit_transform` использует cross-fitting, `fit().transform()` отличается. Обычная встроенная схема не решает автоматически задачу группового или временного кодирования.

## S07 — Ошибки ресемплинга

[imbalanced-learn: Common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html). Ресемплинг до разбиения создаёт утечку и меняет распределение оценки. Страница 0.14.2; API 0.12.4 ею не подтверждается.

## S08 — Категории CatBoost

[CatBoost: Categorical features](https://catboost.ai/en/docs/features/categorical-features). Рекомендуется использовать предусмотренную обработку категорий вместо внешнего one-hot перед CatBoost; действуют ограничения режима обучения. Страница без точного pin.

## S09 — Категории, NaN и ranking LightGBM

[LightGBM 4.6.0: Advanced topics](https://lightgbm.readthedocs.io/en/v4.6.0/Advanced-Topics.html). Описывает обработку категориальных признаков, пропусков и LambdaRank. Наличие опции в библиотеке не подтверждает поддержку нашим адаптером.

## S10 — Ограничение сложности бустинга

[LightGBM 4.6.0: Parameters tuning](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters-Tuning.html). Связь `num_leaves`, глубины, минимального размера листа и стоимости обучения; ранняя остановка требует отдельной внутренней выборки.

## S11 — Категории XGBoost точной версии

[XGBoost v2.1.4: categorical.rst](https://raw.githubusercontent.com/dmlc/xgboost/v2.1.4/doc/tutorials/categorical.rst). Подтверждает особенности экспериментальной категориальной поддержки и сохранения модели. Не переносить автоматическую перекодировку из новых версий.

## S12 — AutoML и стоимость финализации

[AutoGluon 1.3.1: Tabular in depth](https://auto.gluon.ai/1.3.1/tutorials/tabular/tabular-indepth.html). Bagging, stacking и refit имеют разные вычислительные и валидационные последствия. Наш адаптер предоставляет более узкий набор возможностей.

## S13 — Optuna

[Optuna 4.2.1: Efficient optimization algorithms](https://optuna.readthedocs.io/en/v4.2.1/tutorial/10_key_features/003_efficient_optimization_algorithms.html). Sampler и pruner решают разные части поиска; pruning требует сопоставимых промежуточных результатов.

## S14 — Воспроизводимость

[PyTorch v2.5.1: randomness.rst](https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/docs/source/notes/randomness.rst). Одинаковый seed не гарантирует совпадения между платформами, версиями или CPU/GPU.

## S15 — Бустинги как сильная табличная опора

[Grinsztajn et al., 2022](https://arxiv.org/abs/2207.08815). Эмпирическое сравнение на определённом наборе табличных задач; не универсальное доказательство превосходства деревьев.

## S16 — Зависимость выбора модели от данных

[McElfresh et al., 2023, редакция 2024](https://arxiv.org/abs/2305.02997). Сравнение 19 алгоритмов на 176 датасетах; показывает значение свойств данных и умеренной настройки. Упомянутые pretrained-методы не разрешены нашим кейсом.

## S17 — Риск переобучения выбора модели

[Cawley & Talbot, 2010](https://jmlr.csail.mit.edu/papers/v11/cawley10a.html). Даже оценочный критерий можно переобучить многочисленными попытками. Малый выигрыш на многократно используемой validation не равен доказательству обобщения.

## S18 — Поиск гиперпараметров

[Bergstra & Bengio, 2012](https://www.jmlr.org/papers/v13/bergstra12a.html). Random search — содержательная опора для ограниченного поиска, особенно при разной важности параметров. Не доказывает, что он всегда лучше TPE.

## S19 — Сдвиг долей классов

[Lipton, Wang & Smola, 2018](https://proceedings.mlr.press/v80/lipton18a.html). BBSE предполагает устойчивость `p(x|y)` и подходящие свойства confusion matrix; произвольный domain shift не удовлетворяет этим условиям автоматически.

## S20 — Честная проверка прогнозов и обратное преобразование

[Hyndman & Athanasopoulos: Forecaster's toolbox](https://otexts.com/fpp3/toolbox.html). Rolling-origin проверка, соответствие горизонта и bias adjustment после преобразований. Ссылки старых разделов перенаправляются на эту главу. Примеры R не являются исполнимыми рецептами нашего агента.

## S21 — Пропуски

[scikit-learn 1.5.2: Imputation](https://scikit-learn.org/1.5/modules/impute.html). Простые и итеративные подходы, индикаторы и пустые признаки. Документация не обещает роста качества от сложного заполнения.

## S22 — Деревья, ансамбли и ограничения экстраполяции

[scikit-learn 1.5.2: Ensembles](https://scikit-learn.org/1.5/modules/ensemble.html). Свойства ансамблей и поддерживаемых objectives. Выбор альтернативы при переносе за диапазон — проектная гипотеза.

## S23 — Аномалии

[scikit-learn 1.5.2: Outlier detection](https://scikit-learn.org/1.5/modules/outlier_detection.html). Различает поиск аномалий и novelty detection; аномальность не доказывает ошибочность записи.

## S24 — Полноценная диагностика классов

[precision_recall_fscore_support 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.precision_recall_fscore_support.html). Precision, recall и F1 каждого класса вычисляются с учётом ошибок на всей оцениваемой выборке.

## S25 — Группы и дисбаланс

[StratifiedGroupKFold 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html). Пытается совместить непересекающиеся группы и пропорции классов; баланс может быть ограничен структурой групп.

## S26 — Разреженный текст

[TfidfVectorizer 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). Словарь и IDF обучаются на fit; параметры ограничивают размер представления.

## S27 — Stacking

[StackingClassifier 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.StackingClassifier.html). Для метамодели используются out-of-fold прогнозы; обучение на in-sample прогнозах через `prefit` несёт высокий риск переобучения.

## S28 — Macro F1

[f1_score 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.f1_score.html). Macro усредняет F1 классов без весов по числу примеров; политика отсутствующих классов и `zero_division` должна быть явной.

## S29 — RMSE

[root_mean_squared_error 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.root_mean_squared_error.html). Оценка зависит от квадратов ошибок в исходных единицах и заданных весов.

## S30 — Время и gap

[TimeSeriesSplit 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). Упорядоченные разбиения, `gap`, ограничение истории; для нерегулярного времени смысл горизонта требует дополнительной проверки.

## S31 — Индикаторы пропуска

[MissingIndicator 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.impute.MissingIndicator.html). Фиксированная схема и поведение признаков, у которых пропуски появляются после fit.

## S32 — Неотрицательный счётный отклик

[PoissonRegressor 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.PoissonRegressor.html). Пуассоновская GLM с log-link; это не универсальная замена регрессии положительных величин.

## S33 — Квантили

[QuantileRegressor 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.QuantileRegressor.html). Оценивает условный квантиль; медиана и среднее отвечают разным функциям потерь.

## S34 — Устойчивое масштабирование

[RobustScaler 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.RobustScaler.html). Использует устойчивые статистики; центрирование sparse-матрицы может потребовать недопустимого уплотнения.

## S35 — Oversampling

[imbalanced-learn: Over-sampling](https://imbalanced-learn.org/stable/over_sampling.html). Random oversampling, SMOTE и смешанные типы требуют разных предпосылок. Страница 0.14.2; исполнимые вызовы сверять с 0.12.4.

## S36 — Гладкие зависимости

[SplineTransformer 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.SplineTransformer.html). Нелинейный базис, параметры узлов и варианты экстраполяции.

## S37 — Неизвестные и редкие категории

[OneHotEncoder 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OneHotEncoder.html). `handle_unknown`, редкие категории и ограничение размерности; словарь не нужно доучивать на тесте.

## S38 — Ranking-метрика

[ndcg_score 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.ndcg_score.html). Оценивает упорядочивание релевантностей; набор запросов, cutoff и gain должны соответствовать заданию.

## S39 — Гистограммный бустинг

[HistGradientBoostingRegressor 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html). Поддержка NaN и ограничения категориального представления. Проверка адаптера остаётся отдельной.

## S40 — Групповое разбиение точной версии

[GroupKFold 1.5.2](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GroupKFold.html). Сигнатура содержит `n_splits`, без новых `shuffle` и `random_state`.

## S41 — Механизм CatBoost

[Prokhorenkova et al.: CatBoost](https://arxiv.org/abs/1706.09516). Ordered boosting и работа с target statistics предназначены для уменьшения определённых видов prediction shift. Они не отменяют внешнюю валидацию по времени или группам.

## S42 — Сравнение табличных нейросетей

[Gorishniy et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html). Сравнение при общем протоколе, отсутствие универсального победителя. Это исследовательский контекст, а не разрешение добавлять зависимости или pretrained-веса.

## S43 — Несколько целей и multilabel

[scikit-learn 1.5.2: Multiclass and multioutput](https://scikit-learn.org/1.5/modules/multiclass.html). Различие постановок и форм выходов; строковая метка одного класса не заменяет multilabel-матрицу.

## S44 — Предобработка признаков

[scikit-learn 1.5.2: Preprocessing](https://scikit-learn.org/1.5/modules/preprocessing.html). Масштабирование, нелинейные преобразования и кодирование. Параметры представления выбираются внутри fit.

## S45 — Современная общая оценка табличных методов

[TabArena, 2025](https://arxiv.org/abs/2506.16791). Живой бенчмарк показывает влияние валидации, бюджета и ансамблирования; при больших бюджетах нейросети конкурентоспособнее, чем следует из категоричного лозунга о превосходстве деревьев. Изучена публикация, а не актуальный полный leaderboard на 22.10.2026; перенос рангов на наш CPU-стенд не установлен.

## S46 — Эффективные ансамбли табличных MLP

[TabM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html). Предлагает parameter-efficient ensembling для табличных MLP. Исследован научный результат; точная реализация на PyTorch 2.5.1 и время CPU-обучения в нашем агенте не проверены. Готового TabM-адаптера у нас нет.

## S47 — Сильные стартовые настройки

[Better by Default / RealMLP, 2024](https://arxiv.org/abs/2407.04491). Сильные meta-tuned defaults для MLP и бустингов могут давать конкурентный результат без дорогостоящего HPO на каждом датасете. Это не pretrained-веса, но соответствующие внешние Python-пакеты не входят автоматически в наш allowlist; возможна только собственная проверенная реализация на разрешённых зависимостях.

## Локальные первичные материалы

`data_for_participants/requirements.txt`, обе `task.md`, CSV и словари; текущие исходники контроллера, валидации, предобработки и базы знаний; 32 карточки методов, 20 рецептов, 8 пространств поиска; два проектных `SKILL.md`.

Точные хеши ключевых входов — в [inputs.sha256.json](inputs.sha256.json). Полные агрегаты публичных данных — в [public_data_profile.json](public_data_profile.json); их нельзя подмешивать в fit-only контекст нового runtime-прогона. Эксперимент поиска по существующей базе — в [retrieval_probe.json](retrieval_probe.json). Это четыре поисковых запроса, а не измерение полноты или качества агента.
