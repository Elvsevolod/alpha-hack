"""Convert reviewed research into generic, versioned project method cards offline."""
from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

# Public-task observations are research evidence, never runtime training hints.
OVERRIDES = {
    (2, "signals"): "Минимальный период test больше максимального периода train либо задача прямо прогнозная.",
    (4, "diagnostic"): "Подтвердить обе роли. Стратегия temporal_group исключает группы поздних оценочных блоков из ранних обучающих частей.",
    (4, "minimum_experiment"): "Использовать temporal_group с direct/Optuna; проверить непересечение групп и хронологию внешнего и каждого внутреннего split. FLAML/AutoGluon для совместной схемы пока не проверены.",
    (4, "constraints"): "Нельзя заменять совместный протокол одним group/temporal/random. Если после исключения групп части пусты, сообщить несовместимость и не ослаблять границы.",
    (6, "signals"): "Последовательный ID может кодировать порядок сбора; локальные ID разных файлов могут пересекаться.",
    (7, "diagnostic"): "Установить момент доступности и происхождение. Сильная корреляция не доказывает утечку; фактические экзогенные признаки допустимы только если это подтверждено task.",
    (7, "constraints"): "Не восстанавливать исключённые метки и недоступные outcomes; не отбрасывать сильный законный признак только из-за высокого score.",
    (14, "signals"): "Категория predict отсутствует на fit либо число уникальных значений растёт.",
    (21, "diagnostic"): "Читать словарь и допустимую область. Ноль может быть реальным измерением, а строка отсутствия услуги — законной категорией.",
    (25, "signals"): "Редкий класс представлен десятками примеров; на внутренней оценке может остаться несколько независимых случаев.",
    (26, "signals"): "Точные дубли вектора признаков имеют разные метки; причина может быть шумом или недостаточным контекстом.",
    (35, "constraints"): "Положительный удельный показатель не обязательно счётчик. Не округлять rate как count и не заменять RMSE девиансом.",
    (39, "constraints"): "Линейный тренд по анонимному коду периода не доказывает перенос. Для residual-метамодели нужны честные внутренние OOF/holdout прогнозы.",
    (40, "constraints"): "Не вычитать анонимный код периода из календарного года без соответствия шкал; не нормировать повторно уже удельный таргет и не менять единицы выхода.",
    (52, "constraints"): "Известная фактическая погода допустима в одной оценочной постановке и недопустима в другой прогнозной; следовать подтверждённому task.",
}

METHODS = ("task_contract period_transfer group_transfer group_transfer diagnostics task_contract task_contract period_transfer period_transfer rare_classes "
    "diagnostics weights period_transfer unknown_categories group_transfer smooth simple_imputation simple_imputation simple_imputation empty_columns "
    "simple_imputation task_contract task_contract weights rare_classes diagnostics threshold calibration resampling task_contract "
    "multioutput multioutput target_log target_log boosting linear linear diagnostics smooth diagnostics "
    "boosting target_encoding feature_selection linear text text ensemble smooth boosting forecast "
    "forecast forecast forecast ranking hpo budget budget ensemble diagnostics refit").split()

TAGS = ("task_contract|temporal|groups|temporal_group|duplicate_features|identifiers|leakage|label_delay|overlap_windows|rare_classes|"
    "distribution_shift|label_shift|concept_drift|unknown_categories|distribution_shift|extrapolation|missing|informative_missing|missingness_shift|empty_columns|"
    "sentinels|missing_targets|schema|class_imbalance|rare_classes|label_conflicts|multiclass_decision|calibration|resampling|unknown_class|"
    "multilabel|multioutput|heavy_tail|target_log|positive_target|zero_inflation|heteroskedasticity|outliers|extrapolation|units|"
    "high_cardinality|target_encoding|correlation|small_sample|sparse|text|mixed|cyclic|interactions|forecasting|"
    "panel|exogenous|intermittent|ranking|stagnation|pilot|resources|ensemble|reproducibility|refit").split("|")


def export(root=ROOT):
    cases = json.loads((root / "research/ml_best_practices_20261022/hard_cases.json").read_text())["cases"]
    destination = root / "ml_knowledge/catalogs/methods"
    assert len(cases) == len(METHODS) == len(TAGS) == 60
    for number, raw in enumerate(cases, 1):
        card = dict(raw)
        for (index, key), value in OVERRIDES.items():
            if index == number:
                card[key] = value
        text = json.dumps({k: card[k] for k in ("title", "signals", "diagnostic", "minimum_experiment", "constraints", "acceptance")}, ensure_ascii=False)
        if re.search(r"Worms|site_eui|Year_Factor|UNSW|year_built|facility_type|energy_star_rating|1458", text):
            raise ValueError("Public-specific observation in runtime card " + str(number))
        base = json.loads((destination / (METHODS[number - 1] + ".json")).read_text())
        identifier = "hard_case_ml" + str(number).zfill(2)
        obj = {"id": identifier, "method_id": identifier, "version": 1, "title": card["title"],
            "observations": card["signals"], "hypothesis": card["minimum_experiment"],
            "diagnostic": card["diagnostic"], "minimum_experiment": card["minimum_experiment"],
            "acceptance": card["acceptance"], "constraints": card["constraints"],
            "conditions": ["Confirmed task; fit-only learning; unchanged external protocol.", card["signals"]],
            "contraindications": [card["constraints"]], "cost": card["relative_cost"],
            "implementation_path": card["implementation_path"], "runtime_status": "unverified",
            "references": base["references"], "missing_references": [], "applicability_rules": [],
            "tags": [TAGS[number - 1]], "research_sources": card["source_urls"],
            "heuristic_status": "project_hypothesis_not_universal_rule",
            "evidence_policy": "Research synthesis, not measured quality. API references establish capabilities only; check pinned version and concrete adapter.",
            "routing_signal": TAGS[number - 1]}
        obj["routing_priority"] = 1 if number == 25 else 0
        if number in {4, 25, 27}:
            obj["implementation_path"] = "ready"
        if number == 25:
            obj["minimum_experiment"] = "Direct/Optuna PipelineSpec.fit_diagnostics='inner_holdout': один дополнительный fit с полным preprocessing только на внутреннем train; редкие подробности подавляются, внешние метки не используются. " + obj["minimum_experiment"]
        if number == 27:
            obj["minimum_experiment"] = "Для multiclass label-метрики сравнить default с prediction_policy='class_offsets', postprocess_fit_policy='inner_holdout'. Смещения обучаются внутри fit и фиксируются в frozen PipelineSpec перед control/refit."
            obj["constraints"] = "Бинарный threshold не решает multiclass. Class offsets разрешены только для выхода labels и поддерживаемой label-метрики; нельзя применять их к вероятностному выходу или обучать на внешних метках."
        obj["hypothesis"] = obj["minimum_experiment"]
        obj["contraindications"] = [obj["constraints"]]
        (destination / (identifier + ".json")).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    return {"imported": len(cases), "public_observations_embedded": False}


if __name__ == "__main__":
    print(json.dumps(export()))
