from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import DataProfile, MetricSpec, OutputSpec, TaskResolutionProposal, TaskSpec
from .util import digest, file_hash, inside


class ContractUnresolved(ValueError):
    pass


def csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    if not header or len(header) != len(set(header)):
        raise ValueError(f"Empty or duplicate CSV header: {path.name}")
    return header


def locate_csv(task_path: Path, text: str, name: str, required: bool = True) -> Path | None:
    root = task_path.parent
    references = re.findall(r"[A-Za-zА-Яа-яЁё0-9_./ -]*\b" + re.escape(name), text)
    existing: set[Path] = set()
    for reference in references:
        reference = reference.strip(" `\n-\t")
        try:
            candidate = inside(root, reference)
        except ValueError:
            continue
        if candidate.is_file():
            existing.add(candidate)
    if not existing:
        existing = {p.resolve() for p in root.rglob(name) if ".agent_work" not in p.parts and ".venv" not in p.parts and p.resolve().is_relative_to(root.resolve())}
    if len(existing) == 1:
        return existing.pop()
    if not existing and not required:
        return None
    raise ContractUnresolved(f"Expected one unambiguous {name}, found {len(existing)}")


def read_csv(path: Path) -> pd.DataFrame:
    csv_header(path)
    # Only empty cells are missing. Literal NA, '-', and zeros retain their meaning.
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8-sig")


def numeric_if_possible(series: pd.Series) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    observed = series.notna()
    return converted if converted[observed].notna().all() else series


def parse_metric(text: str) -> MetricSpec | None:
    aliases = [
        (r"\bRMSE\b", "rmse", "macro"), (r"\bMSE\b", "mse", "macro"),
        (r"\bMAE\b", "mae", "macro"), (r"\bR[²2]\b", "r2", "macro"),
        (r"\bMAPE\b", "mape", "macro"),
        (r"balanced[ _-]*accuracy", "balanced_accuracy", "macro"),
        (r"macro[ _-]*f1|f1[ _-]*macro", "f1", "macro"),
        (r"weighted[ _-]*f1|f1[ _-]*weighted", "f1", "weighted"),
        (r"micro[ _-]*f1|f1[ _-]*micro", "f1", "micro"),
        (r"\baccuracy\b", "accuracy", "macro"),
        (r"roc[ _-]*auc", "roc_auc", "macro"),
        (r"log[ _-]*loss", "log_loss", "macro"),
        (r"average[ _-]*precision", "average_precision", "macro"),
    ]
    # Metric sections precede descriptions of secondary diagnostics in the template.
    section = re.search(r"(?:Основная метрика|Метрика оценки|Primary metric|Evaluation metric)[^\n]*", text, re.I)
    candidates = []
    for pattern, name, average in aliases:
        match = re.search(pattern, section.group() if section else text, re.I)
        if match:
            candidates.append((match.start(), name, average))
    ranking = re.search(r"\b(NDCG|MAP)\s*@\s*(\d+)", section.group() if section else text, re.I)
    if ranking:
        return MetricSpec(name=ranking[1].lower(), k=int(ranking[2]))
    if not candidates:
        return None
    _, name, average = min(candidates)
    return MetricSpec(name=name, average=average)


def _task_instruction_scope(name: str) -> str:
    """Mark specifically named official documents as protocol-authoritative."""
    path = Path(name)
    stem = re.sub(r"[ _-]+", "_", path.stem.casefold())
    authoritative_stems = {
        "task_description", "task_instructions", "task_rules", "organizer",
        "organizer_instructions", "organizer_rules", "official_instructions",
        "official_rules", "challenge_rules",
        "условие_задачи", "задание_организатора", "инструкция_организатора", "правила_организатора",
    }
    authoritative_dirs = {"organizer", "official", "instructions", "организатор"}
    if stem in authoritative_stems or any(part.casefold() in authoritative_dirs for part in path.parts[:-1]):
        return "organizer_instruction"
    return "supporting_material"


def _protocol_is_direct(clause: str, match: re.Match) -> bool:
    """Attach a directive to the method phrase instead of any word in the sentence."""
    if match.start() < 24 and re.match(
        r"^\s*(?:please\s+)?(?:split|keep|ensure|prefer|recommend|использ[а-я]*|примен[а-я]*|провод[а-я]*|разб[а-я]*|раздел[а-я]*|выбира[а-я]*|сохраня[а-я]*|обеспеч[а-я]*|предпочт[а-я]*|рекоменду[а-я]*)\b",
        clause,
        re.I,
    ):
        return True
    prefix = clause[:match.start()]
    if re.search(
        r"(?:^|[:,;])\s*(?:please\s+)?(?:use|choose|select|apply|perform|adopt|follow|require|requires|split|keep|ensure|prefer|recommend|использ[а-я]*|примен[а-я]*|провод[а-я]*|разб[а-я]*|раздел[а-я]*|выбира[а-я]*|сохраня[а-я]*|обеспеч[а-я]*|предпочт[а-я]*|рекоменду[а-я]*)"
        r"(?:\s+(?:a|an|the|chronological|temporal|time[- ]based|group[- ]based|random|future|validation|split|and|validation|выборку|групповую|временную|случайную|на|по)){0,4}\s*$",
        prefix,
        re.I,
    ):
        return True
    suffix = clause[match.end():match.end() + 45]
    return bool(re.match(
        r"\s*(?:(?:is|are)\s+(?:required|recommended|prescribed)|(?:must|should|shall)\s+be\s+(?:used|applied|followed|performed)|follows\s+(?:the\s+)?past\s+training\s+periods?)\b"
        r"|\s*(?:должн[а-я]*\s+быть\s+(?:использ|примен|провед)[а-я]*|следует|необходимо|требуется|рекомендован[а-я]*)",
        suffix,
        re.I,
    ))


def _protocol_is_negated(clause: str, match: re.Match) -> bool:
    """Recognize prohibitions both before and after the named protocol."""
    prefix = clause[:match.start()].rstrip("`\"' ") + " "
    if re.search(
        r"\b(?:do\s+not|don't|never|avoid|mustn't|must\s+not|shouldn't|should\s+not|cannot|can't|prohibit|forbid)\s+(?:(?:use|using|choose|select|apply|applying|perform|performing|adopt|adopting|recommend|recommending)\s+)?(?:a\s+|an\s+|the\s+)?$"
        r"|\b(?:prohibited|forbidden|banned|disallowed)\s+from\s+(?:(?:using|applying|choosing|selecting|performing|adopting)\s+)?(?:a\s+|an\s+|the\s+)?$"
        r"|нельзя\s+(?:использ|примен|провод|выбира|рекоменд)[а-я]*[^.!?;\n]{0,25}$"
        r"|не\s+(?:использ|примен|провод|выбира|рекоменд)[а-я]*[^.!?;\n]{0,25}$",
        prefix,
        re.I,
    ):
        return True
    suffix = clause[match.end():].lstrip("`\"' ")[:55]
    suffix = " " + suffix
    if re.search(r"\bno\s+(?:(?:a|an|the)\s+)?$", prefix, re.I) and re.match(
        r"\s*(?:(?:must|should|shall|may|can)(?:\s+ever)?\s+be\s+(?:used|applied|selected|performed|recommended)|(?:is|are)\s+(?:allowed|permitted|recommended|required))\b",
        suffix,
        re.I,
    ):
        return True
    return bool(re.match(
        r"\s*(?:(?:(?:must|should|shall|may|can)\s+(?:not|never)|mustn't|shouldn't|cannot|can't)\s+be\s+(?:used|applied|selected|performed|recommended)|(?:must|should|shall|may|can)\s+be\s+avoided|is\s+not\s+(?:used|applied|recommended|required|suitable|appropriate|allowed|permitted)|not\s+(?:used|recommended|suitable|appropriate|allowed|permitted)|(?:is|are)\s+(?:prohibited|forbidden|disallowed|banned|discouraged)|(?:isn't|aren't)\s+(?:permitted|allowed|recommended|discouraged|suitable|appropriate|acceptable))\b"
        r"|\s*(?:не\s+(?:должн[а-я]*\s+использ[а-я]*|использоваться|рекомендован[а-я]*|рекоменд[а-я]*|подход[а-я]*|допустим[а-я]*)|запрещен[а-я]*|(?:использ|примен|провод|выбира)[а-я]*\s+(?:нельзя|запрещен[а-я]*))",
        suffix,
        re.I,
    ))


def _explicit_validation_supported(strategy: str, text: str) -> bool:
    """Require a cited instruction to prescribe a split, not merely describe one."""
    signals = set()
    denied_signals = set()
    alternative_pattern = re.compile(
        r"(?:group(?:[- ]based)?|random|chronological|temporal|time[- ]based)[^.!?;\n]{0,35}\b(?:or|либо|или)\b[^.!?;\n]{0,35}(?:group(?:[- ]based)?|random|chronological|temporal|time[- ]based)"
        r"|(?:group(?:[- ]based)?|random|chronological|temporal|time[- ]based)[^.!?;\n]{0,35}\b(?:or|либо|или)\b[^.!?;\n]{0,35}(?:validation|split|hold[- ]?out)",
        re.I,
    )
    clauses = re.split(r"(?<=[.!?;])\s+|\n+|\b(?:but|however|instead\s+of|rather\s+than)\b|\b(?:and|и)\s+(?=(?:do\s+not|don't|never|avoid|mustn't|must\s+not|shouldn't|should\s+not|cannot|can't|не\s+(?:использ|примен|провод|выбира)))|,\s*(?:(?:and\s+)?not)\b|,\s*(?=(?:(?:please\s+)?(?:use|choose|select|apply|perform|adopt|follow|require|split|keep|ensure|prefer|recommend|использ|примен|провод|разб|раздел|выбира|сохраня|обеспеч|предпочт|рекоменду)\b))|(?:,\s*)?но\b", text, flags=re.I)
    for clause in clauses:
        if re.search(r"\b(?:as an example|for example|in the report|as an illustration|to illustrate)\b|(?:в качестве примера|для примера|в отч[её]т[а-я]*)", clause, re.I):
            continue
        # A direct validation-integrity requirement also prescribes a group split.
        # Require the directive itself, not an observation about external test groups.
        if re.fullmatch(
            r"\s*(?:internal\s+|cross[- ]?)?validation\s+(?:must|shall|should)\s+"
            r"(?:use|keep|have)\s+(?:mutually\s+)?disjoint\s+"
            r"(?:groups?|entities|subjects?|participants?|patients?|customers?|users?)\s*[.!?;]?\s*",
            clause, re.I,
        ):
            signals.add("group")
            continue
        assignments = list(re.finditer(
            r"(?:explicit_validation|validation[_ -]?(?:strategy|protocol)|split[_ -]?strategy)\s*[\"']?\s*[:=]\s*[\"']?(temporal_group|random|group|temporal)\b",
            clause,
            re.I,
        ))
        if assignments:
            for assignment in assignments:
                assigned = assignment.group(1).lower()
                destination = denied_signals if _protocol_is_negated(clause, assignment) else signals
                destination.update({"group", "temporal"} if assigned == "temporal_group" else {assigned})
            continue
        group_integrity = re.search(
            r"\b(?:groups?|queries|patients?|customers?)\b[^.!?;\n]{0,35}\b(?:must|should|shall|need to|are required to)\b[^.!?;\n]{0,20}\b(?:not overlap|not be split|remain separate|stay separate|be kept separate|be held out intact)\b"
            r"|\b(?:do not|don't|must not|should not)\s+split\s+(?:the\s+)?(?:groups?|queries|patients?|customers?)\b"
            r"|\b(?:keep|hold|leave)\b[^.!?;\n]{0,25}\b(?:groups?|queries|patients?|customers?)\b[^.!?;\n]{0,20}\b(?:separate|disjoint|intact|apart)\b"
            r"|(?:групп|запрос|пациент|клиент)[а-я]*[^.!?;\n]{0,25}(?:не\s+пересека|не\s+должн[а-я]*\s+пересека|раздельн|отдельн|не\s+смешив|сохранить\s+целост)"
            r"|не\s+(?:разб|раздел|смешив)[а-я]*[^.!?;\n]{0,25}(?:групп|запрос|пациент|клиент)[а-я]*",
            clause,
            re.I,
        )
        split_boundary = re.search(
            r"\b(?:across|between|между)\b[^.!?;\n]{0,35}\b(?:folds?|фолд[а-я]*|train(?:ing)?\s+(?:and|и|/)\s+(?:validation|test)|validation\s+(?:and|и|/)\s+test|обуча[а-я]*\s+(?:and|и|/)\s+(?:валидац[а-я]*|провероч[а-я]*|тестов[а-я]*)|валидац[а-я]*\s+(?:and|и|/)\s+тест[а-я]*)\b"
            r"|\b(?:do\s+not|don't|never|must\s+not|should\s+not)\s+split\s+(?:the\s+)?(?:groups?|queries|patients?|customers?)\b"
            r"|\b(?:groups?|queries|patients?|customers?)\b[^.!?;\n]{0,35}\b(?:not\s+be\s+split|must\s+not\s+be\s+split|should\s+not\s+be\s+split)\b"
            r"|(?:между|по\s+разным|через)[^.!?;\n]{0,35}(?:фолд|обуча|валидац|провероч|тестов)[а-я]*"
            r"|не\s+(?:разб|раздел)[а-я]*[^.!?;\n]{0,25}(?:групп|запрос|пациент|клиент)[а-я]*",
            clause,
            re.I,
        )
        if not split_boundary:
            group_integrity = None
        group_protocol = re.compile(
            r"\bgroup(?:[- ]based)?\s+(?:cross[- ]validation|validation|split|hold[- ]?out)\b"
            r"|группов[а-я]*\s+(?:валидац[а-я]*|разбиен[а-я]*)",
            re.I,
        )
        if any(_protocol_is_negated(clause, match) for match in group_protocol.finditer(clause)):
            denied_signals.add("group")
        group = re.search(
            r"\b(?:groups?|queries|patients?|customers?)[-_ ]?(?:based|wise)\s+(?:cross[- ]validation|validation|split|hold[- ]?out)\b"
            r"|\bgroup[- ](?:split|validation)\b|\bgroup[-_ ]?k[-_ ]?fold\b"
            r"|\b(?:cross[- ]validation|validation|split|hold[- ]?out)\b[^.!?;\n]{0,35}\b(?:by|at|across)\s+(?:the\s+)?(?:groups?|queries|patients?|customers?)\b"
            r"|\b(?:groups?|queries|patients?|customers?)\b[^.!?;\n]{0,35}\b(?:must|should|shall|need to|are required to)\b[^.!?;\n]{0,25}\b(?:not overlap|not be split|remain separate|stay separate|be kept separate|be held out intact)\b"
            r"|\b(?:do not|don't|must not|should not)\s+split\s+(?:the\s+)?(?:groups?|queries|patients?|customers?)\b"
            r"|\b(?:keep|hold|leave)\b[^.!?;\n]{0,25}\b(?:groups?|queries|patients?|customers?)\b[^.!?;\n]{0,20}\b(?:separate|disjoint|intact|apart)\b"
            r"|\b(?:leave[- ]one[- ](?:group|subject|entity)[- ]out|GroupKFold)\b"
            r"|группов[а-я]*\s+(?:валидац[а-я]*|разбиен[а-я]*)"
            r"|(?:валидац[а-я]*|разбиен[а-я]*|раздел[а-я]*)[^.!?;\n]{0,35}по\s+(?:групп|запрос|пациент|клиент)[а-я]*"
            r"|разб[а-я]*[^.!?;\n]{0,35}по\s+(?:групп|запрос|пациент|клиент)[а-я]*"
            r"|(?:групп|запрос|пациент|клиент)[а-я]*[^.!?;\n]{0,35}(?:не\s+пересека|не\s+должн[а-я]*\s+пересека|раздельн|отдельн|не\s+смешив|сохранить\s+целост)"
            r"|не\s+(?:разб|раздел|смешив)[а-я]*[^.!?;\n]{0,25}(?:групп|запрос|пациент|клиент)[а-я]*",
            clause,
            re.I,
        )
        temporal = re.search(
            r"\b(?:chronological(?:ly)?|time[- ]based|temporal|forward[- ]chaining|rolling[- ]origin|walk[- ]forward|TimeSeriesSplit)\b[^.!?;\n]{0,35}\b(?:cross[- ]validation|validation|split|hold[- ]?out|cv)\b"
            r"|\b(?:cross[- ]validation|validation|split|hold[- ]?out)\b[^.!?;\n]{0,35}\b(?:by|over|across|based\s+on)\s+(?:time|dates?|periods?)\b"
            r"|\b(?:validation|split|hold[- ]?out)\b[^.!?;\n]{0,35}\b(?:on|for)\s+(?:the\s+)?(?:future|next)\s+(?:periods?|dates?|years?|observations?)\b"
            r"|(?:временн[а-я]*|хронолог[а-я]*)[^.!?;\n]{0,35}(?:валидац[а-я]*|разбиен[а-я]*|проверочн[а-я]*\s+выборк[а-я]*)"
            r"|(?:валидац[а-я]*|разбиен[а-я]*)[^.!?;\n]{0,35}(?:по\s+времени|по\s+дат[а-я]*|по\s+период[а-я]*|хронолог[а-я]*)"
            r"|(?:валидац[а-я]*|разбиен[а-я]*)[^.!?;\n]{0,35}на\s+(?:будущ[а-я]*|следующ[а-я]*)\s+(?:период[а-я]*|дат[а-я]*|год[а-я]*)"
            r"|разб[а-я]*[^.!?;\n]{0,35}(?:по\s+времени|по\s+дат[а-я]*|по\s+период[а-я]*)",
            clause,
            re.I,
        )
        random = re.search(
            r"\brandom(?:ly)?\b[^.!?;\n]{0,25}\b(?:cross[- ]validation|validation|split|hold[- ]?out)\b"
            r"|\b(?:cross[- ]validation|validation|split|hold[- ]?out)\b[^.!?;\n]{0,25}\b(?:random|randomly)\b"
            r"|случайн[а-я]*[^.!?;\n]{0,35}(?:разбиен[а-я]*|валидац[а-я]*|выборк[а-я]*)"
            r"|(?:разбиен[а-я]*|валидац[а-я]*)[^.!?;\n]{0,35}случайн[а-я]*"
            r"|перемеш[а-я]*[^.!?;\n]{0,35}(?:раздел[а-я]*|валидац[а-я]*)",
            clause,
            re.I,
        )
        if alternative_pattern.search(clause) and any(
            match and _protocol_is_direct(clause, match) for match in (group, temporal, random)
        ):
            return False
        if group and not group_integrity and (not _protocol_is_direct(clause, group) or _protocol_is_negated(clause, group)):
            if _protocol_is_negated(clause, group):
                denied_signals.add("group")
            group = None
        if temporal and (not _protocol_is_direct(clause, temporal) or _protocol_is_negated(clause, temporal)):
            if _protocol_is_negated(clause, temporal):
                denied_signals.add("temporal")
            temporal = None
        if random and (not _protocol_is_direct(clause, random) or _protocol_is_negated(clause, random)):
            if _protocol_is_negated(clause, random):
                denied_signals.add("random")
            random = None
        if group:
            signals.add("group")
        if temporal:
            signals.add("temporal")
        if random:
            signals.add("random")
    if signals & denied_signals:
        return False
    if {"group", "temporal"} <= signals and "random" not in signals:
        prescribed = "temporal_group"
    elif signals == {"group"}:
        prescribed = "group"
    elif signals == {"temporal"}:
        prescribed = "temporal"
    elif signals == {"random"}:
        prescribed = "random"
    else:
        prescribed = None
    return strategy.lower() == prescribed


class Dataset:
    def __init__(self, task_path: Path, deferred=False):
        self.task_path = task_path.resolve()
        self.text = self.task_path.read_text(encoding="utf-8-sig")
        root = self.task_path.parent
        excluded = {"outputs", "results", "build", "dist", "node_modules", "__pycache__"}
        self.catalog = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (not path.is_file() or not path.resolve().is_relative_to(root)
                    or any(part.startswith(".") or part in excluded for part in relative.parts[:-1])
                    or path.name in {"submission.csv", "predictions.csv"}):
                continue
            if path.suffix.lower() == ".csv":
                self.catalog[relative.as_posix()] = {"sha256": file_hash(path), "columns": csv_header(path)}
        files = {name: item["sha256"] for name, item in self.catalog.items()}
        files["task.md"] = file_hash(self.task_path)
        explicit = root / "task_spec.json"
        if explicit.is_file():files["task_spec.json"] = file_hash(explicit)
        self.source_hash = digest(files)
        self.sources = {"task": {"sha256": files["task.md"], "text": self.text},
            "inventory": {"sha256": digest(self.catalog), "text": json.dumps(self.catalog, ensure_ascii=False)}}
        # Read only structurally verified metadata tables before task resolution.
        # Ordinary train/test values never become interpretation source text.
        fields = {column for item in self.catalog.values() for column in item['columns']}
        identifier_headers = {'feature', 'column', 'field', 'name', 'признак'}
        description_headers = {'description', 'description_ru', 'описание'}
        metadata_headers = identifier_headers | description_headers | {'role', 'dtype', 'unit', 'type', 'тип', 'единица', 'роль'}
        for name, item in self.catalog.items():
            headers = item['columns']
            identifier = next((column for column in headers if column.casefold() in identifier_headers), None)
            if (identifier is None or not any(column.casefold() in description_headers for column in headers)
                    or any(column.casefold() not in metadata_headers for column in headers)):
                continue
            path = root / name
            if path.stat().st_size > 100000:
                continue
            text = path.read_text(encoding='utf-8-sig')
            rows = list(csv.DictReader(io.StringIO(text)))
            keys = [row.get(identifier) for row in rows]
            if (not rows or len(rows)>2000 or len(keys)!=len(set(keys)) or not set(keys)<=fields
                    or any(None in row or any(value is None for value in row.values()) for row in rows)):
                continue
            self.sources['description_csv:'+name] = {'sha256':item['sha256'], 'text':text,
                'scope':'schema_description', 'validation':'unique_field_names_from_inventory; metadata_columns_only'}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (path.is_file() and path != self.task_path and path.suffix.lower() in {".md", ".txt"}
                    and not any(part.startswith(".") or part in excluded for part in relative.parts[:-1])
                    and path.stat().st_size <= 100000):
                self.sources["description:"+relative.as_posix()] = {"sha256": file_hash(path),
                    "text": path.read_text(encoding="utf-8-sig"),
                    "scope": _task_instruction_scope(relative.as_posix())}
                files[relative.as_posix()] = file_hash(path)
        self.source_hash = digest(files)
        self.fixed = {}
        self.task = None
        self.paths = {}
        self.deferred = deferred
        if not deferred:
            self.bind({name: str(path.relative_to(root)) for name in
                ("train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv")
                if (path := locate_csv(self.task_path, self.text, name, required=name in {"train.csv", "test.csv"}))})

    def bind(self, roles):
        if not {"train.csv", "test.csv"} <= set(roles) or set(roles)-{
                "train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv"}:
            raise ValueError("Explicit train/test file roles are required")
        if len(set(roles.values())) != len(roles) or any(value not in self.catalog for value in roles.values()):
            raise ValueError("File role is duplicated or outside the source inventory")
        paths = {name: inside(self.task_path.parent, roles[name]) if name in roles else None
            for name in ("train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv")}
        identities = [(path.stat().st_dev, path.stat().st_ino) for path in paths.values() if path is not None]
        if len(set(identities)) != len(identities):
            raise ValueError("File roles resolve to the same physical source")
        dictionary = self.sources.get('description_csv:' + roles.get('data_dictionary.csv', ''))
        if paths['data_dictionary.csv'] is not None and dictionary is None:
            raise ValueError("Dictionary requires a structurally verified schema-description source")
        if any(file_hash(paths[role]) != self.catalog[name]['sha256'] for role, name in roles.items()):
            raise ValueError("Source changed after inventory")
        self.paths = paths
        loader = getattr(self, "csv_loader", read_csv)
        self.train = loader(self.paths["train.csv"])
        self.test = loader(self.paths["test.csv"])
        self.sample = loader(self.paths["sample_submission.csv"]) if self.paths["sample_submission.csv"] else None
        self.sources["schema"] = {"sha256": digest({"train": list(self.train), "test": list(self.test)}),
            "train": list(self.train), "test": list(self.test)}
        if self.paths.get("data_dictionary.csv"):
            self.sources["dictionary"] = dict(dictionary)
        else:
            self.sources.pop('dictionary', None)
        if self.sample is not None:
            self.sources["output"] = {"sha256": file_hash(self.paths["sample_submission.csv"]), "columns": list(self.sample)}
        self.row_keys = [f"train:{i}" for i in range(len(self.train))]
        self.test_keys = [f"test:{i}" for i in range(len(self.test))]

    def explicit(self, contract_path: Path | None = None):
        path = contract_path or self.task_path.parent / "task_spec.json"
        if not path.is_file():raise ContractUnresolved("Without LLM an explicit task_spec.json contract is required")
        values = json.loads(path.read_text(encoding="utf-8"))
        if values.get("source_hash") not in (None, self.source_hash):raise ValueError("Explicit contract source mismatch")
        roles = values.get("file_roles") or {name: str(path.relative_to(self.task_path.parent)) for name in
            ("train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv")
            if (path := locate_csv(self.task_path, self.text, name, required=name in {"train.csv", "test.csv"}))}
        self.bind(roles)
        values.update(source_hash=self.source_hash, contract_version=2, file_roles=roles)
        task = TaskSpec.model_validate(values)
        self.validate(task)
        return task

    def role_hints(self, features):
        evidence = self.text + "\n" + self.sources.get("dictionary", {}).get("text", "")
        categorical = [c for c in features if not pd.api.types.is_numeric_dtype(numeric_if_possible(self.train[c]))]
        text_columns, time_columns, group_columns = [], [], []
        for column in self.test:
            lines = [line for line in evidence.splitlines() if re.search(r"(?<![\w])" + re.escape(column) + r"(?![\w])", line)]
            # Bind temporal wording to the described field, not a coincidental
            # feature name such as "code" elsewhere in another field's prose.
            time_lines = [line[match.end():] for line in re.split(r"[;\n]", evidence)
                if (match := re.match(r"^\s*(?:\|\s*|[-*]\s*)?(?:the\s+)?(?:(?:column|field|поле|столбец)\s+)?[`\"']?"
                    + re.escape(column) + r"[`\"']?(?!\w)", line, re.I))]
            time_signals = [bool(re.search(r"\b(?:not|never)(?:\s+(?:a|an|the))?\s*$|\bне\s*$", line[:match.start()], re.I))
                for line in time_lines for match in re.finditer(r"код периода|порядковый.*год|time column|период наблюден|timestamp|временн[а-я]* метк|calendar year of (?:the )?observation|year of (?:the )?observation|observation (?:year|date)|год наблюден|календарн[а-я ]*год наблюден|дата наблюден", line, re.I)]
            if time_signals and not any(time_signals):
                time_columns.append(column)
            if any(re.search(r"идентификатор[а-я ]*групп|идентификатор[а-я ]*запрос|group[ _-]*(?:id|identifier)|query[ _-]*id", line, re.I) for line in lines):
                group_columns.append(column)
            if column in features:
                if any(re.search(r"категориальн[а-я]* признак|\bcategorical\b", line, re.I) for line in lines):
                    categorical.append(column)
                if any(re.search(r"текстов[а-я]* признак|свободн[а-я ]*текст|text feature|free[ _-]*text", line, re.I) for line in lines):
                    text_columns.append(column)
        categorical = list(dict.fromkeys(c for c in categorical if c not in text_columns))
        return categorical, text_columns, time_columns, group_columns

    def structural_questions(self, features, time_candidates, group_candidates):
        evidence = self.text + "\n" + self.sources.get("dictionary", {}).get("text", "")
        temporal = re.search(r"year|calendar|chronolog|temporal|next period|following period|врем|период|год|хронолог", evidence, re.I)
        grouping = re.search(r"group|entity|subject|patient|групп|объект|пациент", evidence, re.I)
        uncertain = []
        observations = []
        for column in features:
            tr, te = numeric_if_possible(self.train[column]), numeric_if_possible(self.test[column])
            # An increasing identifier alone is never structural evidence.
            repeated = tr.nunique() < len(tr) / 2
            if temporal and not time_candidates and repeated and 2 <= tr.nunique() <= 100 and pd.api.types.is_numeric_dtype(tr) and pd.api.types.is_numeric_dtype(te):
                if tr.notna().all() and te.notna().all() and te.min() > tr.max():
                    uncertain.append("time_column")
                    observations.append({"column": column, "role": "possible_time", "train_distinct": int(tr.nunique()), "test_follows_train": True})
            lines = [line for line in evidence.splitlines() if re.search(r"(?<!\w)" + re.escape(column) + r"(?!\w)", line)]
            if grouping and not group_candidates and repeated and tr.nunique() >= 3 and any(re.search(r"(?:group|entity|subject|patient)[ _-]*(?:id|identifier|key)|идентификатор[^\n|,]*(?:групп|объект|пациент)", line, re.I) for line in lines):
                uncertain.append("group_column")
                observations.append({"column": column, "role": "possible_group", "train_distinct": int(tr.nunique())})
        if observations:
            text = json.dumps(observations, ensure_ascii=False)
            self.sources["structural_features"] = {"text": text, "sha256": digest(observations), "scope": "features_only"}
        return list(dict.fromkeys(uncertain))

    def draft(self) -> tuple[TaskSpec | None, list[str]]:
        explicit = self.task_path.parent / "task_spec.json"
        if explicit.is_file():
            values = json.loads(explicit.read_text(encoding="utf-8"))
            if values.get("source_hash") not in (None, self.source_hash):
                raise ValueError("Explicit task contract belongs to another source bundle")
            values["source_hash"] = self.source_hash
            values.setdefault("evidence", {"contract": ["task", "schema"]})
            task = TaskSpec.model_validate(values)
            self.validate(task)
            self.fixed = task.model_dump(mode="json")
            return task, []
        targets = [c for c in self.train if c not in self.test]
        extra = [c for c in self.test if c not in self.train]
        if extra:
            raise ValueError(f"Test contains fields absent in train: {extra}")
        if not targets:
            return None, ["targets"]
        self.fixed["targets"] = targets
        output_columns = list(self.sample) if self.sample is not None else None
        if output_columns is None:
            match = re.search(r"```csv\s*\n([^\n]+)", self.text, re.I)
            if match:
                output_columns = next(csv.reader(io.StringIO(match[1].strip())))
        if output_columns is None:
            return None, ["output_schema", "identifiers"]
        ids = [c for c in output_columns if c in self.test and c not in targets]
        self.fixed["id_columns"] = ids
        features = [c for c in self.test if c not in ids]
        self.fixed["feature_columns"] = features
        categorical, text_columns, time_candidates, group_candidates = self.role_hints(features)
        self.fixed["categorical_columns"] = categorical
        self.fixed["text_columns"] = text_columns
        if len(time_candidates) == 1:self.fixed["time_column"] = time_candidates[0]
        if len(group_candidates) == 1:self.fixed["group_column"] = group_candidates[0]
        role_questions = (["time_column"] if len(time_candidates) > 1 else []) + (["group_column"] if len(group_candidates) > 1 else [])
        role_questions += self.structural_questions(features, time_candidates, group_candidates)
        metric = parse_metric(self.text)
        if metric:
            self.fixed["metric.name"] = metric.name
            if metric.name == "f1":
                self.fixed["metric.average"] = metric.average
        else:
            return None, ["metric", "task_family", *role_questions]
        if role_questions:
            return None, role_questions
        numeric_metrics = {"rmse", "mse", "mae", "mape", "r2"}
        if metric.name in numeric_metrics and re.search(r"регресс|regress", self.text, re.I):
            kind = "regression" if len(targets) == 1 else "multioutput_regression"
            prediction_type = "values"
        elif metric.name in {"accuracy", "balanced_accuracy", "f1", "roc_auc", "log_loss", "average_precision"} and re.search(r"классификац|classif", self.text, re.I):
            kind = "classification" if len(targets) == 1 else "multioutput_classification"
            if metric.name in {"roc_auc", "log_loss", "average_precision"}:
                return None, ["probability_output_mapping", "positive_label"]
            prediction_type = "labels"
        else:
            return None, ["task_family", "output_mapping"]
        if len(targets) > 1:
            return None, ["target_aggregation", "task_family"]
        if set(output_columns) != set(ids + targets):
            return None, ["output_mapping"]
        task = TaskSpec(kind=kind, targets=targets, id_columns=ids, feature_columns=features,
            categorical_columns=categorical, text_columns=text_columns, metric=metric,
            output=OutputSpec(columns=output_columns, prediction_type=prediction_type),
            time_column=time_candidates[0] if time_candidates else None,
            group_column=group_candidates[0] if group_candidates else None,
            source_hash=self.source_hash, evidence={"targets": ["schema", "task"], "metric": ["task"],
                "id_columns": ["output", "task"] if self.sample is not None else ["task"],
                "time_column": ["task", *(["dictionary"] if "dictionary" in self.sources else [])] if time_candidates else ["schema"]})
        self.fixed.update({"kind": kind, "output": task.output.model_dump(mode="json")})
        self.validate(task)
        return task, []

    def validate(self, task: TaskSpec) -> None:
        if task.source_hash != self.source_hash:
            raise ValueError("Wrong source bundle hash")
        if not set(task.targets) <= set(self.train):
            raise ValueError("Targets are absent in the selected training source")
        if not set(task.feature_columns + task.id_columns) <= set(self.train):
            raise ValueError("Features or identifiers are absent in the selected training source")
        roles = set(task.feature_columns + task.id_columns + task.service_columns
            + task.series_columns + [c for c in (task.time_column, task.group_column) if c])
        if not set(task.feature_columns + task.id_columns) <= set(self.test) or set(self.test)-roles-set(task.targets):
            raise ValueError("Feature, ID and service roles do not cover the test schema")
        if set(self.train)-roles-set(task.targets):
            raise ValueError("Training-only fields require explicit service roles")
        # Reject an unusable feature role while the semantic proposal can still
        # be repaired, before activation leaves the task-resolution loop.
        invalid_numeric = []
        for column in task.feature_columns:
            if __import__("time").time() >= getattr(self, "deadline", float("inf")):
                raise TimeoutError("Original deadline exhausted during activation")
            if column not in task.categorical_columns + task.text_columns:
                for frame in (self.train, self.test):
                    converted = pd.to_numeric(frame[column], errors="coerce")
                    if converted[frame[column].notna()].isna().any():
                        invalid_numeric.append(column)
                        break
        if invalid_numeric:
            raise ValueError("Non-numeric value in numeric feature " + ", ".join(invalid_numeric))
        classification = task.kind in {"classification", "multioutput_classification", "multilabel"}
        for column in task.targets:
            labels = self.train[column]
            declared = task.classes.get(column)
            if (not classification or task.kind == "multilabel"
                    or declared and all(isinstance(value, (int, float)) for value in declared)
                    or isinstance(task.metric.positive_label, (int, float))):
                try:
                    labels = pd.to_numeric(labels, errors="raise")
                except ValueError:
                    raise ValueError("Non-numeric target " + column) from None
                if not np.isfinite(labels.to_numpy()).all():
                    raise ValueError("Nonfinite training target " + column)
            elif labels.isna().any():
                raise ValueError("Missing training target " + column)
            if task.kind == "multilabel" and not set(labels.unique()) <= {0, 1}:
                raise ValueError("Multilabel targets must be 0/1: " + column)
            observed = {0, 1} if task.kind == "multilabel" else set(labels.unique())
            if declared is not None and not observed <= set(declared):
                raise ValueError("Observed label outside declared class set for " + column)
            if task.metric.name == "f1" and task.metric.average == "binary":
                classes = set(declared) if declared is not None else observed
                if task.metric.positive_label is None or task.metric.positive_label not in classes or len(classes) != 2:
                    raise ValueError("Binary F1 requires an explicit positive label in a binary class space: " + column)
        for role in (task.time_column, task.group_column, *task.series_columns):
            if role is not None and (role not in self.test or role not in self.train):
                raise ValueError(f"Unknown structural column: {role}")
        if self.sample is not None:
            if list(self.sample) != task.output.columns or len(self.sample) != len(self.test):
                raise ValueError(f"Output disagrees with sample_submission schema or row count: expected_columns={list(self.sample.columns)!r}, proposed_columns={task.output.columns!r}, sample_rows={len(self.sample)}, test_rows={len(self.test)}")
            for column in task.id_columns:
                if self.sample[column].fillna("").tolist() != self.test[column].fillna("").tolist():
                    raise ValueError("Sample identifiers disagree with test row order")
        for key, fixed in self.fixed.items():
            actual = task.model_dump(mode="json")
            for component in key.split("."):
                actual = actual.get(component) if isinstance(actual, dict) else None
            if actual != fixed:
                raise ValueError(f"Proposal changes a locally established field: {key}")
        if task.metric.name in {"rmse", "mse", "mae", "mape", "r2"} and task.output.prediction_type != "values":
            raise ValueError("Regression metric requires numeric values")
        if task.metric.requires_probabilities and task.output.prediction_type != "probabilities":
            raise ValueError("Probability metrics require probability output")
        if task.output.prediction_type == "probabilities":
            expected = set(task.output.columns) - set(task.id_columns)
            if set(task.output.probability_columns) != expected:
                raise ValueError("Every probability output requires an explicit target/class mapping")
            if any(t not in task.targets for t, _ in task.output.probability_columns.values()):
                raise ValueError("Unknown probability target")
        elif (len(task.output.columns) != len(task.id_columns) + len(task.targets)
                or set(task.output.columns) != set(task.id_columns + task.targets)):
            raise ValueError("Renamed output columns require an explicit mapping that this adapter does not support; use target and identifier names")
        for column in task.targets:
            if self.train[column].isna().any():
                raise ValueError("Missing training targets are not imputed")
        if task.metric.name in {"roc_auc", "average_precision"} and len(task.targets) == 1 and task.metric.positive_label is None:
            observed = self.train[task.targets[0]].unique()
            if len(observed) == 2:
                raise ValueError("Binary probability metrics require an explicit positive class")

    def confirm_proposal(self, proposal: TaskResolutionProposal, request_id: str) -> TaskSpec:
        state = self.__dict__.copy()
        sources = self.sources.copy()
        try:
            return self._confirm_proposal(proposal, request_id)
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(state)
            # Controller requests retain this dictionary by reference.
            self.sources.clear()
            self.sources.update(sources)
            raise

    def _confirm_proposal(self, proposal: TaskResolutionProposal, request_id: str) -> TaskSpec:
        if proposal.request_id != request_id or proposal.source_hash != self.source_hash:
            raise ValueError("Proposal request/source mismatch")
        if proposal.status == "unresolved":
            raise ContractUnresolved("Model could not resolve the task contract")
        task = proposal.task_spec
        if self.deferred:
            if task.contract_version != 2:raise ValueError("Task contract version 2 is required")
            self.bind(task.file_roles)
        # Deferred v2 resolution does not call draft(), so it must also check
        # explicit column semantics before a missing time role freezes a split.
        _, _, time_candidates, _ = self.role_hints(task.feature_columns)
        if len(time_candidates) == 1 and task.time_column != time_candidates[0]:
            raise ValueError(f"time_column must preserve the documented observation period {time_candidates[0]}; cite its task/dictionary source")
        validated_evidence = {}
        for field in set(task.model_fields) - set(self.fixed) - {"source_hash", "evidence", "classes", "contract_version"}:
            if task.model_dump(mode="json")[field] in (None, [], {}):
                continue
            refs = proposal.evidence.get(field, [])
            if not refs or any(reference not in self.sources for reference in refs):
                raise ValueError(f"Missing or unknown source evidence for {field}")
            validated_evidence[field] = list(refs)
            # A citation must actually contain the proposed semantic role/value.
            text = "\n".join(self.sources[r].get("text", "") for r in refs)
            if field in {"time_column", "group_column"} and getattr(task, field) not in text:
                raise ValueError(f"The cited sources do not identify {field}")
            if field == "explicit_validation":
                protocol_text = "\n".join(self.sources[r].get("text", "") for r in refs
                    if r == "task" or self.sources[r].get("scope") == "organizer_instruction")
                if not _explicit_validation_supported(task.explicit_validation, protocol_text):
                    raise ValueError("The cited sources do not prescribe this explicit_validation protocol")
        metric = parse_metric(self.text)
        if not self.deferred and re.search(r"Основная метрика|Метрика оценки|Primary metric|Evaluation metric", self.text, re.I) and metric and (task.metric.name != metric.name or metric.name == "f1" and task.metric.average != metric.average):
            raise ValueError("Proposal contradicts the explicit metric")
        if len(task.targets) > 1:
            refs = proposal.evidence.get("metric", [])
            text = "\n".join(self.sources.get(ref, {}).get("text", "") for ref in refs)
            if not re.search(r"агрегац|усредн|средн[а-я]*.*цел|uniform|weighted|aggregation|averag", text, re.I):
                raise ValueError("Multioutput metric aggregation is not established by the cited source")
        task = task.model_copy(update={"evidence": validated_evidence})
        self.validate(task)
        return task

    def activate(self, task: TaskSpec) -> TaskSpec:
        for column in task.feature_columns:
            if __import__("time").time() >= getattr(self, "deadline", float("inf")):
                raise TimeoutError("Original deadline exhausted during activation")
            if column not in task.categorical_columns + task.text_columns:
                for frame in (self.train, self.test):
                    converted = pd.to_numeric(frame[column], errors="coerce")
                    if converted[frame[column].notna()].isna().any():
                        raise ValueError(f"Non-numeric value in numeric feature {column}")
                    frame[column] = converted.replace([np.inf, -np.inf], np.nan)
        classification = task.kind in {"classification", "multioutput_classification", "multilabel"}
        for column in task.targets:
            if classification:
                if task.kind == "multilabel":
                    self.train[column] = pd.to_numeric(self.train[column], errors="raise").astype(int)
                    if not set(self.train[column]) <= {0, 1}:
                        raise ValueError("Multilabel targets must be 0/1")
                elif (task.classes.get(column) and all(isinstance(value, (int, float)) for value in task.classes[column])) or isinstance(task.metric.positive_label, (int, float)):
                    self.train[column] = pd.to_numeric(self.train[column], errors="raise")
                labels = [0, 1] if task.kind == "multilabel" else sorted(self.train[column].unique().tolist(), key=str)
                if column in task.classes and not set(labels) <= set(task.classes[column]):
                    raise ValueError("Observed label outside declared class set")
                task.classes.setdefault(column, labels)
            else:
                self.train[column] = pd.to_numeric(self.train[column], errors="raise")
                if not np.isfinite(self.train[column].to_numpy()).all():
                    raise ValueError("Nonfinite training target")
        self.task = task
        return task

    def features(self, indices: list[int] | None = None, test: bool = False) -> pd.DataFrame:
        frame = self.test if test else self.train
        frame = frame if indices is None else frame.iloc[indices]
        metadata = [self.task.time_column, self.task.group_column, *self.task.series_columns]
        columns = list(dict.fromkeys(self.task.feature_columns + [c for c in metadata if c]))
        return frame[columns].copy()

    def profile(self, indices: list[int], scope: str = "fit") -> DataProfile:
        from .profiling import build_profile, profile_key
        key = profile_key(self, indices, scope)
        if not hasattr(self, "_profile_cache"):
            self._profile_cache = {}
        if key not in self._profile_cache:
            self._profile_cache[key] = build_profile(self, indices, scope)
        return self._profile_cache[key].model_copy(deep=True)

    def inspect(self, args, plan, profile_document=None) -> dict[str, Any]:
        if args.scope != "fit" and args.include_target:
            raise ValueError("Target-based analysis is restricted to fit")
        if args.report == "dictionary":
            source = self.sources.get("dictionary")
            if not source:
                return {"status": "not_found", "scope": "schema", "source": "dictionary"}
            columns = args.columns
            if columns and not set(columns) <= set(self.task.feature_columns + self.task.id_columns + self.task.targets):
                raise ValueError("Unknown dictionary column")
            lines = source["text"].splitlines()
            if columns:
                lines = [line for line in lines if any(re.search(r"(?<![\w])" + re.escape(c) + r"(?![\w])", line) for c in columns)]
            text = "\n".join(lines)
            return {"status": "available", "scope": "schema", "source": "dictionary", "sha256": source["sha256"],
                "content": text[:20000], "truncated": len(text) > 20000, "hint": "Request a named subset of columns if truncated"}
        frame, columns = self.select_report(args, plan)
        result: dict[str, Any] = {"scope": args.scope, "data_version": self.source_hash,
            "protocol_hash": plan.hash, "rows": len(frame), "report": args.report}
        selected = frame[columns]
        if args.report == "profile" and not args.group_by:
            from .profiling import build_profile
            indices = plan.fit if args.scope == "fit" else plan.validation if args.scope == "validation_features" else list(range(len(self.test)))
            document = profile_document or (build_profile(self, indices, args.scope, frame=frame).model_dump(mode="json") if args.predicate
                else self.profile(indices, args.scope).model_dump(mode="json"))
            result.update(status=document.get("status", "available"), error=document.get("error"),
                version=document["version"], source=document["source"],
                columns={c: v for c, v in document["columns"].items() if c in columns},
                targets={c: v for c, v in document["targets"].items() if args.include_target and (not args.columns or c in columns)},
                selection={"columns": columns, "predicate": args.predicate})
            target_risks = {"label_conflicts", "duplicate_target_variation", "class_imbalance", "rare_classes", "heavy_tail"}
            result["observations"] = [{**item, "flags": {k: v for k, v in item.get("flags", {}).items()
                if args.include_target or k not in target_risks}} for item in document["observations"]]
            result["findings"] = [item for item in document["findings"] if args.include_target or item["kind"] not in target_risks]
        elif args.report == "correlations":
            raise ValueError("Correlations require the resource-monitored get_dataset_report worker")
        else:
            from .profiling import descriptive_report
            result.update(descriptive_report(frame, columns, args.report, args.group_by))
        return result

    def select_report(self, args, plan):
        if args.scope != "fit" and args.include_target:
            raise ValueError("Target-based analysis is restricted to fit")
        if args.scope == "fit":
            frame = self.train.iloc[plan.fit]
            allowed = set(self.task.feature_columns + (self.task.targets if args.include_target else []))
        else:
            frame = self.test if args.scope == "test_features" else self.train.iloc[plan.validation]
            allowed = set(self.task.feature_columns)
        columns = args.columns or self.task.feature_columns
        if len(columns) != len(set(columns)):
            raise ValueError("Report columns must be unique")
        if not set(columns) <= allowed or (args.group_by and args.group_by not in allowed):
            raise ValueError("Column is unavailable in this analysis scope")
        if args.predicate:
            if args.scope != "fit":
                raise ValueError("Arbitrary predicates are restricted to fit")
            if set(args.predicate) != {"column", "op", "value"} or args.predicate["column"] not in allowed:
                raise ValueError("Invalid fit predicate")
            series = frame[args.predicate["column"]]
            op, value = args.predicate["op"], args.predicate["value"]
            if op == "eq":
                mask = series == value
            elif op == "is_missing":
                mask = series.isna()
            elif op == "gt":
                mask = series > value
            elif op == "lt":
                mask = series < value
            else:
                raise ValueError("Unsupported predicate operator")
            frame = frame.loc[mask]
        return frame, columns
