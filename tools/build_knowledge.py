"""Development-only documentation collector. Never imported by the runtime."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import re
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse, urljoin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automl_agent.contracts import PipelineSpec, ImputationPlan, Transformation
from automl_agent.util import atomic_bytes, atomic_json, canonical, digest, file_hash, read_json

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "data_for_participants/requirements.txt"
RUNTIME = ROOT / "knowledge_runtime"
BUILD = ROOT / "knowledge_build"

PACKAGES = {
    "scikit-learn": ("1.5.2", "https://github.com/scikit-learn/scikit-learn/blob/1.5.2/", "BSD-3-Clause"),
    "catboost": ("1.2.8", "https://github.com/catboost/catboost/blob/v1.2.8/", "Apache-2.0"),
    "lightgbm": ("4.6.0", "https://github.com/microsoft/LightGBM/blob/v4.6.0/", "MIT"),
    "xgboost": ("2.1.4", "https://github.com/dmlc/xgboost/blob/v2.1.4/", "Apache-2.0"),
    "flaml": ("2.3.6", "https://github.com/microsoft/FLAML/blob/v2.3.6/", "MIT"),
    "autogluon.tabular": ("1.3.1", "https://github.com/autogluon/autogluon/blob/v1.3.1/", "Apache-2.0"),
    "optuna": ("4.2.1", "https://github.com/optuna/optuna/blob/v4.2.1/", "MIT"),
    "feature-engine": ("1.8.3", "https://github.com/feature-engine/feature_engine/blob/1.8.3/", "BSD-3-Clause"),
    "pandas": ("2.2.3", "https://github.com/pandas-dev/pandas/blob/v2.2.3/", "BSD-3-Clause"),
    "numpy": ("1.26.4", "https://github.com/numpy/numpy/blob/v1.26.4/", "BSD-3-Clause"),
    "scipy": ("1.12.0", "https://github.com/scipy/scipy/blob/v1.12.0/", "BSD-3-Clause"),
    "imbalanced-learn": ("0.12.4", "https://github.com/scikit-learn-contrib/imbalanced-learn/blob/0.12.4/", "MIT"),
    "category-encoders": ("2.6.4", "https://github.com/scikit-learn-contrib/category_encoders/blob/2.6.4/", "BSD-3-Clause"),
    "statsmodels": ("0.14.0", "https://github.com/statsmodels/statsmodels/blob/v0.14.0/", "BSD-3-Clause"),
    "pydantic": ("2.10.6", "https://github.com/pydantic/pydantic/blob/v2.10.6/", "MIT"),
    "psutil": ("6.1.1", "https://github.com/giampaolo/psutil/blob/release-6.1.1/", "BSD-3-Clause"),
}

SYMBOL_GROUPS = {
    "scikit-learn": {
        "sklearn.pipeline": ["Pipeline"], "sklearn.compose": ["ColumnTransformer", "TransformedTargetRegressor"],
        "sklearn.impute": ["SimpleImputer", "MissingIndicator", "IterativeImputer", "KNNImputer"],
        "sklearn.preprocessing": ["OneHotEncoder", "OrdinalEncoder", "TargetEncoder", "StandardScaler", "RobustScaler", "PowerTransformer", "SplineTransformer", "LabelEncoder"],
        "sklearn.linear_model": ["Ridge", "ElasticNet", "LogisticRegression", "SGDClassifier", "SGDRegressor", "BayesianRidge"],
        "sklearn.ensemble": ["RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor", "HistGradientBoostingClassifier", "HistGradientBoostingRegressor"],
        "sklearn.neural_network": ["MLPClassifier", "MLPRegressor"],
        "sklearn.feature_extraction.text": ["TfidfVectorizer", "HashingVectorizer"],
        "sklearn.feature_selection": ["VarianceThreshold", "SelectKBest", "mutual_info_classif", "mutual_info_regression"],
        "sklearn.model_selection": ["GroupShuffleSplit", "StratifiedKFold", "TimeSeriesSplit", "TunedThresholdClassifierCV"],
        "sklearn.calibration": ["CalibratedClassifierCV"],
        "sklearn.metrics": ["mean_squared_error", "mean_absolute_error", "r2_score", "f1_score", "accuracy_score", "roc_auc_score", "log_loss", "average_precision_score", "ndcg_score"],
    },
    "catboost": {"catboost": ["CatBoostClassifier", "CatBoostRegressor"]},
    "lightgbm": {"lightgbm": ["LGBMClassifier", "LGBMRegressor", "LGBMRanker"]},
    "xgboost": {"xgboost": ["XGBClassifier", "XGBRegressor"]},
    "flaml": {"flaml": ["AutoML"]},
    "autogluon.tabular": {"autogluon.tabular": ["TabularPredictor"]},
    "optuna": {"optuna": ["create_study"], "optuna.study": ["Study"], "optuna.samplers": ["TPESampler"], "optuna.trial": ["Trial"]},
    "feature-engine": {"feature_engine.creation": ["MathFeatures", "RelativeFeatures"],
        "feature_engine.selection": ["DropConstantFeatures", "DropDuplicateFeatures", "DropCorrelatedFeatures"],
        "feature_engine.transformation": ["YeoJohnsonTransformer", "LogCpTransformer"],
        "feature_engine.datetime": ["DatetimeFeatures"], "feature_engine.timeseries.forecasting": ["LagFeatures"]},
    "pandas": {"pandas": ["read_csv", "to_numeric", "to_datetime", "DataFrame"]},
    "numpy": {"numpy": ["asarray", "isfinite", "mean", "hstack"]},
    "scipy": {"scipy.sparse": ["csr_matrix", "hstack"]},
    "imbalanced-learn": {"imblearn.pipeline": ["Pipeline"], "imblearn.over_sampling": ["RandomOverSampler", "SMOTE"]},
    "category-encoders": {"category_encoders": ["TargetEncoder"]},
    "statsmodels": {"statsmodels.tsa.statespace.sarimax": ["SARIMAX"], "statsmodels.tsa.holtwinters": ["ExponentialSmoothing"], "statsmodels.regression.linear_model": ["OLS"]},
    "pydantic": {"pydantic": ["BaseModel", "Field"]}, "psutil": {"psutil": ["Process", "virtual_memory"]},
}

GUIDES = [
    ("scikit-learn", "imputation", "https://scikit-learn.org/1.5/modules/impute.html"),
    ("scikit-learn", "leakage_validation", "https://scikit-learn.org/1.5/common_pitfalls.html"),
    ("scikit-learn", "cross_validation", "https://scikit-learn.org/1.5/modules/cross_validation.html"),
    ("scikit-learn", "multioutput_multilabel", "https://scikit-learn.org/1.5/modules/multiclass.html"),
    ("scikit-learn", "text_features", "https://scikit-learn.org/1.5/modules/feature_extraction.html"),
    ("scikit-learn", "calibration", "https://scikit-learn.org/1.5/modules/calibration.html"),
    ("lightgbm", "parameters", "https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html"),
    ("lightgbm", "missing_categories", "https://lightgbm.readthedocs.io/en/v4.6.0/Advanced-Topics.html"),
    ("autogluon.tabular", "fit", "https://auto.gluon.ai/1.3.1/api/autogluon.tabular.TabularPredictor.fit.html"),
    ("autogluon.tabular", "refit", "https://auto.gluon.ai/1.3.1/api/autogluon.tabular.TabularPredictor.refit_full.html"),
    ("flaml", "task_automl", "https://raw.githubusercontent.com/microsoft/FLAML/v2.3.6/website/docs/Use-Cases/Task-Oriented-AutoML.md"),
    ("xgboost", "parameters", "https://raw.githubusercontent.com/dmlc/xgboost/v2.1.4/doc/parameter.rst"),
    ("catboost", "missing_values", "https://raw.githubusercontent.com/catboost/catboost/v1.2.8/catboost/docs/en/concepts/algorithm-missing-values-processing.md"),
    ("statsmodels", "forecast_validation", "https://www.statsmodels.org/v0.14.0/examples/notebooks/generated/statespace_forecasting.html"),
    ("optuna", "sequential_trials", "https://optuna.readthedocs.io/en/v4.2.1/reference/generated/optuna.study.Study.html"),
]

ALIASES = {"пропуск": "missing imputation", "импутац": "imputation missing values", "утечк": "leakage validation preprocessing",
    "категор": "categorical unknown categories encoding", "редк": "rare class imbalance", "текст": "text tfidf sparse",
    "временн": "time series temporal forecasting", "период": "temporal time split", "групп": "group cross validation",
    "ранжир": "ranking group ndcg", "вероятност": "probability predict_proba calibration", "порог": "threshold tuning",
    "переобучен": "overfitting regularization validation", "нейросет": "MLP neural network", "лес": "random forest extra trees",
    "линейн": "linear ridge logistic", "ансамбл": "ensemble averaging", "памят": "memory sparse dense",
    "сплайн": "SplineTransformer smooth nonlinear", "сохран": "serialization refit persistence"}


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "footer", "header"}:
            self.skip += 1
        if not self.skip and tag in {"p", "div", "section", "h1", "h2", "h3", "h4", "pre", "tr", "li", "br"}:
            self.parts.append("\n")
        if not self.skip and tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "footer", "header"} and self.skip:
            self.skip -= 1
        if not self.skip and tag in {"p", "div", "section", "pre", "tr", "li"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def export_api(package, module_name, name, licenses):
    required, root, license_id = PACKAGES[package]
    actual = metadata.version(package)
    if actual != required:
        raise ValueError(f"Exact-version API extraction refused: {package} {actual}, expected {required}")
    module = importlib.import_module(module_name)
    obj = getattr(module, name)
    symbol = f"{module_name}.{name}"
    doc = inspect.getdoc(obj) or ""
    if inspect.isclass(obj):
        constructor_doc = inspect.getdoc(obj.__init__) or ""
        if len(constructor_doc) > 100 and constructor_doc not in doc:
            doc += "\n\nConstructor parameters:\n" + constructor_doc
    if len(doc) < 40:
        raise ValueError(f"Missing API docstring: {symbol}")
    try:
        signature = str(inspect.signature(obj))
    except (ValueError, TypeError):
        first = doc.splitlines()[0]
        signature = first[len(name):] if first.startswith(name + "(") else "signature unavailable; inspect the documented parameters"
    signature = re.sub(r" at 0x[0-9a-fA-F]+(?=>)", "", signature)
    parts = [f"# {symbol}\n\nVersion: {actual}\n\n{symbol}{signature}\n\n{doc}"]
    symbols = [symbol]
    for method in ("fit", "transform", "fit_transform", "predict", "predict_proba", "save_model", "load_model", "refit_full", "retrain_from_log", "optimize", "suggest_float", "suggest_int", "suggest_categorical", "to_csv", "groupby", "merge"):
        if hasattr(obj, method):
            member = getattr(obj, method)
            description = inspect.getdoc(member)
            if description:
                try:
                    sig = str(inspect.signature(member))
                except (ValueError, TypeError):
                    sig = "(see parameters)"
                parts.append(f"\n\n## {symbol}.{method}{sig}\n\n{description}")
                symbols.append(f"{symbol}.{method}")
    content = re.sub(r" at 0x[0-9a-fA-F]+(?=>)", "", "\n".join(parts))
    doc_id = "api_" + symbol.replace(".", "_")
    relative = f"references/{doc_id}.md"
    original = {"symbol": symbol, "version": actual, "signature": signature, "content": content}
    raw = BUILD / "raw" / f"{doc_id}__{digest(original)}.json"
    if not raw.is_file() or raw.read_text(encoding="utf-8") != canonical(original) + "\n":
        atomic_json(raw, original)
    target = RUNTIME / relative
    atomic_bytes(target, (content + "\n").encode())
    try:
        source = inspect.getsourcefile(obj) if not inspect.isbuiltin(obj) else None
    except TypeError:
        source = None
    source_hash = file_hash(Path(source)) if source and Path(source).is_file() else None
    return {"doc_id": doc_id, "library": package, "required_version": required, "doc_version": actual,
        "version_match": "exact_release", "source_kind": "installed_exact_release_docstring", "source_url": root,
        "source_commit": None, "retrieved_at": datetime.fromtimestamp(raw.stat().st_mtime, timezone.utc).isoformat(), "original_sha256": file_hash(raw),
        "normalized_sha256": file_hash(target), "source_file_sha256": source_hash, "section_path": symbol,
        "api_symbols": symbols, "license": license_id, "license_files": licenses.get(package, []),
        "content_status": "complete" if "unavailable" not in signature else "partial_signature",
        "runtime_api_status": "unverified", "path": relative, "search_text": symbol + " " + doc[:4000]}


def collect_licenses(offline=False):
    licenses, errors = {}, []
    for package, (version, root, name) in PACKAGES.items():
        try:
            dist = metadata.distribution(package)
            candidates = [p for p in dist.files or [] if "dist-info" in str(p) and any(token in p.name.lower() for token in ("license", "copying", "notice"))]
            paths = []
            for i, member in enumerate(candidates):
                original = Path(dist.locate_file(member))
                if not original.is_file():
                    continue
                relative = f"licenses/{package}/{i}_{member.name}"
                atomic_bytes(RUNTIME / relative, original.read_bytes())
                paths.append(relative)
            licenses[package] = paths
            if not paths:
                url = root.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/") + "LICENSE"
                try:
                    body, _ = download(url, offline)
                    relative = f"licenses/{package}/UPSTREAM_LICENSE"
                    atomic_bytes(RUNTIME / relative, body)
                    licenses[package] = [relative]
                except Exception as error:
                    errors.append({"package": package, "reason": f"upstream_license_unavailable: {error}", "source": url})
        except metadata.PackageNotFoundError:
            errors.append({"package": package, "reason": "not_installed"})
    return licenses, errors


def expand_api_sections(record):
    """Index each API object with its own signature, parameters and warnings."""
    path = RUNTIME / record["path"]
    content = path.read_text(encoding="utf-8")
    boundaries = []
    for symbol in record["api_symbols"][1:]:
        match = re.search(r"\n\n## " + re.escape(symbol) + r"\(", content)
        if match:
            boundaries.append((match.start(), symbol))
    if not boundaries:
        return [record]
    boundaries.sort()
    result = []
    parent = dict(record)
    parent_content = content[:boundaries[0][0]] + "\n"
    atomic_bytes(path, parent_content.encode())
    parent["api_symbols"] = [record["api_symbols"][0]]
    parent["normalized_sha256"] = file_hash(path)
    parent["search_text"] = parent_content
    result.append(parent)
    for index, (start, symbol) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(content)
        text = f"# {symbol}\nVersion: {record['doc_version']}\n\n" + content[start:end].strip() + "\n"
        item = {**record, "doc_id": "api_" + symbol.replace(".", "_"), "api_symbols": [symbol], "section_path": symbol}
        item["path"] = f"references/{item['doc_id']}.md"
        atomic_bytes(RUNTIME / item["path"], text.encode())
        item["normalized_sha256"] = file_hash(RUNTIME / item["path"])
        item["search_text"] = text
        result.append(item)
    return result


def download(url, offline):
    import httpx
    cache = BUILD / "source_cache" / digest(url)
    meta = cache.with_suffix(".json")
    if cache.is_file() and meta.is_file():
        record = read_json(meta)
        if file_hash(cache) != record["sha256"]:
            raise ValueError("Cached documentation hash mismatch")
        return cache.read_bytes(), record
    if offline:
        raise ValueError("Source is not present in the pinned offline cache")
    allowed = {urlparse(item[2]).hostname for item in GUIDES} | {"api.github.com"}
    if urlparse(url).scheme != "https" or urlparse(url).hostname not in allowed:
        raise ValueError("Documentation URL is outside the official HTTPS allowlist")
    current = url
    with httpx.Client(follow_redirects=False, trust_env=False, timeout=30) as client:
        for attempt in range(3):
            response = client.get(current)
            if response.is_redirect:
                target = urljoin(current, response.headers["location"])
                if urlparse(target).scheme != "https" or urlparse(target).hostname not in allowed:
                    raise ValueError("Documentation redirect leaves the official allowlist")
                current = target
                continue
            response.raise_for_status()
            body = response.content
            mime = response.headers.get("content-type", "")
            if not body.strip() or len(body) > 3 * 1024 * 1024 or not any(x in mime for x in ("text/", "application/xhtml", "application/json")):
                raise ValueError("Invalid documentation MIME or size")
            if b"404: Not Found" in body[:300] or b"Just a moment..." in body[:500]:
                raise ValueError("Error/interstitial body is not documentation")
            record = {"url": url, "final_url": current, "sha256": hashlib.sha256(body).hexdigest(),
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "mime": mime}
            atomic_bytes(cache, body)
            atomic_json(meta, record)
            return body, record
    raise ValueError("Documentation redirect/retry limit reached")


def guide_record(package, name, url, offline, licenses):
    body, origin = download(url, offline)
    commit = None
    resolved_url = url
    if urlparse(url).hostname == "raw.githubusercontent.com":
        resolved_url, commit = resolve_raw_commit(url, offline)
        pinned, pinned_origin = download(resolved_url, offline)
        if hashlib.sha256(pinned).hexdigest() != origin["sha256"]:
            raise ValueError("Pinned commit content differs from the cached release-tag snapshot")
    text = body.decode("utf-8")
    includes = []
    if package == "catboost" and "raw.githubusercontent.com" in url:
        text, includes = resolve_catboost_includes(text, resolved_url, offline)
    if "html" in origin["mime"]:
        parser = TextExtractor(); parser.feed(text)
        text = "".join(parser.parts)
        text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)
    if len(text.strip()) < 100:
        raise ValueError("The normalized guide is too short to be a complete reference")
    required, _, license_id = PACKAGES[package]
    exact = required in text or "raw.githubusercontent.com" in url
    unresolved = any(token in text for token in ("{% include", "{% cut", ".. autoclass::", ".. automodule::", "{{"))
    doc_id = f"guide_{package}_{name}".replace(".", "_")
    path = f"references/{doc_id}.md"
    atomic_bytes(RUNTIME / path, (f"# {name}\nSource: {url}\n\n" + text).encode())
    return {"doc_id": doc_id, "library": package, "required_version": required,
        "doc_version": required if exact else ".".join(required.split(".")[:2]),
        "version_match": "exact_release" if exact else "version_family", "source_kind": "official_documentation_snapshot",
        "source_url": url, "resolved_source_url": resolved_url, "source_commit": commit, "retrieved_at": origin["retrieved_at"],
        "original_sha256": origin["sha256"], "normalized_sha256": file_hash(RUNTIME / path),
        "section_path": name, "api_symbols": [], "license": license_id,
        "license_files": licenses.get(package, []), "content_status": "partial_unresolved_includes" if unresolved else "complete",
        "runtime_api_status": "unverified", "path": path, "search_text": name + " " + text[:10000], "included_sources": includes}


def resolve_raw_commit(url, offline):
    parts = urlparse(url).path.strip("/").split("/")
    organization, repository, ref = parts[:3]
    response, _ = download(f"https://api.github.com/repos/{organization}/{repository}/git/ref/tags/{ref}", offline)
    obj = json.loads(response)["object"]
    if obj["type"] == "tag":
        response, _ = download(f"https://api.github.com/repos/{organization}/{repository}/git/tags/{obj['sha']}", offline)
        obj = json.loads(response)["object"]
    if obj["type"] != "commit" or not re.fullmatch(r"[a-f0-9]{40}", obj["sha"]):
        raise ValueError("Release tag did not resolve to a concrete commit")
    parts[2] = obj["sha"]
    return "https://raw.githubusercontent.com/" + "/".join(parts), obj["sha"]


def resolve_catboost_includes(text, url, offline, depth=0):
    if depth > 8:
        raise ValueError("Documentation include depth exceeds its bound")
    origins = []
    pattern = r"\{%\s*include\s+\[[^\]]*\]\(([^)]+)\)\s*%\}"
    def replace(match):
        target = urljoin(url, match[1])
        base = "/".join(url.split("/")[:6]) + "/catboost/docs/"
        if not target.startswith(base):
            raise ValueError("Include escapes the fixed official release")
        body, source = download(target, offline)
        content, nested = resolve_catboost_includes(body.decode(), target, offline, depth + 1)
        origins.extend([source, *nested])
        return content
    text = re.sub(pattern, replace, text)
    if depth == 0:
        import yaml
        presets_url = urljoin(url, "../presets.yaml")
        try:
            raw, origin = download(presets_url, offline)
            presets = yaml.safe_load(raw)
            values = presets.get("default", presets)
            for _ in range(5):
                previous = text
                text = re.sub(r"\{\{\s*([^} ]+)\s*\}\}", lambda m: str(values.get(m[1], m[0])), text)
                if text == previous:
                    break
            origins.append(origin)
        except Exception:
            # Unresolved substitutions remain visibly marked as partial coverage.
            pass
    return text, origins


METHODS = [
    ("task_contract", "Постановка и формат ответа / task contract", "Неоднозначны таргет, метрика или формат", "Подтвердить полный контракт по схеме и описанию", "unresolved; уточнение по источникам", "Нельзя выводить семантику из числового типа", ["api_sklearn_metrics_f1_score"]),
    ("period_transfer", "Перенос на следующий период / temporal validation", "Все периоды test идут после train", "Проверять на следующих целых периодах", "случайное разбиение только для независимых строк", "Смешение периода даёт завышенную оценку", ["api_sklearn_model_selection_TimeSeriesSplit"]),
    ("group_transfer", "Новые группы / group validation", "Несколько наблюдений одного объекта или запроса", "Сохранить целостность групп во внешнем и внутреннем split", "групповой holdout; прямое обучение", "ID сам по себе не доказывает группу", ["api_sklearn_model_selection_GroupShuffleSplit"]),
    ("rare_classes", "Редкие классы / rare classes", "Несбалансированные классы и мало наблюдений", "Проверить покрытие и Macro F1", "веса классов; oversampling только fit", "Не считать малый пилот доказательством слабости семьи", ["api_sklearn_metrics_f1_score"]),
    ("simple_imputation", "Простое заполнение / median baseline", "Числовые и категориальные пропуски", "Быстрая опора: медиана и отдельная категория", "нативные NaN; модельное восстановление", "Сохранять полностью пустые столбцы и индикаторы", ["api_sklearn_impute_SimpleImputer"]),
    ("iterative_imputation", "Модельное восстановление / iterative imputation", "Есть информативные числовые признаки и пропуски", "Сравнить BayesianRidge-импутацию с тем же пайплайном", "медиана; нативные NaN; деревья", "Только fit, без таргета задачи; до 3 проходов и 20 предикторов на старте", ["api_sklearn_impute_IterativeImputer"]),
    ("tree_imputation", "Нелинейное восстановление / tree imputation", "Линейный восстановитель может упускать взаимодействия", "Использовать деревья для восстановления наблюдённых столбцов", "BayesianRidge; медиана", "Учитывать все вспомогательные fit; смешанные данные не называть точным missForest", ["api_sklearn_ensemble_ExtraTreesRegressor"]),
    ("native_nan", "Нативные пропуски / native missing values", "Бустинг умеет обрабатывать NaN", "Проверить необходимость явной импутации", "модельное заполнение того же бустинга", "Категориальные NaN требуют отдельной обработки", ["api_lightgbm_LGBMRegressor"]),
    ("empty_columns", "Пустые столбцы / empty features", "На fit столбец полностью отсутствует", "Сохранить столбец, индикатор и фиксированную константу", "исключение как отдельная гипотеза", "Модель восстановителя обучить не на чем", ["api_sklearn_impute_MissingIndicator"]),
    ("unknown_categories", "Новые категории / unknown categories", "Test содержит отсутствующие на fit категории", "Сравнить безопасное one-hot и нативные категории", "редкие категории; линейная модель", "Словарь нельзя обучать на test", ["api_sklearn_preprocessing_OneHotEncoder"]),
    ("target_encoding", "Кодирование таргетом / target encoding", "Высокая кардинальность категорий", "Сравнить cross-fitted encoding с one-hot", "CatBoost; ограниченный one-hot", "KFold TargetEncoder не сохраняет время/группы автоматически", ["api_sklearn_preprocessing_TargetEncoder", "api_category_encoders_TargetEncoder"]),
    ("linear", "Линейная опора / linear regularization", "Нужны быстрые модели и проверка линейного сигнала", "Ridge/Logistic с масштабированием", "леса; бустинг; сплайны", "Масштабирование и словари обучаются только на fit", ["api_sklearn_linear_model_Ridge", "api_sklearn_linear_model_LogisticRegression"]),
    ("forest", "Леса / random forest extra trees", "Возможны нелинейные взаимодействия", "Сравнить RandomForest/ExtraTrees с опорой", "бустинг; линейная модель", "Ограничить деревья, глубину и CPU", ["api_sklearn_ensemble_ExtraTreesRegressor"]),
    ("boosting", "Градиентный бустинг / gradient boosting", "Табличная задача с числовыми и категориальными признаками", "Сравнить разные бустинги и обработку категорий", "леса; линейные; MLP", "Три бустинга не заменяют разные модельные семейства", ["api_catboost_CatBoostRegressor", "api_lightgbm_LGBMRegressor", "api_xgboost_XGBRegressor"]),
    ("mlp", "Компактная нейросеть / MLP from scratch", "Есть бюджет и ограниченное числовое представление", "Проверить небольшую MLP с регуляризацией", "линейная модель; бустинг", "Оценить dense-память и число параметров; MLP 1.5.2 не принимает sample_weight", ["api_sklearn_neural_network_MLPClassifier"]),
    ("text", "Текст / TF-IDF sparse representation", "Объявлены текстовые признаки", "Словные/символьные TF-IDF плюс линейная модель", "HashingVectorizer; смешанная табличная ветвь", "Без внешнего корпуса, эмбеддингов и pretrained", ["api_sklearn_feature_extraction_text_TfidfVectorizer"]),
    ("smooth", "Гладкие зависимости / additive spline", "Возможны гладкие нелинейные числовые эффекты", "SplineTransformer плюс Ridge", "линейная опора; деревья", "Ограничить число узлов и ширину матрицы", ["api_sklearn_preprocessing_SplineTransformer"]),
    ("target_log", "Логарифм таргета / target transform", "Положительный таргет с тяжёлым хвостом", "Сравнить log1p с обратным преобразованием", "исходная шкала и устойчивый objective", "Основная метрика остаётся в исходной шкале", ["api_sklearn_compose_TransformedTargetRegressor"]),
    ("weights", "Дисбаланс / class weights", "Macro F1 страдает на редких классах", "Проверить class_weight как отдельное изменение", "random oversampling; исходные веса", "Не передавать sample_weight неподдержанному API", ["api_sklearn_linear_model_LogisticRegression"]),
    ("resampling", "Ресэмплинг / SMOTE oversampling", "Нужна дополнительная гипотеза дисбаланса", "Изменять только обучающий fold", "веса классов; RandomOverSampler", "Validation/control не ресэмплируются; смешанные категории требуют осторожности", ["api_imblearn_over_sampling_RandomOverSampler", "api_imblearn_over_sampling_SMOTE"]),
    ("threshold", "Порог / threshold tuning", "Бинарная метрика по меткам", "Подобрать порог на внутреннем holdout", "порог 0.5; калибровка", "Не подбирать порог по внешней validation или многоклассовому Macro F1 этим адаптером", ["api_sklearn_model_selection_TunedThresholdClassifierCV"]),
    ("calibration", "Калибровка / probability calibration", "Важна точность вероятностей", "Проверить внутреннюю калибровку", "исходные вероятности; Log Loss", "decision_function не является вероятностью; split сохраняет группы/время", ["api_sklearn_calibration_CalibratedClassifierCV"]),
    ("automl", "Готовый AutoML / budgeted library search", "Нужно проверить портфель в общем бюджете", "Один FLAML/AutoGluon эксперимент с независимой оценкой", "прямые семейства; Optuna", "Внешние метки не передаются tuning_data; search owner один", ["api_flaml_AutoML", "api_autogluon_tabular_TabularPredictor"]),
    ("hpo", "Настройка семейства / sequential Optuna", "Выбрано перспективное семейство", "Ограниченное условное пространство на внутреннем fit", "смена представления; другой алгоритм", "Нельзя оборачивать AutoML вторым HPO", ["api_optuna_study_Study"]),
    ("ensemble", "Ансамбль / aligned averaging", "Есть совместимые модели с разными ошибками", "Равновесное среднее значений/вероятностей", "одна дешёвая модель; внутренний stacking", "Проверить ключи строк, классы и полный граф refit; метки не усредняются", ["api_sklearn_ensemble_RandomForestClassifier"]),
    ("forecast", "Прогноз / causal lag features", "Есть время, горизонт и независимые ряды", "Наивная опора, причинные лаги, statsmodels", "календарные признаки; сезонный наивный прогноз", "Не использовать будущий таргет и соседний ряд; коротким рядам нужен резерв", ["api_statsmodels_tsa_statespace_sarimax_SARIMAX"]),
    ("multioutput", "Несколько целей / multioutput multilabel", "Схема требует несколько выходов", "Независимые модели и явная агрегация", "алгоритм с собственной поддержкой multioutput", "Отдельные пространства классов и стоимость всех fit", ["guide_scikit-learn_multioutput_multilabel"]),
    ("ranking", "Ранжирование / learning to rank", "Даны запросы и релевантности", "LGBMRanker с группами и NDCG/MAP@k", "регрессия релевантности", "Группы целые, размеры после сортировки корректны, порядок ответа восстановлен", ["api_lightgbm_LGBMRanker"]),
    ("feature_selection", "Отбор / feature selection", "Есть константы, дубли и корреляции", "Проверить ограниченный готовый отбор", "регуляризация; исходные признаки", "Отбор обучается внутри fit и переобучается на refit", ["api_feature_engine_selection_DropCorrelatedFeatures"]),
    ("diagnostics", "Ошибки / restricted validation diagnostics", "Нужно понять результат эксперимента", "Использовать разрешённые агрегаты и анализ fit", "новые гипотезы по признакам fit", "Не восстанавливать скрытый срез через дополнения или ID", ["api_sklearn_metrics_mean_squared_error"]),
    ("budget", "Ресурсы / complete graph budget", "Дорогие преобразования и модели", "Пилот, внешний таймаут и резерв всех финалистов", "более дешёвое семейство; finish", "Таймаут не доказывает низкое качество; учитываются импутеры, HPO и предсказания", ["api_psutil_Process"]),
    ("refit", "Переобучение / frozen refit", "Поиск завершён и финалисты заморожены", "Сохранить конфигурацию, полный граф и допустимые строки", "сохранённый пригодный результат победителя контроля", "Не запускать новый HPO и не терять прежний CSV при сбое", ["api_autogluon_tabular_TabularPredictor"]),
]


def catalogs(records):
    known = {record["doc_id"] for record in records}
    for identifier, title, observation, hypothesis, alternatives, restrictions, references in METHODS:
        card = {"id": identifier, "method_id": identifier, "version": 1, "title": title,
            "tags": title.lower().split(), "observations": observation, "hypothesis": hypothesis,
            "alternatives": alternatives, "constraints": restrictions, "minimum_experiment": "Изменить один компонент и сравнить на неизменном протоколе",
            "acceptance": "Основная метрика и стоимость по сохранённой selection_policy", "heuristic_status": "project_hypothesis_not_universal_rule",
            "references": references, "missing_references": [r for r in references if r not in known],
            "review_status": "source_linked" if all(r in known for r in references) else "partial"}
        atomic_json(RUNTIME / "methods" / f"{identifier}.json", card)
    families = [
        ("ridge", ["regression", "multioutput_regression"], "linear"),
        ("elastic_net", ["regression", "multioutput_regression"], "linear"),
        ("logistic", ["classification", "multioutput_classification", "multilabel"], "linear"),
        ("sgd", ["regression", "classification", "multioutput_regression", "multioutput_classification", "multilabel"], "linear"),
        ("random_forest", ["regression", "classification", "multioutput_regression", "multioutput_classification", "multilabel"], "forest"),
        ("extra_trees", ["regression", "classification", "multioutput_regression", "multioutput_classification", "multilabel"], "forest"),
        ("catboost", ["regression", "classification", "multioutput_regression", "multioutput_classification", "multilabel"], "boosting"),
        ("lightgbm", ["regression", "classification", "multioutput_regression", "multioutput_classification", "multilabel"], "boosting"),
        ("xgboost", ["regression", "classification", "multioutput_regression", "multioutput_classification", "multilabel"], "boosting"),
        ("mlp", ["regression", "classification", "multioutput_regression", "multioutput_classification", "multilabel"], "mlp"),
        ("hist_gradient_boosting", ["regression", "classification"], "boosting"),
        ("spline", ["regression", "multioutput_regression"], "smooth"),
        ("naive", ["forecasting"], "forecast"), ("sarimax", ["forecasting"], "forecast"),
        ("exponential_smoothing", ["forecasting"], "forecast"), ("lgbm_ranker", ["ranking"], "ranking"),
    ]
    for model, tasks, method in families:
        spec = PipelineSpec(model=model)
        if model in {"catboost", "lightgbm", "lgbm_ranker"}:
            spec.representation = "native"
        if model == "spline":
            spec.transformations = [Transformation(name="spline")]
        if model in {"naive", "sarimax", "exponential_smoothing"}:
            spec.representation = "lags"
        entry = {"id": model, "version": 1, "task_kinds": tasks, "method_refs": [method], "alternatives": ["linear", "forest", "boosting"],
            "pipeline": spec.model_dump(mode="json"), "capability_status": "unverified_until_execution", "prediction_types": ["values", "probabilities", "scores"],
            "pilot": {"max_rows": 400, "preserve": ["classes", "groups", "time"], "max_seconds": 120},
            "refit_policy": {"configuration": "freeze_successful_parameters", "rows": "permitted_training_rows", "new_search": False},
            "search_space_ref": model if model in {"ridge", "logistic", "random_forest", "extra_trees", "catboost", "lightgbm", "xgboost", "mlp"} else None}
        atomic_json(RUNTIME / "recipes" / f"{model}.json", entry)
    for engine in ("flaml", "autogluon"):
        spec = PipelineSpec(engine=engine, model="library_search", search_owner=engine)
        atomic_json(RUNTIME / "recipes" / f"{engine}.json", {"id": engine, "version": 1,
            "task_kinds": ["regression", "classification", "multioutput_regression", "multioutput_classification", "multilabel"],
            "pipeline": spec.model_dump(mode="json"), "method_refs": ["automl"], "alternatives": ["linear", "boosting", "forest"],
            "capability_status": "unverified_until_execution", "pilot": {"fit_only": True},
            "refit_policy": {"new_search": False, "configuration": "logged_record_or_original_model_graph"}})
    for model in ("ridge", "logistic"):
        source = read_json(RUNTIME / "recipes" / f"{model}.json")
        source["id"] = f"text_{model}"; source["method_refs"] = ["text"]
        source["pipeline"]["representation"] = "mixed"
        atomic_json(RUNTIME / "recipes" / f"text_{model}.json", source)
    spaces = {
        "ridge": {"alpha": {"type": "float", "low": .001, "high": 100., "log": True}},
        "logistic": {"C": {"type": "float", "low": .001, "high": 100., "log": True}},
        "random_forest": {"n_estimators": {"type": "int", "low": 50, "high": 250, "step": 50}, "max_depth": {"type": "categorical", "choices": [5, 10, 20, None]}, "min_samples_leaf": {"type": "int", "low": 1, "high": 8}},
        "extra_trees": {"n_estimators": {"type": "int", "low": 50, "high": 250, "step": 50}, "min_samples_leaf": {"type": "int", "low": 1, "high": 8}},
        "catboost": {"iterations": {"type": "int", "low": 100, "high": 700, "step": 100}, "depth": {"type": "int", "low": 4, "high": 8}, "learning_rate": {"type": "float", "low": .02, "high": .2, "log": True}},
        "lightgbm": {"n_estimators": {"type": "int", "low": 100, "high": 700, "step": 100}, "num_leaves": {"type": "int", "low": 7, "high": 63}, "learning_rate": {"type": "float", "low": .02, "high": .2, "log": True}},
        "xgboost": {"n_estimators": {"type": "int", "low": 100, "high": 500, "step": 100}, "max_depth": {"type": "int", "low": 3, "high": 7}, "learning_rate": {"type": "float", "low": .02, "high": .2, "log": True}},
        "mlp": {"hidden_layer_sizes": {"type": "categorical", "choices": [[64], [128], [128, 64], [256, 128]]}, "alpha": {"type": "float", "low": .0001, "high": .1, "log": True}, "learning_rate_init": {"type": "float", "low": .0003, "high": .01, "log": True}, "max_iter": {"type": "categorical", "choices": [50, 100, 150]}},
    }
    for name, parameters in spaces.items():
        atomic_json(RUNTIME / "search_spaces" / f"{name}.json", {"id": name, "version": 1, "owner": "optuna", "parameters": parameters,
            "limits": {"n_jobs": 1, "max_trials": 40, "no_nested_search": True, "scope": "inner_fit_only"},
            "cost_constraints": "Check matrix width, parameter count and full-refit reserve before fit"})


def build(offline=False):
    started = time.monotonic()
    BUILD.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "documentation_sources.yaml", BUILD / "documentation_sources.yaml")
    for directory in ("references", "methods", "recipes", "search_spaces", "fallback_search", "licenses"):
        (RUNTIME / directory).mkdir(parents=True, exist_ok=True)
    for previous in (RUNTIME / "references").glob("*.md"):
        if previous.name.startswith(("api_", "guide_")):
            previous.unlink()
    from sklearn.experimental import enable_iterative_imputer
    licenses, errors = collect_licenses(offline)
    records = []
    for package, modules in SYMBOL_GROUPS.items():
        for module, names in modules.items():
            for name in names:
                try:
                    records.extend(expand_api_sections(export_api(package, module, name, licenses)))
                except Exception as error:
                    errors.append({"package": package, "symbol": f"{module}.{name}", "reason": f"{type(error).__name__}: {error}"})
    for package, name, url in GUIDES:
        try:
            records.append(guide_record(package, name, url, offline, licenses))
        except Exception as error:
            errors.append({"package": package, "url": url, "reason": f"{type(error).__name__}: {error}"})
    catalogs(records)
    from automl_agent.references import validate_catalogs
    validate_catalogs(RUNTIME)
    records.sort(key=lambda record: record["doc_id"])
    atomic_bytes(RUNTIME / "references/records.jsonl", ("\n".join(canonical(record) for record in records) + "\n").encode())
    atomic_json(RUNTIME / "symbols.json", {symbol: record["doc_id"] for record in records for symbol in record["api_symbols"]})
    atomic_json(RUNTIME / "aliases_ru_en.json", ALIASES)
    atomic_json(RUNTIME / "fallback_search/documents.json", [{"doc_id": r["doc_id"], "symbols": r["api_symbols"], "text": r["search_text"]} for r in records])
    index = RUNTIME / "search.sqlite"
    index.unlink(missing_ok=True)
    connection = sqlite3.connect(index)
    fts = True
    try:
        connection.execute("CREATE VIRTUAL TABLE docs USING fts5(doc_id UNINDEXED,symbols,tags,content, tokenize=\"unicode61 tokenchars '._'\")")
        connection.executemany("INSERT INTO docs VALUES (?,?,?,?)", [(r["doc_id"], " ".join(r["api_symbols"]), r["section_path"], r["search_text"]) for r in records])
        connection.commit()
    except sqlite3.OperationalError:
        fts = False
    finally:
        connection.close()
    gaps = [{"package": "torch", "required_version": "2.5.1", "status": "not_used_by_active_recipes", "reason": "MLP uses scikit-learn; no downloaded weights or torch recipe"},
        {"package": "lightautoml", "required_version": "0.4.2", "status": "disabled_unverified_optional_adapter"}]
    coverage = {"api_records": sum(r["source_kind"] == "installed_exact_release_docstring" for r in records),
        "guide_records": sum(r["source_kind"] == "official_documentation_snapshot" for r in records),
        "methods": len(list((RUNTIME / "methods").glob("*.json"))), "recipes": len(list((RUNTIME / "recipes").glob("*.json"))),
        "search_spaces": len(list((RUNTIME / "search_spaces").glob("*.json"))), "fts5": fts, "errors": errors, "registered_gaps": gaps,
        "competition_bundle_permission": "requires_organizer_confirmation", "zip_size_limit": "unknown"}
    atomic_json(RUNTIME / "coverage.json", coverage)
    notices = "# Third-party documentation notices\n\nAPI references are exact installed-release docstrings and selected official guide snapshots.\nNo examples are executed. Documentation does not authorize downloads or additional dependencies.\n\n"
    for package, (_, source, license_id) in PACKAGES.items():
        notices += f"- {package}: {license_id}; official source {source}; copies: {', '.join(licenses.get(package, [])) or 'MISSING — see coverage'}\n"
    atomic_bytes(RUNTIME / "THIRD_PARTY_NOTICES.md", notices.encode())
    files = {str(path.relative_to(RUNTIME)): file_hash(path) for path in sorted(RUNTIME.rglob("*")) if path.is_file() and path.name != "knowledge.lock.json"}
    version = digest(files)
    lock = {"schema_version": 1, "bundle_version": version, "builder_version": 1,
        "requirements_sha256": file_hash(REQUIREMENTS), "files": files, "coverage": coverage,
        "size_bytes": sum((RUNTIME / path).stat().st_size for path in files), "runtime_network": False}
    atomic_json(RUNTIME / "knowledge.lock.json", lock)
    atomic_json(BUILD / "extraction_reports/latest.json", {"bundle_version": version, "seconds": time.monotonic() - started,
        "files": len(files), "coverage": coverage, "offline": offline})
    print(canonical({"bundle_version": version, "api_records": coverage["api_records"], "guides": coverage["guide_records"],
        "methods": coverage["methods"], "recipes": coverage["recipes"], "errors": len(errors), "bytes": lock["size_bytes"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--offline", action="store_true")
    build(parser.parse_args().offline)
