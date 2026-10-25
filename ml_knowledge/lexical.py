"""Shared dependency-free bilingual lexical vocabulary."""
import re

STOPWORDS = set("a an and are as at be been by can do for from has have if in is it of on or that the this to was will with self none true false import return class def parameters returns examples notes see also".split())
ALIASES = {"f1": "sklearn.metrics.f1_score macro classification class_offsets", "макро": "macro f1 sklearn.metrics.f1_score", "ансамбл": "ensemble weighted_oof stacked_oof", "сдвиг": "distribution shift domain importance_weighting", "пропуск": "missing imputation", "импутац": "imputation missing", "утечк": "leakage cross validation",
           "категор": "categories categorical encoding", "неизвестн": "unknown", "временн": "temporal time series",
           "период": "temporal time", "групп": "group split", "нейросет": "MLP neural network", "вес": "weight",
           "памят": "memory", "порог": "threshold", "калибров": "calibration", "редк": "rare imbalance",
           "модельн": "iterative", "кодирован": "encoding", "сплайн": "spline", "текст": "text tfidf",
           "сохран": "save serialization", "вероятност": "probability predict_proba"}


def tokens(text):
    words = re.findall(r"[\w]+(?:\.[\w]+)*", text.casefold())
    expanded = []
    for word in words:
        expanded.append(word)
        if "." in word:
            expanded.extend(word.split("."))
    return {word for word in expanded if len(word) > 1 and word not in STOPWORDS}

