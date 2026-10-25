from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, SplineTransformer, StandardScaler

from .compatibility import native_branch, require_compatible


def selected_features(spec, task):
    include = task.feature_columns if spec.include_features is None else spec.include_features
    if len(include) != len(set(include)) or len(spec.exclude_features) != len(set(spec.exclude_features)):
        raise ValueError("Duplicate feature selection entries")
    if not set(include + spec.exclude_features) <= set(task.feature_columns):
        raise ValueError("Feature selection must use confirmed features, never targets or service columns")
    result = [column for column in task.feature_columns if column in include and column not in spec.exclude_features]
    if not result and not (task.kind == "forecasting" and not task.feature_columns and spec.include_features is None and not spec.exclude_features):
        raise ValueError("Feature selection must remain nonempty")
    return result


class FrameTransform(BaseEstimator, TransformerMixin):
    """Stateless, explicitly requested feature operations before fitted encoding."""

    def __init__(self, transformations=()):
        self.transformations = transformations

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for operation in self.transformations:
            if operation.name == "datetime":
                for column in operation.parameters.get("columns", []):
                    dt = pd.to_datetime(X[column], errors="coerce", utc=True)
                    for part in ("year", "month", "day", "dayofweek", "hour"):
                        X[f"{column}__{part}"] = getattr(dt.dt, part).astype(float)
                    X = X.drop(columns=[column])
            elif operation.name == "math":
                from feature_engine.creation import MathFeatures, RelativeFeatures
                p = dict(operation.parameters)
                kind = p.pop("kind", "math")
                transform = RelativeFeatures(**p) if kind == "relative" else MathFeatures(**p)
                X = transform.fit_transform(X)
            elif operation.name == "log1p":
                for column in operation.parameters.get("columns", []):
                    if (X[column].dropna() < 0).any():
                        raise ValueError("Feature log1p requires nonnegative values")
                    X[column] = np.log1p(X[column])
            elif operation.name == "aggregate":
                p = operation.parameters
                columns, functions = p.get("columns", []), p.get("functions", ["mean"])
                if not columns or not set(columns) <= set(X) or not set(functions) <= {"mean", "std", "min", "max", "sum"}:
                    raise ValueError("Aggregates require explicit available columns and bounded row functions")
                values = X[columns].apply(pd.to_numeric, errors="raise")
                prefix = p.get("prefix", "aggregate")
                for function in functions:
                    name = f"{prefix}__{function}"
                    if name in X:
                        raise ValueError("Aggregate output name already exists")
                    X[name] = values.sum(axis=1, min_count=1) if function == "sum" else getattr(values, function)(axis=1)
                if p.get("drop_source", False):
                    X = X.drop(columns=columns)
        return X


class FeatureMatrix:
    def __init__(self, spec, task, cpu=1, memory_mb=1024, strategy="random"):
        self.spec, self.task, self.cpu, self.memory_mb = spec, task, cpu, memory_mb
        self.strategy = strategy
        self.summary = {}

    def fit(self, X, y=None):
        started = time.monotonic()
        require_compatible(self.spec, self.task)
        self.feature_columns = selected_features(self.spec, self.task)
        from .supervised_features import SupervisedFeatures
        self.frame_steps = []
        frame = X[self.feature_columns].copy()
        supervised_report = []
        for operation in self.spec.transformations:
            if operation.name in {"history", "target_encode"}:
                step = SupervisedFeatures(self.spec.model_copy(update={"transformations": [operation]}), self.task, self.strategy)
                frame = step.fit_transform(frame, y, X)
                supervised_report.extend(step.report)
            elif operation.name in {"datetime", "math", "log1p", "aggregate"}:
                step = FrameTransform([operation])
                frame = step.fit_transform(frame)
            else:
                continue
            self.frame_steps.append(step)
        self.training_frame = frame
        self.text_columns = [c for c in self.task.text_columns if c in frame]
        self.categorical_columns = [c for c in frame if c not in self.text_columns and not pd.api.types.is_numeric_dtype(frame[c])]
        self.numeric_columns = [c for c in frame if c not in self.text_columns + self.categorical_columns]
        self.empty_columns = [c for c in self.numeric_columns if frame[c].isna().all()]
        self.complete_columns = [c for c in self.numeric_columns if frame[c].notna().all()]
        self.imputer = None
        self.numeric_scaler = None
        self.numeric_transformers = []
        self.categorical_models = {}
        numeric = frame[self.numeric_columns].to_numpy(dtype=float)
        numeric = np.where(np.isfinite(numeric), numeric, np.nan)
        self.numeric_method = self.spec.imputation.numeric
        if self.numeric_method == "native" and self.spec.model not in {"catboost", "lightgbm", "xgboost", "hist_gradient_boosting", "lgbm_ranker"}:
            raise ValueError("Native missing values are unsupported by this model")
        auxiliary = 0
        self.imputer_positions = list(range(len(self.numeric_columns)))
        if self.numeric_columns and self.numeric_method != "native":
            plan = self.spec.imputation
            if self.numeric_method in {"iterative_bayesian_ridge", "iterative_trees"} and len(self.empty_columns) < len(self.numeric_columns):
                # sklearn 1.5.2's keep_empty_features=True marks every valid
                # column missing and returns with n_iter_=0. Preserve empty
                # columns ourselves and fit only the observed-width submatrix.
                self.imputer_positions = [i for i, column in enumerate(self.numeric_columns) if column not in self.empty_columns]
                estimator = BayesianRidge() if self.numeric_method == "iterative_bayesian_ridge" else ExtraTreesRegressor(
                    n_estimators=20, max_depth=8, n_jobs=self.cpu, random_state=self.spec.seed)
                self.imputer = IterativeImputer(estimator=estimator, max_iter=plan.max_iter,
                    n_nearest_features=min(plan.n_nearest_features, max(1, len(self.imputer_positions) - 1)),
                    initial_strategy="median", sample_posterior=False, random_state=self.spec.seed,
                    skip_complete=True, keep_empty_features=False)
            elif self.numeric_method == "knn":
                if len(frame) * len(self.numeric_columns) * 8 > self.memory_mb * 1024 * 1024 * 0.1 or len(frame) > 20000:
                    raise ValueError("KNN imputation donor cost exceeds the pilot bound")
                self.numeric_scaler = StandardScaler()
                numeric = self.numeric_scaler.fit_transform(numeric)
                self.imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
            else:
                self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            numeric[:, self.imputer_positions] = self.imputer.fit_transform(numeric[:, self.imputer_positions])
            if hasattr(self.imputer, "imputation_sequence_"):
                auxiliary += len(self.imputer.imputation_sequence_)
            if self.numeric_scaler is not None:
                numeric = self.numeric_scaler.inverse_transform(numeric)
            for column in self.empty_columns:
                numeric[:, self.numeric_columns.index(column)] = plan.empty_numeric
        elif self.numeric_columns:
            for column in self.empty_columns:
                numeric[:, self.numeric_columns.index(column)] = self.spec.imputation.empty_numeric
        if self.spec.imputation.categorical == "model":
            requested = self.spec.imputation.categorical_model_columns or [c for c in self.categorical_columns if frame[c].isna().any()]
            predictors = np.nan_to_num(numeric, nan=0.0)
            for column in requested:
                if column not in self.categorical_columns:
                    raise ValueError("Categorical imputation requested for a non-categorical feature")
                observed = frame[column].notna().to_numpy()
                if not len(self.numeric_columns) or observed.sum() < 10 or frame.loc[observed, column].nunique() < 2:
                    continue
                model = ExtraTreesClassifier(n_estimators=25, max_depth=8, n_jobs=self.cpu, random_state=self.spec.seed)
                model.fit(predictors[observed], frame.loc[observed, column].astype(str))
                self.categorical_models[column] = model
                auxiliary += 1
        self.category_fill = {}
        for column in self.categorical_columns:
            values = frame[column].dropna()
            sentinel = "__MISSING__"
            while sentinel in set(values):
                sentinel += "_"
            self.category_fill[column] = values.mode().iloc[0] if len(values) and self.spec.imputation.categorical == "most_frequent" else sentinel
        cat = self._categorical(frame, numeric)
        self.native = native_branch(self.spec, self.task)
        if self.native:
            self.categories = {column: list(pd.unique(cat[column])) for column in self.categorical_columns}
        elif self.categorical_columns:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=2, max_categories=256, dtype=np.float64)
            self.encoder.fit(cat)
        if len(self.numeric_columns) and self.numeric_method != "native":
            linear = self.spec.model in {"ridge", "elastic_net", "logistic", "sgd", "mlp", "spline"}
            transforms = [op for op in self.spec.transformations if op.name in {"standard_scale", "robust_scale", "yeo_johnson", "spline"}]
            if linear and not any(op.name in {"standard_scale", "robust_scale"} for op in transforms):
                from .contracts import Transformation
                transforms.append(Transformation(name="standard_scale"))
            for operation in transforms:
                params = dict(operation.parameters)
                if operation.name == "standard_scale":
                    transformer = StandardScaler(**params)
                elif operation.name == "robust_scale":
                    transformer = RobustScaler(**params)
                elif operation.name == "yeo_johnson":
                    transformer = PowerTransformer(method="yeo-johnson", **params)
                else:
                    transformer = SplineTransformer(n_knots=params.pop("n_knots", 5), degree=params.pop("degree", 3), **params)
                numeric = transformer.fit_transform(numeric)
                self.numeric_transformers.append(transformer)
        if self.spec.model == "knn" and self.spec.neighbor_scale != "none" and self.numeric_columns and not any(
                op.name in {"standard_scale", "robust_scale"} for op in self.spec.transformations):
            transformer = StandardScaler() if self.spec.neighbor_scale == "standard" else RobustScaler()
            numeric = transformer.fit_transform(numeric)
            self.numeric_transformers.append(transformer)
        self.vectorizers = {}
        for column in self.text_columns:
            if self.spec.text_analyzer == "hashing":
                vectorizer = HashingVectorizer(n_features=self.spec.max_text_features, alternate_sign=False)
            else:
                ngram = (1, 2) if self.spec.text_analyzer == "word" else (3, 5)
                vectorizer = TfidfVectorizer(analyzer=self.spec.text_analyzer, ngram_range=ngram,
                    max_features=self.spec.max_text_features, sublinear_tf=True, min_df=1)
            texts = frame[column].fillna("").astype(str)
            if not texts.str.strip().str.len().gt(0).any():
                texts = pd.Series(["emptytext"] * len(frame), index=frame.index)
            vectorizer.fit(texts)
            self.vectorizers[column] = vectorizer
        self.selector = None
        self.summary = {"method": self.numeric_method, "numeric_columns": self.numeric_columns,
            "empty_numeric_columns": self.empty_columns, "complete_fit_columns_initial_fallback": self.complete_columns,
            "categorical_method": self.spec.imputation.categorical, "categorical_models": list(self.categorical_models),
            "indicators": self.spec.imputation.indicators, "auxiliary_fits": auxiliary,
            "fit_seconds": time.monotonic() - started, "training_scope": "fit_only_no_target"}
        if self.numeric_method in {"iterative_bayesian_ridge", "iterative_trees"}:
            self.summary["empty_columns_preserved_outside_iterative_imputer"] = True
            self.summary["actual_iterations"] = int(getattr(self.imputer, "n_iter_", 0))
            self.summary["model_imputation_executed"] = auxiliary > 0
        transformed = self._transform_frame(frame)
        if self.native:
            names = list(transformed.columns)
        else:
            numeric_names = list(self.numeric_columns)
            for transformer in self.numeric_transformers:
                numeric_names = list(transformer.get_feature_names_out(numeric_names))
            names = numeric_names
            if self.numeric_columns and self.spec.imputation.indicators:
                names += [f"{c}__missing" for c in self.numeric_columns]
            if self.categorical_columns:
                names += list(self.encoder.get_feature_names_out(self.categorical_columns))
                if self.spec.imputation.indicators:
                    names += [f"{c}__missing" for c in self.categorical_columns]
            for column, vectorizer in self.vectorizers.items():
                tokens = vectorizer.get_feature_names_out() if hasattr(vectorizer, "get_feature_names_out") else range(self.spec.max_text_features)
                names += [f"{column}__{token}" for token in tokens]
                if self.spec.imputation.indicators:
                    names.append(f"{column}__missing")
        if not self.native:
            from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
            for operation in self.spec.transformations:
                if operation.name == "drop_constant":
                    selector = VarianceThreshold(**operation.parameters)
                elif operation.name == "select_kbest":
                    if y is None:
                        raise ValueError("Supervised feature selection requires fit targets")
                    classification = self.task.kind in {"classification", "multioutput_classification", "multilabel"}
                    selector = SelectKBest(f_classif if classification else f_regression, k=min(operation.parameters.get("k", 30), transformed.shape[1]))
                elif operation.name == "drop_correlated":
                    if transformed.shape[1] > 500 or transformed.shape[0] * transformed.shape[1] * 8 > self.memory_mb * 1024**2 * .1:
                        raise ValueError("Correlation selection requires a bounded dense representation")
                    if sparse.issparse(transformed):
                        transformed = transformed.toarray()
                    from feature_engine.selection import DropCorrelatedFeatures
                    selector = DropCorrelatedFeatures(**operation.parameters)
                    transformed = pd.DataFrame(transformed)
                else:
                    continue
                target = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
                transformed = selector.fit_transform(transformed, target)
                if hasattr(selector, "get_support"):
                    names = [name for name, keep in zip(names, selector.get_support()) if keep]
                else:
                    names = [name for index, name in enumerate(names) if index not in selector.features_to_drop_]
                if self.selector is None:
                    self.selector = []
                self.selector.append(selector)
        self.width = transformed.shape[1]
        if len(names) != self.width:
            raise ValueError("Executed preprocessing feature schema differs from matrix width")
        self.output_features = names
        self.summary["transformed_features"] = self.width
        self.processing_summary = {"sequence": [op.model_dump(mode="json") for op in self.spec.transformations],
            "implicit_numeric_transforms": [type(t).__name__ for t in self.numeric_transformers],
            "representation": "native" if self.native else "encoded", "feature_count": self.width}
        self.processing_summary["supervised_features"] = supervised_report
        return self

    def _categorical(self, frame, numeric):
        result = frame[self.categorical_columns].copy()
        for column in self.categorical_columns:
            mask = result[column].isna()
            if column in self.categorical_models and mask.any():
                result.loc[mask, column] = self.categorical_models[column].predict(np.nan_to_num(numeric[mask], nan=0.0))
            result[column] = result[column].fillna(self.category_fill[column]).astype(str)
        return result

    def transform(self, X):
        return self._transform_frame(self.feature_frame(X))

    def feature_frame(self, X):
        from .supervised_features import SupervisedFeatures
        frame = X[self.feature_columns].copy()
        for step in self.frame_steps:
            frame = step.transform(frame, X) if isinstance(step, SupervisedFeatures) else step.transform(frame)
        return frame

    def _transform_frame(self, frame):
        numeric = frame[self.numeric_columns].to_numpy(dtype=float)
        numeric[~np.isfinite(numeric)] = np.nan
        masks = frame[self.numeric_columns].isna().to_numpy(dtype=float)
        if self.numeric_columns and self.numeric_method != "native":
            if self.numeric_scaler is not None:
                numeric = self.numeric_scaler.transform(numeric)
            numeric[:, self.imputer_positions] = self.imputer.transform(numeric[:, self.imputer_positions])
            if self.numeric_scaler is not None:
                numeric = self.numeric_scaler.inverse_transform(numeric)
        for column in self.empty_columns:
            numeric[:, self.numeric_columns.index(column)] = self.spec.imputation.empty_numeric
        for column, constraint in self.spec.imputation.constraints.items():
            if column not in self.numeric_columns:
                raise ValueError("Unknown constrained feature")
            position = self.numeric_columns.index(column)
            if "min" in constraint:
                numeric[:, position] = np.maximum(numeric[:, position], constraint["min"])
            if "max" in constraint:
                numeric[:, position] = np.minimum(numeric[:, position], constraint["max"])
            if constraint.get("integer"):
                numeric[:, position] = np.rint(numeric[:, position])
        categorical = self._categorical(frame, numeric)
        if self.native:
            result = pd.DataFrame(numeric, columns=self.numeric_columns, index=frame.index)
            for column in self.categorical_columns:
                result[column] = categorical[column]
                if self.spec.model in {"lightgbm", "lgbm_ranker"}:
                    result[column] = pd.Categorical(result[column], categories=self.categories[column])
            if self.spec.imputation.indicators:
                for i, column in enumerate(self.numeric_columns):
                    result[f"{column}__missing"] = masks[:, i]
                for column in self.categorical_columns:
                    result[f"{column}__missing"] = frame[column].isna().astype(float)
            return result
        blocks = []
        if self.numeric_columns:
            for transform in self.numeric_transformers:
                numeric = transform.transform(numeric)
            blocks.append(sparse.csr_matrix(numeric))
            if self.spec.imputation.indicators:
                blocks.append(sparse.csr_matrix(masks))
        if self.categorical_columns:
            blocks.append(self.encoder.transform(categorical))
            if self.spec.imputation.indicators:
                blocks.append(sparse.csr_matrix(frame[self.categorical_columns].isna().to_numpy(dtype=float)))
        for column, vectorizer in self.vectorizers.items():
            blocks.append(vectorizer.transform(frame[column].fillna("").astype(str)))
            if self.spec.imputation.indicators:
                blocks.append(sparse.csr_matrix(frame[[column]].isna().to_numpy(dtype=float)))
        if not blocks:
            raise ValueError("No usable features")
        result = sparse.hstack(blocks, format="csr")
        if self.spec.model in {"mlp", "hist_gradient_boosting"}:
            dense_bytes = result.shape[0] * result.shape[1] * 8
            if dense_bytes > self.memory_mb * 1024 * 1024 * 0.25:
                raise MemoryError("Dense representation exceeds the memory bound")
            result = result.toarray()
        for selector in self.selector or ():
            if selector.__class__.__name__ == "DropCorrelatedFeatures":
                if sparse.issparse(result):
                    required = result.shape[0] * result.shape[1] * 8
                    if required > self.memory_mb * 1024**2 * .1:
                        raise MemoryError("Dense correlation transform exceeds the memory bound")
                    result = result.toarray()
                result = pd.DataFrame(result)
            result = selector.transform(result)
        return result

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        result = self._transform_frame(self.training_frame)
        del self.training_frame
        return result
