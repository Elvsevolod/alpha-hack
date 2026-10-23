from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .api import prediction_bundle
from .contracts import TrainingSummary
from .pipelines import DirectPipeline
from .validation import ordered_periods


class ForecastPipeline:
    def __init__(self, spec, task):
        self.spec, self.task = spec, task
        self.series_columns = task.series_columns

    def _series_key(self, row):
        return tuple(str(row[c]) for c in self.series_columns) if self.series_columns else ("all",)

    def _time(self, X):
        values = X[self.task.time_column]
        if pd.api.types.is_datetime64_any_dtype(values):
            return pd.to_datetime(values, utc=True)
        return ordered_periods(values)

    def _features(self, row, history):
        features = {}
        for target in self.task.targets:
            values = [item[1][target] for item in history]
            for lag in self.lags:
                features[f"{target}__lag_{lag}"] = values[-lag] if len(values) >= lag else np.nan
            features[f"{target}__rolling_mean"] = float(np.mean(values[-self.window:])) if values else np.nan
        stamp = row[self.task.time_column]
        try:
            features["time_value"] = float(stamp)
        except (ValueError, TypeError):
            date = pd.Timestamp(stamp)
            features.update({"month": date.month, "day": date.day, "dayofweek": date.dayofweek,
                "time_value": date.timestamp() / 86400})
        for column in self.series_columns:
            features[f"series__{column}"] = str(row[column])
        for column in self.task.feature_columns:
            if column not in self.series_columns and column != self.task.time_column:
                features[f"raw__{column}"] = row[column]
        return features

    def fit(self, X, y, context):
        started = time.monotonic()
        self.history, self.models = {}, {}
        self.lags = sorted(set(self.spec.hyperparameters.get("lags", [1, 2, 3])))
        if not self.lags or min(self.lags) < 1:
            raise ValueError("Forecast lags must be positive")
        self.window = self.spec.hyperparameters.get("window", 3)
        self.global_mean = {target: float(y[target].mean()) for target in self.task.targets}
        records, targets, keys = [], [], []
        times = self._time(X)
        ordered = np.argsort(times.to_numpy(), kind="stable")
        seen = set()
        for position in ordered:
            row = X.iloc[position]
            key, stamp = self._series_key(row), times.iloc[position]
            if (key, stamp) in seen:
                raise ValueError("Each series must have one observation per timestamp")
            seen.add((key, stamp))
            history = self.history.setdefault(key, [])
            if history:
                records.append(self._features(row, history))
                targets.append(y.iloc[position].to_dict())
                keys.append(X.index[position])
            history.append((stamp, y.iloc[position].to_dict(), row.to_dict()))
        for key, history in self.history.items():
            if self.spec.model in {"exponential_smoothing", "sarimax"}:
                for target in self.task.targets:
                    values = [item[1][target] for item in history]
                    if len(values) < 4:
                        continue
                    if self.spec.model == "exponential_smoothing":
                        from statsmodels.tsa.holtwinters import ExponentialSmoothing
                        model = ExponentialSmoothing(values, trend=self.spec.hyperparameters.get("trend", None),
                            seasonal=self.spec.hyperparameters.get("seasonal", None),
                            seasonal_periods=self.spec.hyperparameters.get("seasonal_periods", None)).fit()
                    else:
                        from statsmodels.tsa.statespace.sarimax import SARIMAX
                        model = SARIMAX(values, order=tuple(self.spec.hyperparameters.get("order", [1, 0, 0])),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=50)
                    self.models[(key, target)] = model
        if self.spec.model not in {"naive", "constant", "sarimax", "exponential_smoothing"}:
            if not records:
                raise ValueError("No series has enough history for lag training")
            features = pd.DataFrame(records, index=keys)
            target_frame = pd.DataFrame(targets, index=keys)
            adapted = self.task.model_copy(update={"kind": "multioutput_regression" if len(self.task.targets) > 1 else "regression",
                "feature_columns": list(features), "categorical_columns": [c for c in features if c.startswith("series__")],
                "text_columns": [], "series_columns": [], "time_column": None})
            parameters = {k: v for k, v in self.spec.hyperparameters.items() if k not in {"lags", "window"}}
            inner = self.spec.model_copy(update={"representation": "tabular", "hyperparameters": parameters})
            self.regressor = DirectPipeline(inner, adapted)
            self.regressor.fit(features, target_frame, context)
        self.summary = TrainingSummary(rows=len(X), seconds=time.monotonic() - started,
            selected_parameters={"model": self.spec.model, "lags": self.lags, "hyperparameters": self.spec.hyperparameters},
            imputation={"scope": "causal_history_within_series", "unknown_series_fallback": "fit_mean",
                "unknown_exogenous_gap_policy": "last_available_features_no_future_targets"})
        return self.summary

    def predict(self, X, context):
        times = self._time(X)
        result = {target: np.empty(len(X), dtype=float) for target in self.task.targets}
        grouped = {}
        for i in np.argsort(times.to_numpy(), kind="stable"):
            grouped.setdefault(self._series_key(X.iloc[i]), []).append(int(i))
        for key, positions in grouped.items():
            history = list(self.history.get(key, []))
            if not history:
                for target in result:
                    result[target][positions] = self.global_mean[target]
                continue
            last_stamp = history[-1][0]
            if any(times.iloc[i] <= last_stamp for i in positions):
                raise ValueError("Forecast contains timestamps at or before the training boundary")
            differences = [b[0] - a[0] for a, b in zip(history, history[1:])]
            step = differences[0] if differences and all(d == differences[0] for d in differences) else None
            if isinstance(last_stamp, pd.Timestamp):
                frequency = pd.infer_freq(pd.DatetimeIndex([item[0] for item in history])) if len(history) >= 3 else None
                step = None
                if frequency is not None:
                    step = pd.tseries.frequencies.to_offset(frequency)
            if step is not None and last_stamp + step <= last_stamp:
                raise ValueError("Non-increasing series time")
            requested = {times.iloc[i]: i for i in positions}
            if len(requested) != len(positions):
                raise ValueError("Each series must have one forecast row per timestamp")
            stamps = sorted(requested)
            if step is not None:
                maximum = max(stamps)
                cursor = last_stamp + step
                intermediate = []
                while cursor < maximum:
                    intermediate.append(cursor)
                    if len(intermediate) > 10000:
                        raise ValueError("Forecast gap exceeds the bounded recursive horizon")
                    cursor += step
                grid = set(intermediate) | {cursor}
                if not set(requested) <= grid:
                    raise ValueError("Forecast timestamps do not follow the training frequency")
                stamps = sorted(grid)
            classical_forecasts = {target: np.asarray(model.forecast(steps=len(stamps)))
                for (series_key, target), model in self.models.items() if series_key == key}
            for forecast_index, stamp in enumerate(stamps):
                position = requested.get(stamp)
                row = X.iloc[position].astype(object) if position is not None else pd.Series(history[-1][2], dtype=object)
                row[self.task.time_column] = stamp
                if hasattr(self, "regressor"):
                    features = pd.DataFrame([self._features(row, history)], index=["recursive"])
                    prediction = self.regressor.predict(features, context)
                    values = {target: float(prediction.values[target][0]) for target in result}
                else:
                    values = {}
                    for target in result:
                        model = self.models.get((key, target))
                        if model is not None:
                            value = float(classical_forecasts[target][forecast_index])
                        else:
                            seasonal = max(1, int(self.spec.hyperparameters.get("seasonal_period", 1)))
                            value = history[-min(seasonal, len(history))][1][target]
                        values[target] = value
                history.append((stamp, values, row.to_dict()))
                if position is not None:
                    for target in result:
                        result[target][position] = values[target]
        return prediction_bundle(result, X, context, "values")

    def frozen_spec(self):
        return self.spec
