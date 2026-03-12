from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from q_lab_hl.config import ExecutionConfig
from q_lab_hl.data import default_universe_kwargs

_FRAME_CACHE: dict[tuple[int, tuple[tuple[str, str, int], ...]], dict] = {}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    lookback: int
    transform: str = "zscore"
    clip: float | None = 3.0


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str


@dataclass(frozen=True)
class ModelSpec:
    family: str
    l2_reg: float = 0.0
    prediction_clip: float = 3.0


@dataclass(frozen=True)
class StrategySpec:
    train_window_bars: int
    min_train_rows: int
    position_bucket: int
    features: tuple[FeatureSpec, ...]
    target: TargetSpec
    model: ModelSpec

    def summary(self) -> dict:
        return asdict(self)


def strategy_spec_from_dict(payload: dict[str, Any], base: StrategySpec) -> StrategySpec:
    features_payload = payload.get("features")
    features = base.features if features_payload is None else tuple(feature_spec_from_dict(item) for item in features_payload)
    target = base.target if payload.get("target") is None else target_spec_from_dict(payload["target"])
    model = base.model if payload.get("model") is None else model_spec_from_dict(payload["model"])
    return StrategySpec(
        train_window_bars=int(payload.get("train_window_bars", base.train_window_bars)),
        min_train_rows=int(payload.get("min_train_rows", base.min_train_rows)),
        position_bucket=int(payload.get("position_bucket", base.position_bucket)),
        features=features,
        target=target,
        model=model,
    )


def feature_spec_from_dict(payload: dict[str, Any]) -> FeatureSpec:
    return FeatureSpec(
        name=str(payload["name"]),
        kind=str(payload["kind"]),
        lookback=int(payload["lookback"]),
        transform=str(payload.get("transform", "zscore")),
        clip=None if payload.get("clip") is None else float(payload.get("clip")),
    )


def target_spec_from_dict(payload: dict[str, Any]) -> TargetSpec:
    return TargetSpec(name=str(payload["name"]), kind=str(payload["kind"]))


def model_spec_from_dict(payload: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        family=str(payload["family"]),
        l2_reg=float(payload.get("l2_reg", 0.0)),
        prediction_clip=float(payload.get("prediction_clip", 3.0)),
    )


@dataclass(frozen=True)
class LinearModel:
    feature_names: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    family: str
    l2_reg: float
    n_train_rows: int
    train_start: str
    train_end: str

    def summary(self) -> dict:
        return {
            "family": self.family,
            "l2_reg": self.l2_reg,
            "n_train_rows": self.n_train_rows,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "coefficients": dict(zip(self.feature_names, self.coefficients)),
            "intercept": self.intercept,
        }


def build_training_dataset(
    data,
    ts,
    *,
    execution: ExecutionConfig,
    strategy_spec: StrategySpec,
) -> dict:
    ts = pd.Timestamp(ts)
    base_store = getattr(data, "base", data)
    min_assets = strategy_spec.position_bucket * 2
    cached = _get_cached_frames(base_store, strategy_spec.features, strategy_spec.target)
    close = cached["close"]
    if close.empty or ts not in close.index:
        return _empty_dataset(strategy_spec.features)
    feature_frames = cached["feature_frames"]
    target_frame = cached["target_frame"]
    index = close.index
    current_pos = int(index.get_loc(ts))
    if current_pos <= 0:
        return _empty_dataset(strategy_spec.features)
    train_start_pos = max(0, current_pos - strategy_spec.train_window_bars)
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    train_timestamps: list[pd.Timestamp] = []
    universe_kwargs = default_universe_kwargs(execution)

    for pos in range(train_start_pos, current_pos):
        sample_ts = index[pos]
        next_ts = index[pos + 1]
        universe = data.tradable_universe(sample_ts, **universe_kwargs)
        if len(universe) < min_assets:
            continue
        features = _current_feature_matrix(feature_frames, strategy_spec.features, sample_ts, universe)
        if features.empty:
            continue
        valid_assets = [
            asset
            for asset in features.index
            if data.can_trade(asset, next_ts) and pd.notna(target_frame.at[sample_ts, asset])
        ]
        if len(valid_assets) < min_assets:
            continue
        feature_block = features.loc[valid_assets]
        target_block = pd.Series(target_frame.loc[sample_ts, valid_assets], dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        aligned_assets = [asset for asset in feature_block.index if asset in target_block.index]
        if len(aligned_assets) < min_assets:
            continue
        x_rows.append(feature_block.loc[aligned_assets].to_numpy(dtype=float))
        y_rows.append(target_block.loc[aligned_assets].to_numpy(dtype=float))
        train_timestamps.extend([sample_ts] * len(aligned_assets))

    current_universe = data.tradable_universe(ts, **universe_kwargs)
    current_features = _current_feature_matrix(feature_frames, strategy_spec.features, ts, current_universe)
    if current_features.empty or len(current_features) < min_assets:
        current_features = pd.DataFrame(columns=[spec.name for spec in strategy_spec.features], dtype=float)

    if x_rows:
        x_train = np.vstack(x_rows)
        y_train = np.concatenate(y_rows)
        train_start = min(train_timestamps).isoformat()
        train_end = max(train_timestamps).isoformat()
    else:
        x_train = np.empty((0, len(strategy_spec.features)), dtype=float)
        y_train = np.empty(0, dtype=float)
        train_start = ""
        train_end = ""

    return {
        "feature_names": tuple(spec.name for spec in strategy_spec.features),
        "X_train": x_train,
        "y_train": y_train,
        "X_now": current_features.to_numpy(dtype=float),
        "current_assets": list(current_features.index),
        "train_start": train_start,
        "train_end": train_end,
    }


def fit_linear_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    family: str,
    l2_reg: float,
    train_start: str,
    train_end: str,
) -> LinearModel | None:
    if len(X_train) == 0 or len(y_train) == 0:
        return None
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        return None
    design = np.column_stack([np.ones(X.shape[0], dtype=float), X])
    if family == "ridge":
        penalty = np.eye(design.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + float(l2_reg) * penalty, design.T @ y)
    elif family == "ols":
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    else:
        raise ValueError(f"Unsupported model family '{family}'")
    return LinearModel(
        feature_names=feature_names,
        intercept=float(beta[0]),
        coefficients=tuple(float(value) for value in beta[1:]),
        family=family,
        l2_reg=float(l2_reg),
        n_train_rows=int(X.shape[0]),
        train_start=train_start,
        train_end=train_end,
    )


def predict_scores(
    model: LinearModel | None,
    X_now: np.ndarray,
    current_assets: list[str],
    *,
    clip_predictions: float,
) -> pd.Series:
    if model is None or len(current_assets) == 0:
        return pd.Series(dtype=float)
    X = np.asarray(X_now, dtype=float)
    if X.ndim != 2 or X.shape[0] != len(current_assets):
        return pd.Series(dtype=float)
    coeffs = np.asarray(model.coefficients, dtype=float)
    predictions = model.intercept + X @ coeffs
    scores = pd.Series(predictions, index=current_assets, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if scores.empty:
        return scores
    scores = scores.clip(lower=-clip_predictions, upper=clip_predictions)
    return scores - float(scores.mean())


def _build_feature_frames(
    close: pd.DataFrame,
    funding: pd.DataFrame,
    feature_specs: tuple[FeatureSpec, ...],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    returns_1h = close.pct_change(fill_method=None)
    for spec in feature_specs:
        if spec.kind == "return":
            frames[spec.name] = close / close.shift(spec.lookback) - 1.0
        elif spec.kind == "volatility":
            frames[spec.name] = returns_1h.rolling(spec.lookback).std()
        elif spec.kind == "ma_gap":
            moving_average = close.rolling(spec.lookback).mean()
            frames[spec.name] = close / moving_average - 1.0
        elif spec.kind == "funding_mean":
            if funding.empty:
                frames[spec.name] = pd.DataFrame(0.0, index=close.index, columns=close.columns)
            else:
                frames[spec.name] = funding.rolling(spec.lookback).mean().reindex_like(close).fillna(0.0)
        else:
            raise ValueError(f"Unsupported feature kind '{spec.kind}'")
    return frames


def _current_feature_matrix(
    feature_frames: dict[str, pd.DataFrame],
    feature_specs: tuple[FeatureSpec, ...],
    ts,
    universe: list[str],
) -> pd.DataFrame:
    if len(universe) == 0:
        return pd.DataFrame()
    rows = {}
    for name, frame in feature_frames.items():
        if ts not in frame.index:
            return pd.DataFrame()
        rows[name] = pd.Series(frame.loc[ts, universe], dtype=float)
    features = pd.DataFrame(rows, index=universe).replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        return features
    standardized = pd.DataFrame(index=features.index)
    spec_by_name = {spec.name: spec for spec in feature_specs}
    for column in features.columns:
        values = pd.Series(features[column], dtype=float)
        spec = spec_by_name[column]
        if spec.transform == "none":
            standardized[column] = values
            continue
        if spec.transform == "rank":
            ranked = values.rank(method="average", pct=True) - 0.5
            standardized[column] = ranked
            continue
        std = float(values.std(ddof=1))
        if not np.isfinite(std) or std == 0.0:
            standardized[column] = 0.0
            continue
        zscore = (values - float(values.mean())) / std
        clip = spec.clip if spec.clip is not None else np.inf
        standardized[column] = zscore.clip(lower=-clip, upper=clip)
    return standardized.dropna()


def _empty_dataset(feature_specs: tuple[FeatureSpec, ...]) -> dict:
    return {
        "feature_names": tuple(spec.name for spec in feature_specs),
        "X_train": np.empty((0, len(feature_specs)), dtype=float),
        "y_train": np.empty(0, dtype=float),
        "X_now": np.empty((0, len(feature_specs)), dtype=float),
        "current_assets": [],
        "train_start": "",
        "train_end": "",
    }


def _get_cached_frames(base_store, feature_specs: tuple[FeatureSpec, ...], target_spec: TargetSpec) -> dict:
    key = (
        id(base_store),
        tuple((spec.name, spec.kind, spec.lookback) for spec in feature_specs),
        target_spec.kind,
    )
    cached = _FRAME_CACHE.get(key)
    if cached is not None:
        return cached
    close = base_store.prices(field="close")
    open_ = base_store.prices(field="open")
    funding = base_store.funding()
    feature_frames = _build_feature_frames(close, funding, feature_specs)
    target_frame = _build_target_frame(close, open_, target_spec)
    cached = {
        "close": close,
        "feature_frames": feature_frames,
        "target_frame": target_frame,
    }
    _FRAME_CACHE[key] = cached
    return cached


def _build_target_frame(close: pd.DataFrame, open_: pd.DataFrame, target_spec: TargetSpec) -> pd.DataFrame:
    if target_spec.kind == "next_open_to_close_return":
        return close.shift(-1) / open_.shift(-1) - 1.0
    if target_spec.kind == "next_close_to_close_return":
        return close.shift(-1) / close - 1.0
    raise ValueError(f"Unsupported target kind '{target_spec.kind}'")
