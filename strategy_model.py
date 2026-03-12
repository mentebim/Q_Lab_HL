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
    diagnostics: dict[str, Any]

    def summary(self) -> dict:
        return {
            "family": self.family,
            "l2_reg": self.l2_reg,
            "n_train_rows": self.n_train_rows,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "coefficients": dict(zip(self.feature_names, self.coefficients)),
            "intercept": self.intercept,
            "diagnostics": self.diagnostics,
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
    train_assets: list[str] = []
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
        train_assets.extend(aligned_assets)

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
        "train_timestamps": pd.DatetimeIndex(train_timestamps),
        "train_assets": tuple(train_assets),
        "current_timestamp": ts.isoformat(),
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
    train_timestamps: pd.DatetimeIndex | None = None,
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
    diagnostics = _fit_diagnostics(
        design=design,
        y=y,
        beta=np.asarray(beta, dtype=float),
        feature_names=feature_names,
        family=family,
        train_timestamps=train_timestamps,
    )
    return LinearModel(
        feature_names=feature_names,
        intercept=float(beta[0]),
        coefficients=tuple(float(value) for value in beta[1:]),
        family=family,
        l2_reg=float(l2_reg),
        n_train_rows=int(X.shape[0]),
        train_start=train_start,
        train_end=train_end,
        diagnostics=diagnostics,
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


def latest_snapshot(
    data,
    ts,
    *,
    execution: ExecutionConfig,
    strategy_spec: StrategySpec,
) -> dict[str, Any]:
    dataset = build_training_dataset(data, ts, execution=execution, strategy_spec=strategy_spec)
    model = fit_linear_model(
        dataset["X_train"],
        dataset["y_train"],
        feature_names=dataset["feature_names"],
        family=strategy_spec.model.family,
        l2_reg=strategy_spec.model.l2_reg,
        train_start=dataset["train_start"],
        train_end=dataset["train_end"],
        train_timestamps=dataset.get("train_timestamps"),
    )
    if model is None:
        return {
            "timestamp": pd.Timestamp(ts).isoformat(),
            "current_assets": [],
            "scores": {},
            "weights": {},
            "model_fit": {},
        }
    scores = predict_scores(
        model,
        dataset["X_now"],
        dataset["current_assets"],
        clip_predictions=strategy_spec.model.prediction_clip,
    )
    weights = construct_portfolio(scores, strategy_spec.position_bucket)
    return {
        "timestamp": pd.Timestamp(ts).isoformat(),
        "current_assets": list(scores.index),
        "scores": {asset: float(value) for asset, value in scores.sort_values(ascending=False).items()},
        "weights": {asset: float(value) for asset, value in weights.sort_values(ascending=False).items()},
        "model_fit": model.summary(),
    }


def construct_portfolio(scores: pd.Series, position_bucket: int) -> pd.Series:
    scores = pd.Series(scores, dtype=float).dropna().sort_values(ascending=False)
    if len(scores) < position_bucket * 2:
        return pd.Series(dtype=float)
    longs = scores.head(position_bucket)
    shorts = scores.tail(position_bucket)
    long_weights = longs.abs() / float(longs.abs().sum())
    short_weights = shorts.abs() / float(shorts.abs().sum())
    weights = pd.concat([0.5 * long_weights, -0.5 * short_weights])
    return weights.groupby(level=0).sum()


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
        "train_timestamps": pd.DatetimeIndex([]),
        "train_assets": tuple(),
        "current_timestamp": "",
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


def _fit_diagnostics(
    *,
    design: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    feature_names: tuple[str, ...],
    family: str,
    train_timestamps: pd.DatetimeIndex | None,
) -> dict[str, Any]:
    predictions = design @ beta
    residuals = y - predictions
    n_obs = int(len(y))
    n_features = int(len(feature_names))
    rss = float(np.square(residuals).sum())
    centered = y - float(np.mean(y))
    tss = float(np.square(centered).sum())
    r2 = 0.0 if tss <= 0.0 else max(min(1.0 - rss / tss, 1.0), -1.0)
    adj_r2 = None
    if n_obs > n_features + 1:
        adj_r2 = 1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - n_features - 1)
    rmse = float(np.sqrt(np.mean(np.square(residuals)))) if n_obs else 0.0
    mae = float(np.mean(np.abs(residuals))) if n_obs else 0.0
    pearson = _safe_corr(predictions, y)
    spearman = _safe_corr(_rank_vector(predictions), _rank_vector(y))

    diagnostics: dict[str, Any] = {
        "train_r2": float(r2),
        "train_adj_r2": None if adj_r2 is None else float(adj_r2),
        "train_rmse": rmse,
        "train_mae": mae,
        "train_prediction_target_corr": pearson,
        "train_prediction_target_rank_corr": spearman,
        "n_features": n_features,
    }

    if train_timestamps is not None and len(train_timestamps) == n_obs:
        panel = pd.DataFrame(
            {
                "timestamp": pd.DatetimeIndex(train_timestamps),
                "prediction": predictions,
                "target": y,
            }
        )
        diagnostics.update(_cross_sectional_diagnostics(panel))

    if family == "ols":
        diagnostics.update(_ols_parameter_diagnostics(design, residuals, beta, feature_names))
    return diagnostics


def _ols_parameter_diagnostics(
    design: np.ndarray,
    residuals: np.ndarray,
    beta: np.ndarray,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    n_obs, n_params = design.shape
    if n_obs <= n_params:
        return {}
    dof = n_obs - n_params
    sigma2 = float(np.square(residuals).sum() / dof)
    xtx_inv = np.linalg.pinv(design.T @ design)
    cov = sigma2 * xtx_inv
    std_err = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    t_stats = np.divide(beta, std_err, out=np.zeros_like(beta), where=std_err > 0)
    names = ("intercept",) + tuple(feature_names)
    return {
        "ols_sigma2": sigma2,
        "ols_residual_dof": int(dof),
        "parameter_std_error": {name: float(value) for name, value in zip(names, std_err)},
        "parameter_t_stat": {name: float(value) for name, value in zip(names, t_stats)},
    }


def _cross_sectional_diagnostics(panel: pd.DataFrame) -> dict[str, Any]:
    grouped = panel.groupby("timestamp", sort=True)
    pearson_values: list[float] = []
    spearman_values: list[float] = []
    spread_values: list[float] = []
    for _, frame in grouped:
        if len(frame) < 4:
            continue
        pearson = _safe_corr(frame["prediction"].to_numpy(dtype=float), frame["target"].to_numpy(dtype=float))
        spearman = _safe_corr(
            _rank_vector(frame["prediction"].to_numpy(dtype=float)),
            _rank_vector(frame["target"].to_numpy(dtype=float)),
        )
        if pearson is not None:
            pearson_values.append(float(pearson))
        if spearman is not None:
            spearman_values.append(float(spearman))
        spread = _top_bottom_spread(frame["prediction"], frame["target"])
        if spread is not None:
            spread_values.append(float(spread))
    return {
        "cross_sectional_pearson_ic_mean": _mean_or_none(pearson_values),
        "cross_sectional_pearson_ic_median": _median_or_none(pearson_values),
        "cross_sectional_pearson_ic_positive_share": _positive_share_or_none(pearson_values),
        "cross_sectional_rank_ic_mean": _mean_or_none(spearman_values),
        "cross_sectional_rank_ic_median": _median_or_none(spearman_values),
        "cross_sectional_rank_ic_positive_share": _positive_share_or_none(spearman_values),
        "top_bottom_quintile_target_spread_mean": _mean_or_none(spread_values),
        "top_bottom_quintile_target_spread_median": _median_or_none(spread_values),
    }


def _top_bottom_spread(prediction: pd.Series, target: pd.Series) -> float | None:
    frame = pd.DataFrame({"prediction": prediction, "target": target}).dropna()
    if len(frame) < 10:
        return None
    ranked = frame.sort_values("prediction", ascending=False)
    bucket = max(1, len(ranked) // 5)
    top = ranked.head(bucket)["target"]
    bottom = ranked.tail(bucket)["target"]
    if top.empty or bottom.empty:
        return None
    return float(top.mean() - bottom.mean())


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or right.size < 2 or left.size != right.size:
        return None
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return None
    left_std = float(np.std(left, ddof=1))
    right_std = float(np.std(right, ddof=1))
    if left_std == 0.0 or right_std == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _rank_vector(values: np.ndarray) -> np.ndarray:
    return pd.Series(values, dtype=float).rank(method="average").to_numpy(dtype=float)


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _median_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.median(values))


def _positive_share_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean([value > 0 for value in values]))
