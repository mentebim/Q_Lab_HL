"""Baseline Hyperliquid long-short residual reversal strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from q_lab_hl.config import ExecutionConfig
from q_lab_hl.data import default_universe_kwargs


EXECUTION = ExecutionConfig(
    min_history_bars=24 * 14,
    min_dollar_volume=500_000.0,
    min_price=0.10,
    listing_cooldown_bars=24 * 3,
)
MODEL_FAMILY = "residual_reversal"
POSITION_BUCKET = 4
LOOKBACK_SHORT_HOURS = 6
LOOKBACK_MEDIUM_HOURS = 24
FUNDING_WINDOW_HOURS = 8
LIQUIDITY_WINDOW_HOURS = 24
VOL_WINDOW_HOURS = 72
BETA_WINDOW_HOURS = 72
MOMENTUM_CONFIRM_WEIGHT = 0.20
RESIDUAL_WEIGHT = 0.60
MEDIUM_WEIGHT = 0.20
FUNDING_WEIGHT = 0.20
PARAM_GRID = {
    "MODEL_FAMILY": ["residual_reversal", "funding_dislocation", "beta_neutral_momentum"],
    "POSITION_BUCKET": [3, 4, 5],
    "LOOKBACK_SHORT_HOURS": [6, 12],
    "LOOKBACK_MEDIUM_HOURS": [24, 48],
    "FUNDING_WINDOW_HOURS": [8],
    "VOL_WINDOW_HOURS": [48, 72],
}


def reset_state():
    return None


def signals(data, ts):
    universe, closes = _load_universe_prices(data, ts)
    if len(universe) < POSITION_BUCKET * 2 + 4 or closes.empty:
        return pd.Series(dtype=float)
    family = str(MODEL_FAMILY)
    if family == "residual_reversal":
        score = _signals_residual_reversal(data, ts, universe, closes)
    elif family == "funding_dislocation":
        score = _signals_funding_dislocation(data, ts, universe, closes)
    elif family == "beta_neutral_momentum":
        score = _signals_beta_neutral_momentum(data, ts, universe, closes)
    else:
        raise ValueError(f"Unknown MODEL_FAMILY '{family}'")
    return _finalize_scores(data, ts, universe, score).sort_values(ascending=False)


def construct(scores, data, ts):
    scores = pd.Series(scores, dtype=float).dropna().sort_values(ascending=False)
    if len(scores) < POSITION_BUCKET * 2:
        return pd.Series(dtype=float)
    longs = scores.head(POSITION_BUCKET)
    shorts = scores.tail(POSITION_BUCKET)
    long_weights = longs.abs() / float(longs.abs().sum())
    short_weights = shorts.abs() / float(shorts.abs().sum())
    weights = pd.concat([0.5 * long_weights, -0.5 * short_weights])
    return weights.groupby(level=0).sum()


def risk(weights, data, ts):
    return pd.Series(weights, dtype=float)


def _load_universe_prices(data, ts) -> tuple[list[str], pd.DataFrame]:
    universe = data.tradable_universe(ts, **default_universe_kwargs(EXECUTION))
    required_bars = max(LOOKBACK_SHORT_HOURS, LOOKBACK_MEDIUM_HOURS, VOL_WINDOW_HOURS, BETA_WINDOW_HOURS) + 1
    if len(universe) < POSITION_BUCKET * 2 + 4:
        return universe, pd.DataFrame()
    closes = data.prices(universe, end=ts).iloc[-required_bars:]
    if len(closes) < required_bars:
        return universe, pd.DataFrame()
    return universe, closes


def _signals_residual_reversal(data, ts, universe: list[str], closes: pd.DataFrame) -> pd.Series:
    short_return = _horizon_return(closes, LOOKBACK_SHORT_HOURS)
    medium_return = _horizon_return(closes, LOOKBACK_MEDIUM_HOURS)
    market_short = float(short_return.mean())
    residual = short_return - market_short
    funding = _funding_mean(data, universe, ts)
    return (
        -RESIDUAL_WEIGHT * data.zscore_cross_section(residual)
        -MEDIUM_WEIGHT * data.zscore_cross_section(medium_return)
        -FUNDING_WEIGHT * data.zscore_cross_section(funding)
    )


def _signals_funding_dislocation(data, ts, universe: list[str], closes: pd.DataFrame) -> pd.Series:
    short_return = _horizon_return(closes, LOOKBACK_SHORT_HOURS)
    medium_return = _horizon_return(closes, LOOKBACK_MEDIUM_HOURS)
    funding = _funding_mean(data, universe, ts)
    funding_z = data.zscore_cross_section(funding)
    short_z = data.zscore_cross_section(short_return)
    medium_z = data.zscore_cross_section(medium_return)
    # Crowded positive funding plus stretched price is a fade; deeply negative funding plus weak price is the long leg.
    dislocation = 0.55 * funding_z + 0.30 * short_z + 0.15 * medium_z
    return -dislocation


def _signals_beta_neutral_momentum(data, ts, universe: list[str], closes: pd.DataFrame) -> pd.Series:
    medium_return = _horizon_return(closes, LOOKBACK_MEDIUM_HOURS)
    short_return = _horizon_return(closes, LOOKBACK_SHORT_HOURS)
    asset_returns = closes.pct_change().iloc[-BETA_WINDOW_HOURS:].dropna(how="all")
    if asset_returns.empty:
        return pd.Series(dtype=float)
    market_returns = asset_returns.mean(axis=1)
    beta = _rolling_beta(asset_returns, market_returns).reindex(universe).fillna(1.0)
    vol = asset_returns.std().replace(0.0, np.nan).reindex(universe)
    momentum = (medium_return / vol).replace([np.inf, -np.inf], np.nan)
    score = data.zscore_cross_section(momentum) + MOMENTUM_CONFIRM_WEIGHT * data.zscore_cross_section(short_return)
    # Strip market beta directly from the alpha vector before the generic neutralization pass.
    return data.neutralize_cross_section(score.fillna(0.0), by=[beta])


def _finalize_scores(data, ts, universe: list[str], raw_score: pd.Series) -> pd.Series:
    score = pd.Series(raw_score, dtype=float).reindex(universe).dropna()
    if score.empty:
        return score
    score = data.winsorize_cross_section(score, 0.05, 0.95)
    sector = pd.Series({asset: data.sector(asset) for asset in score.index})
    liquidity = _liquidity_exposure(data, ts, score.index)
    return data.neutralize_cross_section(score, by=[sector, liquidity.reindex(score.index).fillna(0.0)])


def _liquidity_exposure(data, ts, universe) -> pd.Series:
    dollar_volume = data.dollar_volume(LIQUIDITY_WINDOW_HOURS, ts).reindex(universe)
    return data.zscore_cross_section(dollar_volume.replace(0.0, pd.NA).dropna())


def _funding_mean(data, universe: list[str], ts) -> pd.Series:
    funding_panel = data.funding(universe, end=ts)
    if funding_panel.empty:
        return pd.Series(0.0, index=universe, dtype=float)
    return funding_panel.iloc[-min(FUNDING_WINDOW_HOURS, len(funding_panel)) :].mean().reindex(universe).fillna(0.0)


def _horizon_return(closes: pd.DataFrame, lookback_hours: int) -> pd.Series:
    if len(closes) < lookback_hours + 1:
        return pd.Series(dtype=float)
    return closes.iloc[-1] / closes.iloc[-(lookback_hours + 1)] - 1.0


def _rolling_beta(asset_returns: pd.DataFrame, market_returns: pd.Series) -> pd.Series:
    market_var = float(pd.Series(market_returns).var(ddof=1))
    if not np.isfinite(market_var) or market_var == 0:
        return pd.Series(1.0, index=asset_returns.columns, dtype=float)
    betas = {}
    for asset in asset_returns.columns:
        pair = pd.concat([asset_returns[asset], market_returns], axis=1).dropna()
        if len(pair) < 5:
            betas[asset] = 1.0
            continue
        betas[asset] = float(pair.iloc[:, 0].cov(pair.iloc[:, 1]) / market_var)
    return pd.Series(betas, dtype=float)
