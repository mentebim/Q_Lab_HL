"""Baseline hardened quantitative strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

NUM_HOLDINGS = 60
REBALANCE_FREQ = "M"
MIN_DOLLAR_VOLUME = 10_000_000.0
TARGET_CASH_WEIGHT = 0.05
MAX_POSITION_WEIGHT = 0.035
MAX_SECTOR_WEIGHT = 0.25


def _safe_rank(data, series: pd.Series) -> pd.Series:
    series = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) == 0:
        return pd.Series(dtype=float)
    series = data.winsorize_cross_section(series, lower_pct=0.05, upper_pct=0.95)
    return data.factor_rank(series)


def _latest_or_neutral(data, field: str, date, universe: list[str]) -> pd.Series:
    try:
        values = data.latest_fundamental(field, date).reindex(universe)
    except Exception:
        values = pd.Series(0.0, index=universe)
    return values.fillna(0.0)


def signals(data, date):
    """Return a point-in-time cross-sectional alpha score."""
    universe = data.tradable_universe(
        date,
        min_history_days=252,
        min_price=5.0,
        min_dollar_volume=MIN_DOLLAR_VOLUME,
        countries=("US",),
    )
    if len(universe) < min(10, NUM_HOLDINGS):
        return pd.Series(dtype=float)

    prices = data.prices_signal(universe, end=date).iloc[-252:]
    if len(prices) < 126:
        return pd.Series(dtype=float)

    ret_12m = prices.iloc[-1] / prices.iloc[-252] - 1.0
    ret_6m = prices.iloc[-1] / prices.iloc[-126] - 1.0
    ret_1m = prices.iloc[-1] / prices.iloc[-21] - 1.0
    momentum = 0.6 * (ret_12m - ret_1m) + 0.4 * (ret_6m - ret_1m)

    value = _latest_or_neutral(data, "book_to_price", date, universe)
    profitability = _latest_or_neutral(data, "gross_profitability", date, universe)
    quality = (
        profitability.fillna(0.0)
        + 0.5 * _latest_or_neutral(data, "current_ratio", date, universe)
        - 0.5 * _latest_or_neutral(data, "asset_growth", date, universe)
    )
    low_vol = -prices.pct_change().iloc[-63:].std()
    log_mcap = np.log(data.market_cap(date).reindex(universe).replace(0, np.nan))
    sector = pd.Series({ticker: data.sector(ticker) for ticker in universe})

    ranks = pd.concat(
        [
            _safe_rank(data, momentum).rename("momentum"),
            _safe_rank(data, value.fillna(value.median() if value.notna().any() else 0.0)).rename("value"),
            _safe_rank(data, profitability.fillna(0.0)).rename("profitability"),
            _safe_rank(data, quality).rename("quality"),
            _safe_rank(data, low_vol).rename("low_vol"),
        ],
        axis=1,
    )
    score = ranks.mean(axis=1, skipna=True).reindex(universe).dropna()
    exposures = [sector]
    if log_mcap.notna().sum() >= max(10, len(score) // 3):
        exposures.append(log_mcap)
    score = data.neutralize_cross_section(score, by=exposures)
    return score.dropna().sort_values(ascending=False)


def construct(scores, data, date):
    """Convert scores into long-only target weights plus explicit cash."""
    top = scores.nlargest(NUM_HOLDINGS)
    if len(top) == 0:
        return pd.Series(dtype=float)

    prices = data.prices_signal(list(top.index), end=date).iloc[-63:]
    vol = prices.pct_change().std().replace(0, np.nan)
    inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan).dropna()
    inv_vol = inv_vol.reindex(top.index).fillna(inv_vol.median() if len(inv_vol) else 1.0)
    inv_vol = inv_vol.clip(lower=inv_vol.quantile(0.10), upper=inv_vol.quantile(0.90))

    gross_target = 1.0 - TARGET_CASH_WEIGHT
    weights = inv_vol / inv_vol.sum()
    weights = weights * gross_target
    weights.loc["__CASH__"] = TARGET_CASH_WEIGHT
    return weights


def risk(weights, data, date):
    """Apply simple concentration controls without changing net exposure."""
    if len(weights) == 0:
        return weights

    cash = float(weights.get("__CASH__", 0.0))
    asset_weights = weights.drop(labels=["__CASH__"], errors="ignore").copy()
    asset_weights = asset_weights.clip(lower=0.0, upper=MAX_POSITION_WEIGHT)

    sectors = {}
    for ticker in asset_weights.index:
        sectors.setdefault(data.sector(ticker), []).append(ticker)

    for members in sectors.values():
        sector_weight = float(asset_weights[members].sum())
        if sector_weight > MAX_SECTOR_WEIGHT:
            asset_weights.loc[members] *= MAX_SECTOR_WEIGHT / sector_weight

    gross_target = max(0.0, 1.0 - cash)
    total = float(asset_weights.sum())
    if total > 0:
        asset_weights *= gross_target / total

    final = asset_weights.copy()
    final.loc["__CASH__"] = 1.0 - float(asset_weights.sum())
    return final
