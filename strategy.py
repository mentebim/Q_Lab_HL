"""Stability-focused long-only strategy for the hardened Q_Lab harness."""

from __future__ import annotations

import numpy as np
import pandas as pd

NUM_HOLDINGS = 100
REBALANCE_FREQ = "M"
MIN_DOLLAR_VOLUME = 20_000_000.0
MAX_POSITION_WEIGHT = 0.020
MAX_SECTOR_WEIGHT = 0.25
ENTRY_RANK = NUM_HOLDINGS
EXIT_RANK = 195

_PREV_SELECTION: list[str] = []
_PREV_TARGET_WEIGHTS = pd.Series(dtype=float)


def _safe_rank(data, series: pd.Series) -> pd.Series:
    series = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return pd.Series(dtype=float)
    series = data.winsorize_cross_section(series, lower_pct=0.05, upper_pct=0.95)
    return data.factor_rank(series)


def _latest_or_neutral(data, field: str, date, universe: list[str]) -> pd.Series:
    try:
        values = data.latest_fundamental(field, date).reindex(universe)
    except Exception:
        values = pd.Series(0.0, index=universe)
    return values.fillna(0.0)


def _buffered_selection(scores: pd.Series) -> pd.Series:
    global _PREV_SELECTION

    ranked = pd.Series(scores).dropna().sort_values(ascending=False)
    if ranked.empty:
        _PREV_SELECTION = []
        return ranked

    rank_map = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
    keep = [ticker for ticker in _PREV_SELECTION if ticker in rank_map.index and int(rank_map[ticker]) <= EXIT_RANK]
    keep = sorted(dict.fromkeys(keep), key=lambda ticker: int(rank_map[ticker]))

    selected = keep[:NUM_HOLDINGS]
    entrants = [ticker for ticker in ranked.head(ENTRY_RANK).index if ticker not in selected]
    needed = max(0, NUM_HOLDINGS - len(selected))
    selected.extend(entrants[:needed])

    if len(selected) < NUM_HOLDINGS:
        filler = [ticker for ticker in ranked.index if ticker not in selected]
        selected.extend(filler[: NUM_HOLDINGS - len(selected)])

    _PREV_SELECTION = selected[:NUM_HOLDINGS]
    return ranked.reindex(_PREV_SELECTION).dropna()


def signals(data, date):
    """Return a quality-tilted cross-sectional alpha score, sector-neutralized only."""
    universe = data.tradable_universe(
        date,
        min_history_days=252,
        min_price=10.0,
        min_dollar_volume=MIN_DOLLAR_VOLUME,
        countries=("US",),
    )
    if len(universe) < min(20, NUM_HOLDINGS):
        return pd.Series(dtype=float)

    prices = data.prices_signal(universe, end=date).iloc[-252:]
    if len(prices) < 190:
        return pd.Series(dtype=float)

    # Risk-adjusted momentum: 12-1m return / volatility, trend-penalized
    ret_12_1 = prices.iloc[-21] / prices.iloc[0] - 1.0
    vol_12m = prices.pct_change().iloc[-252:].std().replace(0, np.nan)
    risk_adj_mom = (ret_12_1 / vol_12m).replace([np.inf, -np.inf], np.nan)
    ma_200 = prices.iloc[-200:].mean()
    above_trend = (prices.iloc[-1] > ma_200).astype(float)
    momentum = risk_adj_mom * (0.5 + 0.5 * above_trend)

    # Quality composite: profitability + ROE + earnings yield
    roe = _latest_or_neutral(data, "roe", date, universe)
    profitability = _latest_or_neutral(data, "gross_profitability", date, universe)
    earnings_yield = _latest_or_neutral(data, "earnings_yield", date, universe)
    quality = 0.4 * profitability + 0.3 * roe + 0.3 * earnings_yield

    # FCF yield
    free_cash_flow_yield = _latest_or_neutral(data, "free_cash_flow_yield", date, universe)

    # Low volatility (6-month window)
    low_vol = -prices.pct_change().iloc[-126:].std()

    # Sector for neutralization (no size neutralization)
    sector = pd.Series({ticker: data.sector(ticker) for ticker in universe})

    ranks = pd.concat(
        [
            _safe_rank(data, momentum).rename("momentum"),
            _safe_rank(data, quality).rename("quality"),
            _safe_rank(
                data,
                free_cash_flow_yield.fillna(
                    free_cash_flow_yield.median() if free_cash_flow_yield.notna().any() else 0.0
                ),
            ).rename("fcf_yield"),
            _safe_rank(data, low_vol).rename("low_vol"),
        ],
        axis=1,
    )

    score = (
        0.30 * ranks["momentum"]
        + 0.35 * ranks["quality"]
        + 0.20 * ranks["fcf_yield"]
        + 0.15 * ranks["low_vol"]
    ).reindex(universe).dropna()

    score = data.neutralize_cross_section(score, by=[sector])
    return score.dropna().sort_values(ascending=False)


def construct(scores, data, date):
    """Convert scores into diversified target weights with buffered membership."""
    global _PREV_TARGET_WEIGHTS

    selected = _buffered_selection(scores)
    if selected.empty:
        _PREV_TARGET_WEIGHTS = pd.Series(dtype=float)
        return pd.Series(dtype=float)

    current = pd.Series(1.0 / len(selected), index=selected.index, dtype=float)
    prev = _PREV_TARGET_WEIGHTS.reindex(selected.index).fillna(0.0)
    if float(prev.sum()) > 0:
        prev /= float(prev.sum())
        weights = 0.35 * prev + 0.65 * current
    else:
        weights = current
    weights /= float(weights.sum())
    _PREV_TARGET_WEIGHTS = weights.copy()
    weights.loc["__CASH__"] = 0.0
    return weights


def risk(weights, data, date):
    """Apply position and sector caps."""
    if len(weights) == 0:
        return weights

    cash = float(weights.get("__CASH__", 0.0))
    asset_weights = weights.drop(labels=["__CASH__"], errors="ignore").copy()
    asset_weights = asset_weights.clip(lower=0.0, upper=MAX_POSITION_WEIGHT)

    sectors: dict[str, list[str]] = {}
    for ticker in asset_weights.index:
        sectors.setdefault(data.sector(ticker), []).append(ticker)

    for members in sectors.values():
        sector_weight = float(asset_weights[members].sum())
        if sector_weight > MAX_SECTOR_WEIGHT:
            asset_weights.loc[members] *= MAX_SECTOR_WEIGHT / sector_weight

    total = float(asset_weights.sum())
    if total > 0:
        asset_weights /= total
        asset_weights *= max(0.0, 1.0 - cash)

    final = asset_weights.copy()
    final.loc["__CASH__"] = 1.0 - float(asset_weights.sum())
    return final
