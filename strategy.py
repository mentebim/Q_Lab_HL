"""Quantitative strategy. Agent modifies this file."""

import numpy as np
import pandas as pd

# === CONFIGURATION ===
NUM_HOLDINGS = 75
REBALANCE_FREQ = "M"  # monthly
LONG_ONLY = True

# === SIGNAL GENERATION ===
def signals(data, date):
    """Score each ticker. Higher = more bullish. Returns pd.Series."""
    universe = data.universe(date)
    prices = data.prices(universe)
    prices_to_date = prices.loc[:date]

    if len(prices_to_date) < 252:
        return pd.Series(dtype=float)

    # Filter: US large-caps only
    us_tickers = [t for t in universe if data.country(t) in ("US", "")]
    if len(us_tickers) < 50:
        us_tickers = universe
    prices_to_date = prices_to_date[
        [t for t in us_tickers if t in prices_to_date.columns]
    ]

    # Momentum: 12-month return minus 1-month return
    ret_12m = prices_to_date.iloc[-1] / prices_to_date.iloc[-252] - 1
    ret_1m = prices_to_date.iloc[-1] / prices_to_date.iloc[-21] - 1
    momentum = ret_12m - ret_1m

    # Value: earnings yield
    try:
        value = data.fundamental("earnings_yield")
        value = value.reindex(prices_to_date.columns)
    except Exception:
        value = pd.Series(0.0, index=prices_to_date.columns)

    # Quality: ROE
    try:
        roe = data.fundamental("roe")
        roe = roe.reindex(prices_to_date.columns)
    except Exception:
        roe = pd.Series(0.0, index=prices_to_date.columns)

    # Quality: Piotroski F-score
    try:
        piotroski = data.fundamental("piotroski")
        piotroski = piotroski.reindex(prices_to_date.columns)
    except Exception:
        piotroski = pd.Series(0.0, index=prices_to_date.columns)

    # Combine: rank-based
    mom_rank = momentum.rank(pct=True)
    val_rank = value.rank(pct=True)
    roe_rank = roe.rank(pct=True)
    pio_rank = piotroski.rank(pct=True)

    # Quality composite
    quality_rank = 0.5 * roe_rank + 0.5 * pio_rank

    score = 0.4 * mom_rank + 0.3 * val_rank + 0.3 * quality_rank
    return score.dropna()

# === PORTFOLIO CONSTRUCTION ===
def construct(scores, data, date):
    """Convert scores to target weights. Returns pd.Series."""
    top = scores.nlargest(NUM_HOLDINGS)

    # Inverse-volatility weighting
    prices = data.prices(list(top.index))
    prices_to_date = prices.loc[:date]
    if len(prices_to_date) >= 63:
        rets = prices_to_date.pct_change().iloc[-63:]
        vol = rets.std()
        vol = vol.reindex(top.index).fillna(vol.median())
        vol = vol.clip(lower=vol.quantile(0.05))
        inv_vol = 1.0 / vol
        weights = inv_vol / inv_vol.sum()
    else:
        weights = pd.Series(1.0 / len(top), index=top.index)

    return weights

# === RISK MANAGEMENT ===
def risk(weights, data, date):
    """Apply risk constraints. Returns pd.Series."""
    # 3% max per position
    weights = weights.clip(upper=0.03)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights
