from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


CACHE_DIR = Path(os.environ.get("QLAB_HL_CACHE_DIR", Path.cwd() / ".cache" / "q_lab_hl"))
DEFAULT_BARS_PER_YEAR = 24 * 365
DEFAULT_WARMUP_BARS = 24 * 14


@dataclass(frozen=True)
class ExecutionConfig:
    bars_per_year: int = DEFAULT_BARS_PER_YEAR
    target_gross_exposure: float = 1.0
    target_net_exposure: float = 0.0
    max_gross_exposure: float = 1.5
    max_abs_weight: float = 0.08
    max_group_gross: float = 0.35
    taker_fee_bps: float = 4.5
    slippage_bps: float = 3.0
    rebalance_every_bars: int = 8
    min_history_bars: int = 24 * 30
    min_dollar_volume: float = 2_000_000.0
    min_price: float = 0.5
    listing_cooldown_bars: int = 24 * 7
    reference_account_value_usd: float = 250.0
    target_margin_usage_ratio: float = 0.50
    assumed_leverage: float = 3.0
    min_trade_notional_usd: float = 10.0
    max_single_order_notional_usd: float = 250.0


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.40
    inner_fraction: float = 0.25
    outer_fraction: float = 0.20
    test_fraction: float = 0.15
