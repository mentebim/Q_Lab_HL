from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeInstruction:
    coin: str
    price: float
    current_size: float
    target_size: float
    delta_size: float
    current_notional_usd: float
    target_notional_usd: float
    delta_notional_usd: float
    side: str
    status: str
    reason: str = ""

    def summary(self) -> dict:
        return asdict(self)


def build_trade_instructions(
    target_weights: pd.Series,
    prices: pd.Series,
    current_positions: dict[str, float],
    *,
    account_value: float,
    size_decimals: dict[str, int],
    min_trade_notional_usd: float,
    max_single_order_notional_usd: float,
) -> list[TradeInstruction]:
    weights = pd.Series(target_weights, dtype=float)
    mids = pd.Series(prices, dtype=float)
    instructions: list[TradeInstruction] = []
    coins = sorted(set(weights.index) | set(current_positions) | set(mids.index))
    for coin in coins:
        price = float(mids.get(coin, np.nan))
        if not np.isfinite(price) or price <= 0.0:
            continue
        current_size = float(current_positions.get(coin, 0.0))
        target_notional = float(weights.get(coin, 0.0) * account_value)
        target_size = _round_toward_zero(target_notional / price, size_decimals.get(coin, 4))
        delta_size = _round_toward_zero(target_size - current_size, size_decimals.get(coin, 4))
        current_notional = current_size * price
        delta_notional = delta_size * price
        side = "buy" if delta_size > 0 else "sell" if delta_size < 0 else "flat"
        status = "trade"
        reason = ""
        if abs(delta_notional) < min_trade_notional_usd or abs(delta_size) <= 0.0:
            status = "skip"
            reason = "below_min_trade_notional"
        elif abs(delta_notional) > max_single_order_notional_usd:
            status = "blocked"
            reason = "delta_notional_exceeds_cap"
        instructions.append(
            TradeInstruction(
                coin=coin,
                price=price,
                current_size=current_size,
                target_size=target_size,
                delta_size=delta_size,
                current_notional_usd=current_notional,
                target_notional_usd=target_size * price,
                delta_notional_usd=delta_notional,
                side=side,
                status=status,
                reason=reason,
            )
        )
    return instructions


def summarize_instructions(instructions: list[TradeInstruction]) -> dict:
    trade_count = sum(1 for item in instructions if item.status == "trade")
    blocked = [item.coin for item in instructions if item.status == "blocked"]
    gross_target = float(sum(abs(item.target_notional_usd) for item in instructions))
    gross_delta = float(sum(abs(item.delta_notional_usd) for item in instructions if item.status == "trade"))
    return {
        "coins_considered": len(instructions),
        "trade_count": trade_count,
        "blocked_coins": blocked,
        "gross_target_notional_usd": gross_target,
        "gross_delta_notional_usd": gross_delta,
    }


def _round_toward_zero(value: float, decimals: int) -> float:
    factor = 10**int(decimals)
    if factor <= 0:
        return float(value)
    if value >= 0.0:
        return float(np.floor(value * factor) / factor)
    return float(np.ceil(value * factor) / factor)
