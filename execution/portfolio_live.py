from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import zip_longest

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
    size_decimals: int = 4

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
    target_gross_notional_usd: float | None = None,
) -> list[TradeInstruction]:
    weights = pd.Series(target_weights, dtype=float)
    mids = pd.Series(prices, dtype=float)
    instructions: list[TradeInstruction] = []
    gross_target_base = float(target_gross_notional_usd) if target_gross_notional_usd is not None else float(account_value)
    coins = sorted(set(weights.index) | set(current_positions) | set(mids.index))
    for coin in coins:
        price = float(mids.get(coin, np.nan))
        if not np.isfinite(price) or price <= 0.0:
            continue
        current_size = float(current_positions.get(coin, 0.0))
        target_notional = float(weights.get(coin, 0.0) * gross_target_base)
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
                size_decimals=int(size_decimals.get(coin, 4)),
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


def prioritize_instructions(instructions: list[TradeInstruction]) -> list[TradeInstruction]:
    items = list(instructions)
    risk_reducing = [item for item in items if item.status == "trade" and _is_risk_reducing(item)]
    risk_adding = [item for item in items if item.status == "trade" and not _is_risk_reducing(item)]
    deferred = [item for item in items if item.status != "trade"]
    ordered = _pairwise_by_side(risk_reducing) + _pairwise_by_side(risk_adding)
    return ordered + deferred


def _is_risk_reducing(item: TradeInstruction) -> bool:
    if item.current_size == 0.0:
        return False
    if item.target_size == 0.0:
        return True
    if item.current_size * item.target_size < 0.0:
        return True
    return abs(item.target_size) < abs(item.current_size)


def _pairwise_by_side(items: list[TradeInstruction]) -> list[TradeInstruction]:
    longs = sorted([i for i in items if i.delta_size > 0], key=lambda x: abs(x.delta_notional_usd), reverse=True)
    shorts = sorted([i for i in items if i.delta_size < 0], key=lambda x: abs(x.delta_notional_usd), reverse=True)
    out: list[TradeInstruction] = []
    while longs or shorts:
        next_long = longs.pop(0) if longs else None
        next_short = shorts.pop(0) if shorts else None
        choices = [x for x in (next_long, next_short) if x is not None]
        if len(choices) == 2 and abs(choices[1].delta_notional_usd) > abs(choices[0].delta_notional_usd):
            out.append(choices[1]); out.append(choices[0])
        else:
            if next_long is not None: out.append(next_long)
            if next_short is not None: out.append(next_short)
    return out
