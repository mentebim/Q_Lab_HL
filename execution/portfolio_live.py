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
    execution_stage: str
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
        instructions.extend(
            _plan_coin_instructions(
                coin=coin,
                price=price,
                current_size=current_size,
                target_size=target_size,
                size_decimals=int(size_decimals.get(coin, 4)),
                min_trade_notional_usd=min_trade_notional_usd,
                max_single_order_notional_usd=max_single_order_notional_usd,
            )
        )
    return instructions


def derive_target_gross_notional(
    target_weights: pd.Series,
    *,
    account_value: float,
    leverage_by_coin: dict[str, int],
    target_margin_usage_ratio: float,
    max_margin_usage_ratio: float,
    min_margin_headroom_usd: float,
) -> dict[str, float]:
    weights = pd.Series(target_weights, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if weights.empty:
        return {
            "target_gross_notional_usd": 0.0,
            "target_initial_margin_usd": 0.0,
            "effective_margin_usage_ratio": 0.0,
        }
    max_margin_budget = max(
        min(
            float(account_value) * float(max_margin_usage_ratio),
            float(account_value) - float(min_margin_headroom_usd),
        ),
        0.0,
    )
    target_margin_budget = max(
        min(float(account_value) * float(target_margin_usage_ratio), max_margin_budget),
        0.0,
    )
    inverse_leverage_weight = 0.0
    for coin, weight in weights.items():
        leverage = max(int(leverage_by_coin.get(coin, 1)), 1)
        inverse_leverage_weight += abs(float(weight)) / leverage
    if inverse_leverage_weight <= 0.0 or target_margin_budget <= 0.0:
        return {
            "target_gross_notional_usd": 0.0,
            "target_initial_margin_usd": 0.0,
            "effective_margin_usage_ratio": 0.0,
        }
    gross_target = float(target_margin_budget / inverse_leverage_weight)
    return {
        "target_gross_notional_usd": gross_target,
        "target_initial_margin_usd": float(target_margin_budget),
        "effective_margin_usage_ratio": 0.0 if account_value <= 0.0 else float(target_margin_budget / float(account_value)),
    }


def summarize_instructions(instructions: list[TradeInstruction]) -> dict:
    final_by_coin = _last_instruction_by_coin(instructions)
    trade_count = sum(1 for item in instructions if item.status == "trade")
    blocked = [item.coin for item in instructions if item.status == "blocked"]
    gross_target = float(sum(abs(item.target_notional_usd) for item in final_by_coin.values()))
    gross_delta = float(sum(abs(item.delta_notional_usd) for item in instructions if item.status == "trade"))
    return {
        "coins_considered": len(instructions),
        "trade_count": trade_count,
        "blocked_coins": blocked,
        "gross_target_notional_usd": gross_target,
        "gross_delta_notional_usd": gross_delta,
        "stage_counts": {
            stage: sum(1 for item in instructions if item.execution_stage == stage and item.status == "trade")
            for stage in sorted({item.execution_stage for item in instructions})
        },
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
    risk_reducing = [item for item in items if item.status == "trade" and item.execution_stage in {"close", "decrease", "flip_close"}]
    risk_adding = [item for item in items if item.status == "trade" and item.execution_stage not in {"close", "decrease", "flip_close"}]
    deferred = [item for item in items if item.status != "trade"]
    ordered = _pairwise_by_side(risk_reducing) + _pairwise_by_side(risk_adding)
    return ordered + deferred


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


def _plan_coin_instructions(
    *,
    coin: str,
    price: float,
    current_size: float,
    target_size: float,
    size_decimals: int,
    min_trade_notional_usd: float,
    max_single_order_notional_usd: float,
) -> list[TradeInstruction]:
    if current_size * target_size < 0.0:
        return _slice_stage(
            coin=coin,
            price=price,
            start_size=current_size,
            end_size=0.0,
            size_decimals=size_decimals,
            min_trade_notional_usd=min_trade_notional_usd,
            max_single_order_notional_usd=max_single_order_notional_usd,
            execution_stage="flip_close",
        ) + _slice_stage(
            coin=coin,
            price=price,
            start_size=0.0,
            end_size=target_size,
            size_decimals=size_decimals,
            min_trade_notional_usd=min_trade_notional_usd,
            max_single_order_notional_usd=max_single_order_notional_usd,
            execution_stage="flip_open",
        )
    stage = "close" if target_size == 0.0 else "decrease" if current_size != 0.0 and abs(target_size) < abs(current_size) else "open" if current_size == 0.0 else "increase"
    return _slice_stage(
        coin=coin,
        price=price,
        start_size=current_size,
        end_size=target_size,
        size_decimals=size_decimals,
        min_trade_notional_usd=min_trade_notional_usd,
        max_single_order_notional_usd=max_single_order_notional_usd,
        execution_stage=stage,
    )


def _slice_stage(
    *,
    coin: str,
    price: float,
    start_size: float,
    end_size: float,
    size_decimals: int,
    min_trade_notional_usd: float,
    max_single_order_notional_usd: float,
    execution_stage: str,
) -> list[TradeInstruction]:
    delta_total = _round_toward_zero(end_size - start_size, size_decimals)
    total_notional = abs(delta_total * price)
    if abs(delta_total) <= 0.0 or total_notional < min_trade_notional_usd:
        return [
            _instruction(
                coin=coin,
                price=price,
                current_size=start_size,
                target_size=start_size,
                delta_size=0.0,
                status="skip",
                reason="below_min_trade_notional",
                size_decimals=size_decimals,
                execution_stage=execution_stage,
            )
        ]
    if max_single_order_notional_usd <= 0.0:
        return [
            _instruction(
                coin=coin,
                price=price,
                current_size=start_size,
                target_size=start_size,
                delta_size=0.0,
                status="blocked",
                reason="invalid_max_single_order_notional",
                size_decimals=size_decimals,
                execution_stage=execution_stage,
            )
        ]
    if total_notional <= max_single_order_notional_usd + 1e-9:
        return [
            _instruction(
                coin=coin,
                price=price,
                current_size=start_size,
                target_size=end_size,
                delta_size=delta_total,
                status="trade",
                reason="",
                size_decimals=size_decimals,
                execution_stage=execution_stage,
            )
        ]

    max_chunk_size = _round_toward_zero(max_single_order_notional_usd / price, size_decimals)
    if max_chunk_size <= 0.0:
        return [
            _instruction(
                coin=coin,
                price=price,
                current_size=start_size,
                target_size=start_size,
                delta_size=0.0,
                status="blocked",
                reason="cap_too_small_for_size_precision",
                size_decimals=size_decimals,
                execution_stage=execution_stage,
            )
        ]

    sign = 1.0 if delta_total > 0.0 else -1.0
    instructions: list[TradeInstruction] = []
    current = float(start_size)
    target = float(end_size)
    while abs((target - current) * price) > max_single_order_notional_usd + 1e-9:
        chunk_delta = sign * max_chunk_size
        next_size = _round_toward_zero(current + chunk_delta, size_decimals)
        if next_size == current:
            return instructions + [
                _instruction(
                    coin=coin,
                    price=price,
                    current_size=current,
                    target_size=current,
                    delta_size=0.0,
                    status="blocked",
                    reason="size_precision_prevents_slicing",
                    size_decimals=size_decimals,
                    execution_stage=execution_stage,
                )
            ]
        instructions.append(
            _instruction(
                coin=coin,
                price=price,
                current_size=current,
                target_size=next_size,
                delta_size=next_size - current,
                status="trade",
                reason="",
                size_decimals=size_decimals,
                execution_stage=execution_stage,
            )
        )
        current = next_size
    final_delta = _round_toward_zero(target - current, size_decimals)
    final_notional = abs(final_delta * price)
    if abs(final_delta) > 0.0 and final_notional >= min_trade_notional_usd:
        instructions.append(
            _instruction(
                coin=coin,
                price=price,
                current_size=current,
                target_size=target,
                delta_size=final_delta,
                status="trade",
                reason="",
                size_decimals=size_decimals,
                execution_stage=execution_stage,
            )
        )
    elif abs(final_delta) > 0.0:
        instructions.append(
            _instruction(
                coin=coin,
                price=price,
                current_size=current,
                target_size=current,
                delta_size=0.0,
                status="skip",
                reason="residual_below_min_trade_notional",
                size_decimals=size_decimals,
                execution_stage=execution_stage,
            )
        )
    return instructions


def _instruction(
    *,
    coin: str,
    price: float,
    current_size: float,
    target_size: float,
    delta_size: float,
    status: str,
    reason: str,
    size_decimals: int,
    execution_stage: str,
) -> TradeInstruction:
    current_notional = float(current_size * price)
    target_notional = float(target_size * price)
    delta_notional = float(delta_size * price)
    side = "buy" if delta_size > 0 else "sell" if delta_size < 0 else "flat"
    return TradeInstruction(
        coin=coin,
        price=float(price),
        current_size=float(current_size),
        target_size=float(target_size),
        delta_size=float(delta_size),
        current_notional_usd=current_notional,
        target_notional_usd=target_notional,
        delta_notional_usd=delta_notional,
        side=side,
        status=status,
        execution_stage=execution_stage,
        reason=reason,
        size_decimals=int(size_decimals),
    )


def _last_instruction_by_coin(instructions: list[TradeInstruction]) -> dict[str, TradeInstruction]:
    last: dict[str, TradeInstruction] = {}
    for instruction in instructions:
        last[instruction.coin] = instruction
    return last
