from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from q_lab_hl.portfolio import validate_exposures


@dataclass(frozen=True)
class MarginCheck:
    account_value: float
    target_initial_margin_usd: float
    allowed_initial_margin_usd: float
    margin_headroom_usd: float
    max_margin_usage_ratio: float
    min_margin_headroom_usd: float


def validate_target_weights(weights: pd.Series, max_gross_exposure: float, target_net_exposure: float) -> None:
    if pd.Series(weights, dtype=float).empty:
        return
    validate_exposures(pd.Series(weights, dtype=float), max_gross_exposure, target_net_exposure)


def kill_switch_active(path: str | Path) -> bool:
    return Path(path).exists()


def validate_margin_plan(
    instructions,
    *,
    account_value: float,
    leverage_by_coin: dict[str, int],
    max_margin_usage_ratio: float,
    min_margin_headroom_usd: float,
) -> MarginCheck:
    final_by_coin = {}
    for instruction in instructions:
        final_by_coin[instruction.coin] = instruction
    target_initial_margin = 0.0
    for coin, instruction in final_by_coin.items():
        leverage = max(int(leverage_by_coin.get(coin, 1)), 1)
        target_initial_margin += abs(float(instruction.target_notional_usd)) / leverage
    allowed_initial_margin = max(float(account_value) * float(max_margin_usage_ratio) - float(min_margin_headroom_usd), 0.0)
    if target_initial_margin > allowed_initial_margin + 1e-9:
        raise ValueError(
            "target initial margin "
            f"{target_initial_margin:.2f} exceeds allowed {allowed_initial_margin:.2f} "
            f"(ratio={max_margin_usage_ratio:.2f}, headroom={min_margin_headroom_usd:.2f})"
        )
    return MarginCheck(
        account_value=float(account_value),
        target_initial_margin_usd=float(target_initial_margin),
        allowed_initial_margin_usd=float(allowed_initial_margin),
        margin_headroom_usd=float(account_value) - float(target_initial_margin),
        max_margin_usage_ratio=float(max_margin_usage_ratio),
        min_margin_headroom_usd=float(min_margin_headroom_usd),
    )


def validate_requested_leverages(
    *,
    leverage_by_coin: dict[str, int],
    metadata: dict[str, dict],
) -> None:
    for coin, leverage in leverage_by_coin.items():
        asset_meta = metadata.get(coin, {}) or {}
        max_leverage = asset_meta.get("maxLeverage")
        if max_leverage is None:
            continue
        if int(leverage) > int(max_leverage):
            raise ValueError(
                f"requested leverage {int(leverage)} for {coin} exceeds metadata maxLeverage {int(max_leverage)}"
            )
