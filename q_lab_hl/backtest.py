from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from q_lab_hl.config import ExecutionConfig
from q_lab_hl.data import DataStore, DateLimitedStore, default_universe_kwargs, recommended_warmup_bars
from q_lab_hl.portfolio import exposure_diagnostics, gross_exposure, net_exposure, normalize_long_short_weights, validate_exposures


@dataclass
class BacktestResult:
    returns: pd.Series
    active_returns: pd.Series
    equity_curve: pd.Series
    turnover: pd.Series
    gross: pd.Series
    net: pd.Series
    funding_pnl: pd.Series
    cost_pnl: pd.Series
    price_pnl: pd.Series
    weights_history: list[tuple[pd.Timestamp, pd.Series]]
    diagnostics: dict


def load_strategy(path: str | Path = "strategy.py"):
    path = Path(path)
    spec = importlib.util.spec_from_file_location("strategy_hl", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load strategy module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_rebalance_timestamps(index: pd.DatetimeIndex, every_bars: int) -> pd.DatetimeIndex:
    if every_bars <= 1:
        return index
    return pd.DatetimeIndex(index[::every_bars])


def run_backtest(
    strategy_module,
    data_store: DataStore,
    timestamps: pd.DatetimeIndex | None = None,
    execution: ExecutionConfig | None = None,
) -> BacktestResult:
    execution = execution or ExecutionConfig()
    if timestamps is None:
        timestamps = data_store.index
    if hasattr(strategy_module, "reset_state") and callable(strategy_module.reset_state):
        strategy_module.reset_state()
    timestamps = pd.DatetimeIndex(timestamps)
    rebalance_timestamps = set(select_rebalance_timestamps(timestamps, execution.rebalance_every_bars))
    current_weights = pd.Series(dtype=float)
    pending_target = None
    returns = []
    active_returns = []
    equity = 1.0
    equity_curve = []
    turnover_rows = []
    gross_rows = []
    net_rows = []
    funding_rows = []
    cost_rows = []
    price_rows = []
    weights_history = []
    trade_count = 0
    filtered_assets_total = 0
    market_return = data_store.market_return_series(timestamps)
    for i, ts in enumerate(timestamps):
        cost_ret = 0.0
        if pending_target is not None:
            new_weights, filtered_assets = _filter_tradeable_target(
                pd.Series(pending_target, dtype=float),
                data_store,
                ts,
                execution,
            )
            if not new_weights.empty:
                validate_exposures(new_weights, execution.max_gross_exposure, execution.target_net_exposure)
            turnover = _turnover(current_weights, new_weights)
            cost_ret = turnover * (execution.taker_fee_bps + execution.slippage_bps) / 10_000.0
            current_weights = new_weights
            filtered_assets_total += filtered_assets
            trade_count += 1
            turnover_rows.append((ts, turnover))
            weights_history.append((ts, current_weights.copy()))
            pending_target = None
        bar_open = data_store.prices(start=ts, end=ts, field="open")
        bar_close = data_store.prices(start=ts, end=ts, field="close")
        asset_rets = bar_close.iloc[0] / bar_open.iloc[0] - 1.0
        funding_row = data_store.funding(start=ts, end=ts)
        funding = funding_row.iloc[0] if not funding_row.empty else pd.Series(0.0, index=asset_rets.index)
        aligned_weights = current_weights.reindex(asset_rets.index).fillna(0.0)
        price_ret = float((aligned_weights * asset_rets.fillna(0.0)).sum())
        funding_ret = float((-aligned_weights * funding.reindex(asset_rets.index).fillna(0.0)).sum())
        total_ret = price_ret + funding_ret - cost_ret
        equity *= 1.0 + total_ret
        returns.append(total_ret)
        active_returns.append(total_ret - float(market_return.get(ts, 0.0)))
        equity_curve.append(equity)
        price_rows.append((ts, price_ret))
        funding_rows.append((ts, funding_ret))
        cost_rows.append((ts, cost_ret))
        gross_rows.append((ts, gross_exposure(current_weights)))
        net_rows.append((ts, net_exposure(current_weights)))
        if ts in rebalance_timestamps and i + 1 < len(timestamps):
            limited = DateLimitedStore(data_store, ts)
            scores = strategy_module.signals(limited, ts)
            target = strategy_module.construct(scores, limited, ts)
            pending_target = strategy_module.risk(target, limited, ts)
        elif i + 1 == len(timestamps):
            pending_target = None
    turnover = pd.Series(dict(turnover_rows), dtype=float, name="turnover")
    gross = pd.Series(dict(gross_rows), dtype=float, name="gross")
    net = pd.Series(dict(net_rows), dtype=float, name="net")
    diagnostics = {
        "trade_count": trade_count,
        "filtered_assets_total": filtered_assets_total,
        "avg_gross": float(gross.mean()) if not gross.empty else 0.0,
        "avg_net": float(net.mean()) if not net.empty else 0.0,
        "avg_turnover": float(turnover.mean()) if not turnover.empty else 0.0,
        "final_weights": current_weights.copy(),
        "final_weight_diagnostics": exposure_diagnostics(current_weights),
    }
    return BacktestResult(
        returns=pd.Series(returns, index=timestamps, dtype=float, name="returns"),
        active_returns=pd.Series(active_returns, index=timestamps, dtype=float, name="active_returns"),
        equity_curve=pd.Series(equity_curve, index=timestamps, dtype=float, name="equity"),
        turnover=turnover,
        gross=gross,
        net=net,
        funding_pnl=pd.Series(dict(funding_rows), dtype=float, name="funding_pnl"),
        cost_pnl=pd.Series(dict(cost_rows), dtype=float, name="cost_pnl"),
        price_pnl=pd.Series(dict(price_rows), dtype=float, name="price_pnl"),
        weights_history=weights_history,
        diagnostics=diagnostics,
    )


def strategy_warmup_timestamps(data_store: DataStore, execution: ExecutionConfig | None = None) -> pd.DatetimeIndex:
    execution = execution or ExecutionConfig()
    warmup = recommended_warmup_bars(execution)
    if len(data_store.index) <= warmup:
        return data_store.index
    return data_store.index[warmup:]


def default_strategy_universe(data_store: DataStore, ts, execution: ExecutionConfig | None = None) -> list[str]:
    execution = execution or ExecutionConfig()
    return data_store.tradable_universe(ts, **default_universe_kwargs(execution))


def _filter_tradeable_target(
    target: pd.Series,
    data_store: DataStore,
    ts,
    execution: ExecutionConfig,
) -> tuple[pd.Series, int]:
    filtered = target[target.index.map(lambda asset: data_store.can_trade(asset, ts))]
    filtered_assets = int(len(target) - len(filtered))
    if filtered.empty:
        return pd.Series(dtype=float), filtered_assets
    groups = pd.Series({asset: data_store.sector(asset) for asset in filtered.index})
    normalized = normalize_long_short_weights(
        filtered,
        gross_target=min(execution.target_gross_exposure, execution.max_gross_exposure),
        net_target=execution.target_net_exposure,
        max_abs_weight=execution.max_abs_weight,
        groups=groups,
        max_group_gross=execution.max_group_gross,
    )
    return normalized, filtered_assets


def _turnover(old_weights: pd.Series, new_weights: pd.Series) -> float:
    names = sorted(set(old_weights.index) | set(new_weights.index))
    if not names:
        return 0.0
    return 0.5 * float(
        np.abs(
            old_weights.reindex(names).fillna(0.0).to_numpy()
            - new_weights.reindex(names).fillna(0.0).to_numpy()
        ).sum()
    )

