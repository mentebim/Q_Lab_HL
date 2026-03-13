from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.exchange_hl import HyperliquidExecutionClient, VenueConfig
from execution.portfolio_live import (
    build_trade_instructions,
    derive_target_gross_notional,
    prioritize_instructions,
    summarize_instructions,
)
from execution.risk import kill_switch_active, validate_margin_plan, validate_requested_leverages, validate_target_weights
from execution.select_champion import validate_champion_payload
from execution.state import last_fill_time_ms, load_state, record_reconciliation, record_run, save_state, seen_signal_bar
from q_lab_hl.backtest import load_strategy, select_rebalance_timestamps, strategy_warmup_timestamps
from q_lab_hl.config import ExecutionConfig
from q_lab_hl.data import DataStore, DateLimitedStore


def is_rebalance_bar(data_store: DataStore, execution: ExecutionConfig, ts) -> bool:
    usable_index = strategy_warmup_timestamps(data_store, execution)
    if len(usable_index) == 0:
        return False
    return pd.Timestamp(ts) in set(select_rebalance_timestamps(usable_index, execution.rebalance_every_bars))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one live or paper execution cycle for the pinned champion strategy.")
    parser.add_argument("--champion", default="execution/champion.paper.json")
    parser.add_argument("--force", action="store_true", help="Allow trading the same signal bar again.")
    parser.add_argument("--dry-run-orders", action="store_true", help="Build the order plan but do not apply it.")
    args = parser.parse_args()

    champion = json.loads(Path(args.champion).read_text())
    validate_champion_payload(champion)
    venue = VenueConfig(**champion.get("live", {}))
    if kill_switch_active(champion["live"].get("kill_switch_path", "execution/STOP")):
        raise SystemExit("Kill switch is active; refusing to run live execution.")
    data_dir = champion.get("data_dir", "data/market_cache_1h")
    data_store = DataStore.from_parquet_dir(data_dir)
    latest_ts = pd.Timestamp(data_store.index.max())
    lag_hours = (pd.Timestamp.now("UTC").tz_localize(None) - latest_ts) / pd.Timedelta(hours=1)
    if lag_hours > float(champion["live"].get("max_data_lag_hours", 3.0)):
        raise SystemExit(f"Data is stale by {lag_hours:.2f} hours; refresh cache before trading.")
    state_path = champion["live"].get("state_path", "execution/state.json")
    state = load_state(state_path)

    strategy = load_strategy(champion.get("strategy_path", "strategy.py"))
    if hasattr(strategy, "apply_runtime_overrides"):
        strategy.apply_runtime_overrides(
            strategy_spec=champion.get("strategy_spec"),
            execution_overrides=champion.get("execution_overrides"),
        )
    execution = getattr(strategy, "EXECUTION", ExecutionConfig())
    rebalance_due = is_rebalance_bar(data_store, execution, latest_ts)

    client = HyperliquidExecutionClient(venue)
    reconciliation_pre = client.reconciliation_snapshot(state, fills_since_ms=last_fill_time_ms(state))
    if reconciliation_pre.open_orders and not args.force:
        next_state = record_reconciliation(state, reconciliation_pre.summary())
        save_state(state_path, next_state)
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "open_orders_present",
                    "signal_bar": str(latest_ts),
                    "open_orders": reconciliation_pre.open_orders,
                },
                indent=2,
            )
        )
        return
    if not rebalance_due and not args.force:
        next_state = record_reconciliation(state, reconciliation_pre.summary())
        save_state(state_path, next_state)
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "not_rebalance_bar",
                    "signal_bar": str(latest_ts),
                    "rebalance_every_bars": execution.rebalance_every_bars,
                },
                indent=2,
            )
        )
        return
    if seen_signal_bar(state, latest_ts) and not args.force:
        next_state = record_reconciliation(state, reconciliation_pre.summary())
        save_state(state_path, next_state)
        print(json.dumps({"status": "skipped", "reason": "signal_bar_already_processed", "signal_bar": str(latest_ts)}, indent=2))
        return
    latest_store = DateLimitedStore(data_store, latest_ts)
    scores = strategy.signals(latest_store, latest_ts)
    target = strategy.risk(strategy.construct(scores, latest_store, latest_ts), latest_store, latest_ts)
    validate_target_weights(
        pd.Series(target, dtype=float),
        max_gross_exposure=execution.max_gross_exposure,
        target_net_exposure=execution.target_net_exposure,
    )
    current_positions = reconciliation_pre.positions
    account_value = reconciliation_pre.account_value
    prices = pd.Series(client.mid_prices(sorted(set(target.index) | set(current_positions.keys()))), dtype=float)
    if prices.empty:
        prices = data_store.prices(field="close").loc[latest_ts].dropna()
    metadata = data_store.metadata or {}
    size_decimals = {
        coin: int((metadata.get(coin, {}) or {}).get("szDecimals", 4) or 4)
        for coin in prices.index
    }
    leverage_by_coin = {
        coin: int(venue.leverage_overrides.get(coin, venue.default_leverage))
        for coin in sorted(set(target.index) | set(current_positions.keys()))
    }
    validate_requested_leverages(leverage_by_coin=leverage_by_coin, metadata=metadata)
    sizing = None
    target_gross_notional_usd = venue.target_gross_notional_usd
    if venue.target_margin_usage_ratio is not None:
        sizing = derive_target_gross_notional(
            pd.Series(target, dtype=float),
            account_value=account_value,
            leverage_by_coin=leverage_by_coin,
            target_margin_usage_ratio=venue.target_margin_usage_ratio,
            max_margin_usage_ratio=venue.max_margin_usage_ratio,
            min_margin_headroom_usd=venue.min_margin_headroom_usd,
        )
        target_gross_notional_usd = sizing["target_gross_notional_usd"]
    instructions = build_trade_instructions(
        pd.Series(target, dtype=float),
        prices,
        current_positions,
        account_value=account_value,
        size_decimals=size_decimals,
        min_trade_notional_usd=venue.min_trade_notional_usd,
        max_single_order_notional_usd=venue.max_single_order_notional_usd,
        target_gross_notional_usd=target_gross_notional_usd,
    )
    margin_check = validate_margin_plan(
        instructions,
        account_value=account_value,
        leverage_by_coin=leverage_by_coin,
        max_margin_usage_ratio=venue.max_margin_usage_ratio,
        min_margin_headroom_usd=venue.min_margin_headroom_usd,
    )
    ordered_instructions = prioritize_instructions(instructions)
    if args.dry_run_orders:
        fills = [{"coin": item.coin, "status": item.status, "reason": item.reason} for item in ordered_instructions]
        next_state = dict(state)
    else:
        fills, next_state = client.apply_instructions(ordered_instructions, state)
    reconciliation_post = client.reconciliation_snapshot(next_state, fills_since_ms=last_fill_time_ms(state))

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal_bar": str(latest_ts),
        "venue": venue.summary(),
        "account_value": account_value,
        "rebalance_due": rebalance_due,
        "rebalance_every_bars": execution.rebalance_every_bars,
        "strategy_path": champion.get("strategy_path", "strategy.py"),
        "promotion": champion.get("promotion", {}),
        "source": champion.get("source", {}),
        "target_weights": {coin: float(weight) for coin, weight in pd.Series(target, dtype=float).items()},
        "prices": {coin: float(price) for coin, price in prices.items()},
        "instructions": [item.summary() for item in ordered_instructions],
        "instruction_summary": summarize_instructions(instructions),
        "sizing": sizing,
        "margin_check": margin_check.__dict__,
        "reconciliation_pre": reconciliation_pre.summary(),
        "reconciliation_post": reconciliation_post.summary(),
        "fills": fills,
        "dry_run_orders": bool(args.dry_run_orders),
    }
    log_dir = Path(champion["live"].get("log_dir", "execution/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{latest_ts.isoformat().replace(':', '_')}.json"
    log_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    if not args.dry_run_orders:
        final_state = record_reconciliation(next_state, reconciliation_post.summary())
        final_state = record_run(final_state, latest_ts, {"signal_bar": str(latest_ts), "log_path": str(log_path)})
        save_state(state_path, final_state)
    print(json.dumps({"status": "ok", "signal_bar": str(latest_ts), "log_path": str(log_path), "trades": report["instruction_summary"]["trade_count"]}, indent=2))


if __name__ == "__main__":
    main()
