from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.exchange_hl import HyperliquidExecutionClient, VenueConfig
from execution.portfolio_live import build_trade_instructions, summarize_instructions
from execution.risk import kill_switch_active, validate_target_weights
from execution.state import load_state, record_run, save_state, seen_signal_bar
from q_lab_hl.backtest import load_strategy
from q_lab_hl.config import ExecutionConfig
from q_lab_hl.data import DataStore, DateLimitedStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one live or paper execution cycle for the pinned champion strategy.")
    parser.add_argument("--champion", default="execution/champion.json")
    parser.add_argument("--force", action="store_true", help="Allow trading the same signal bar again.")
    parser.add_argument("--dry-run-orders", action="store_true", help="Build the order plan but do not apply it.")
    args = parser.parse_args()

    champion = json.loads(Path(args.champion).read_text())
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
    if seen_signal_bar(state, latest_ts) and not args.force:
        print(json.dumps({"status": "skipped", "reason": "signal_bar_already_processed", "signal_bar": str(latest_ts)}, indent=2))
        return

    strategy = load_strategy(champion.get("strategy_path", "strategy.py"))
    if hasattr(strategy, "apply_runtime_overrides"):
        strategy.apply_runtime_overrides(
            strategy_spec=champion.get("strategy_spec"),
            execution_overrides=champion.get("execution_overrides"),
        )
    execution = getattr(strategy, "EXECUTION", ExecutionConfig())
    latest_store = DateLimitedStore(data_store, latest_ts)
    scores = strategy.signals(latest_store, latest_ts)
    target = strategy.risk(strategy.construct(scores, latest_store, latest_ts), latest_store, latest_ts)
    validate_target_weights(
        pd.Series(target, dtype=float),
        max_gross_exposure=execution.max_gross_exposure,
        target_net_exposure=execution.target_net_exposure,
    )

    client = HyperliquidExecutionClient(venue)
    current_positions = client.current_positions(state)
    account_value = client.account_value()
    prices = pd.Series(client.mid_prices(sorted(set(target.index) | set(current_positions.keys()))), dtype=float)
    if prices.empty:
        prices = data_store.prices(field="close").loc[latest_ts].dropna()
    metadata = data_store.metadata or {}
    size_decimals = {
        coin: int((metadata.get(coin, {}) or {}).get("szDecimals", 4) or 4)
        for coin in prices.index
    }
    instructions = build_trade_instructions(
        pd.Series(target, dtype=float),
        prices,
        current_positions,
        account_value=account_value,
        size_decimals=size_decimals,
        min_trade_notional_usd=venue.min_trade_notional_usd,
        max_single_order_notional_usd=venue.max_single_order_notional_usd,
    )
    if args.dry_run_orders:
        fills = [{"coin": item.coin, "status": item.status, "reason": item.reason} for item in instructions]
        next_state = dict(state)
    else:
        fills, next_state = client.apply_instructions(instructions, state)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal_bar": str(latest_ts),
        "venue": venue.summary(),
        "account_value": account_value,
        "strategy_path": champion.get("strategy_path", "strategy.py"),
        "source": champion.get("source", {}),
        "target_weights": {coin: float(weight) for coin, weight in pd.Series(target, dtype=float).items()},
        "prices": {coin: float(price) for coin, price in prices.items()},
        "instructions": [item.summary() for item in instructions],
        "instruction_summary": summarize_instructions(instructions),
        "fills": fills,
        "dry_run_orders": bool(args.dry_run_orders),
    }
    log_dir = Path(champion["live"].get("log_dir", "execution/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{latest_ts.isoformat().replace(':', '_')}.json"
    log_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    final_state = record_run(next_state, latest_ts, {"signal_bar": str(latest_ts), "log_path": str(log_path)})
    save_state(state_path, final_state)
    print(json.dumps({"status": "ok", "signal_bar": str(latest_ts), "log_path": str(log_path), "trades": report["instruction_summary"]["trade_count"]}, indent=2))


if __name__ == "__main__":
    main()
