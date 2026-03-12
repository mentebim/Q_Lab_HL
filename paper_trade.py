from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from q_lab_hl.backtest import load_strategy
from q_lab_hl.config import ExecutionConfig
from q_lab_hl.data import DataStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a paper-trading snapshot for the current Q_Lab_HL strategy")
    parser.add_argument("--data-dir", type=str, default="data/market_cache_1h", help="Directory with matrix parquet market panels")
    parser.add_argument("--strategy-path", type=str, default="strategy.py")
    parser.add_argument("--timestamp", type=str, default=None, help="Optional explicit timestamp; defaults to latest data timestamp")
    parser.add_argument("--output", type=str, default="paper/live_signal_latest.json", help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=5, help="Number of longs/shorts to print in the console summary")
    args = parser.parse_args()

    data_store = DataStore.from_parquet_dir(args.data_dir)
    strategy = load_strategy(args.strategy_path)
    execution = getattr(strategy, "EXECUTION", ExecutionConfig())
    ts = pd.Timestamp(args.timestamp) if args.timestamp else pd.Timestamp(data_store.index[-1])

    if not hasattr(strategy, "paper_trade_snapshot"):
        raise SystemExit("Strategy does not expose paper_trade_snapshot().")

    snapshot = strategy.paper_trade_snapshot(data_store, ts)
    payload = {
        "timestamp": pd.Timestamp(ts).isoformat(),
        "data_dir": args.data_dir,
        "strategy_path": args.strategy_path,
        "execution": execution.__dict__,
        **snapshot,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    weights = pd.Series(payload.get("weights", {}), dtype=float).sort_values(ascending=False)
    longs = weights[weights > 0].head(args.top_k)
    shorts = weights[weights < 0].sort_values().head(args.top_k)
    print(f"paper_timestamp: {payload['timestamp']}")
    print(f"output_path: {output_path}")
    print("top_longs:")
    for asset, value in longs.items():
        print(f"  {asset}: {value:.6f}")
    print("top_shorts:")
    for asset, value in shorts.items():
        print(f"  {asset}: {value:.6f}")


if __name__ == "__main__":
    main()
