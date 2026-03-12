from __future__ import annotations

import argparse
import json
import pandas as pd

from q_lab_hl.backtest import load_strategy
from q_lab_hl.cache import CacheBuildConfig, build_hyperliquid_cache
from q_lab_hl.config import ExecutionConfig
from q_lab_hl.data import DataStore
from q_lab_hl.evaluate import evaluate, format_metrics
from q_lab_hl.ingest import HyperliquidInfoClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal statistical Hyperliquid research harness")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with matrix parquet market panels")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic market data")
    parser.add_argument("--build-cache", action="store_true", help="Build a real Hyperliquid parquet cache")
    parser.add_argument("--evaluate", action="store_true", help="Run a period evaluation")
    parser.add_argument("--period", type=str, default="inner", choices=["train", "inner", "outer", "test"])
    parser.add_argument("--strategy-path", type=str, default="strategy.py")
    parser.add_argument("--strategy-spec", type=str, default=None, help="Path to a JSON file with strategy spec overrides.")
    parser.add_argument("--execution-overrides", type=str, default=None, help="Path to a JSON file with execution overrides.")
    parser.add_argument("--cache-dir", type=str, default="data/market_cache_1h", help="Directory for built market cache or backtest input")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Cache build start timestamp")
    parser.add_argument("--end", type=str, default=None, help="Cache build end timestamp")
    parser.add_argument("--timeframe", type=str, default="1h", help="Hyperliquid candle interval, e.g. 15m, 1h, 4h")
    parser.add_argument("--coins", type=str, default=None, help="Comma-separated explicit coin list")
    parser.add_argument("--top-n", type=int, default=25, help="Top current-liquidity coins to fetch when --coins is omitted")
    parser.add_argument("--min-current-day-ntl-vlm", type=float, default=None, help="Current day notional volume filter")
    parser.add_argument("--include-delisted", action="store_true", help="Include delisted markets in the cache")
    parser.add_argument("--no-ssl-verify", action="store_true", help="Disable SSL verification for API fetches")
    parser.add_argument("--json", action="store_true", help="Emit JSON metrics instead of formatted text")
    parser.add_argument("--show-fit", action="store_true", help="Include the strategy's latest fit summary when available")
    args = parser.parse_args()

    if not any([args.build_cache, args.evaluate]):
        parser.print_help()
        return
    if not args.synthetic and not args.data_dir and not args.build_cache:
        raise SystemExit("Provide --data-dir for real data or use --synthetic.")

    if args.build_cache:
        coins = [coin.strip() for coin in args.coins.split(",") if coin.strip()] if args.coins else None
        client = HyperliquidInfoClient(verify_ssl=not args.no_ssl_verify)
        summary = build_hyperliquid_cache(
            output_dir=args.cache_dir,
            config=CacheBuildConfig(
                start=args.start,
                end=args.end or pd.Timestamp.now("UTC").tz_localize(None).isoformat(),
                interval=args.timeframe,
                include_delisted=args.include_delisted,
                top_n=args.top_n if coins is None else None,
                min_current_day_ntl_vlm=args.min_current_day_ntl_vlm,
            ),
            client=client,
            coins=coins,
        )
        for key, value in summary.items():
            print(f"{key}: {value}")

    if not args.evaluate:
        return

    data_root = args.data_dir or args.cache_dir
    data_store = DataStore.synthetic(n_assets=16, periods=24 * 25, seed=7) if args.synthetic else DataStore.from_parquet_dir(data_root)
    strategy = load_strategy(args.strategy_path)
    if hasattr(strategy, "apply_runtime_overrides"):
        strategy.apply_runtime_overrides(
            strategy_spec=_load_optional_json(args.strategy_spec),
            execution_overrides=_load_optional_json(args.execution_overrides),
        )
    execution = getattr(strategy, "EXECUTION", ExecutionConfig())
    metrics = evaluate(strategy, data_store, period=args.period, execution=execution)
    if args.show_fit and hasattr(strategy, "last_fit_summary"):
        metrics["model_fit"] = strategy.last_fit_summary()
    if args.json:
        compact = {key: value for key, value in metrics.items() if not isinstance(value, (pd.Series, pd.DataFrame))}
        print(json.dumps(compact, indent=2, sort_keys=True, default=str))
        return
    print(format_metrics(metrics))

def _load_optional_json(path: str | None):
    if path is None:
        return None
    with open(path) as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
