from __future__ import annotations

import argparse
import json
import pandas as pd

from q_lab_hl.backtest import load_strategy
from q_lab_hl.config import ExecutionConfig
from q_lab_hl.data import DataStore
from q_lab_hl.evaluate import evaluate, format_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal statistical Hyperliquid research harness")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with matrix parquet market panels")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic market data")
    parser.add_argument("--evaluate", action="store_true", help="Run a period evaluation")
    parser.add_argument("--period", type=str, default="inner", choices=["train", "inner", "outer", "test"])
    parser.add_argument("--strategy-path", type=str, default="strategy.py")
    parser.add_argument("--strategy-spec", type=str, default=None, help="Path to a JSON file with strategy spec overrides.")
    parser.add_argument("--execution-overrides", type=str, default=None, help="Path to a JSON file with execution overrides.")
    parser.add_argument("--json", action="store_true", help="Emit JSON metrics instead of formatted text")
    parser.add_argument("--show-fit", action="store_true", help="Include the strategy's latest fit summary when available")
    args = parser.parse_args()

    if not args.evaluate:
        parser.print_help()
        return
    if not args.synthetic and not args.data_dir:
        raise SystemExit("Provide --data-dir for real data or use --synthetic.")

    data_store = DataStore.synthetic(n_assets=16, periods=24 * 25, seed=7) if args.synthetic else DataStore.from_parquet_dir(args.data_dir)
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
