# Program

This repo is a constrained Hyperliquid research harness. The default workflow is:

1. Edit `strategy.py`.
2. Run `python3 run.py --data-dir data/market_cache_1h --grid-search ...` or `--backtest`.
3. Compare out-of-sample metrics, not just in-sample Sharpe.
4. Iterate.

## Hard Rules

- `strategy.py` is the normal research surface.
- `q_lab_hl/` is fixed infrastructure unless we are explicitly changing the harness.
- Signals are computed on bar `t`; execution happens on bar `t+1`.
- Funding, fees, and slippage are part of the result.

## What Matters

- out-of-sample score
- active Sharpe
- turnover
- beta drift
- concentration
- stability across CV folds

## What Does Not Count

- same-bar fills
- alpha from untradeable names
- performance that disappears after funding or cost drag
- parameter choices justified only by one lucky split
