# Q_Lab_HL Execution

This branch is the execution layer only.

Its job is narrow:

1. read pinned champions
2. load trusted market cache
3. mechanically refit the pinned strategy on fresh data
4. calculate target positions
5. trade them in paper or live mode

It does not run research search.
It does not decide promotion.

## Inputs

- `execution/champion.paper.json`
- `execution/champion.live.json`
- `autoresearch/results/*.json` referenced by champion files
- `data/market_cache_1h/`
- Hyperliquid account state and credentials

## Execution Contract

Execution must:

- respect `rebalance_every_bars`
- reconcile account state before trading
- read actual HL account value in live mode
- size positions from the configured margin budget
- treat all wallet capital as strategy capital while still leaving configured margin headroom

The key sizing controls are:

- `target_margin_usage_ratio`
- `max_margin_usage_ratio`
- `min_margin_headroom_usd`
- per-coin leverage settings from the champion runtime config

## Core Commands

Paper:

```bash
python3 -m execution.run_live --champion execution/champion.paper.json
```

Live:

```bash
python3 -m execution.run_live --champion execution/champion.live.json
```

## Tests

Run:

```bash
python3 -m unittest discover -s tests
```
