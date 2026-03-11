# Q_Lab_HL

This repo is now a single-purpose Hyperliquid research harness for long/short statistical arbitrage.

The working layout is:

- `strategy.py`: editable research surface
- `run.py`: CLI for cache build, backtest, audit, and CV search
- `q_lab_hl/`: fixed harness modules
- `data/market_cache_1h/`: real 1-hour Hyperliquid cache
- `tests/`: regression tests
- `artifacts/`, `registries/`: generated outputs

## What Stays Fixed

- next-bar execution only
- long/short gross and net exposure control
- fees, slippage, and funding in PnL
- time-series cross-validation with gap, purge, and embargo support

## What You Edit

Normal research should change `strategy.py` only.

## Quick Start

Install dependencies:

```bash
python3 -m pip install -e .
```

Run a synthetic backtest:

```bash
python3 run.py --synthetic --backtest --period inner
```

Run a real-data CV search on the bundled cache:

```bash
python3 run.py --data-dir data/market_cache_1h --grid-search --cv-mode expanding --folds 2 --cv-train-bars 360 --cv-validation-bars 240 --cv-gap-bars 24 --cv-purge-bars 24 --cv-embargo-bars 24 --top-k 5
```

Rebuild the Hyperliquid cache:

```bash
python3 run.py --build-cache --cache-dir data/market_cache_1h --start 2020-01-01 --timeframe 1h --top-n 20 --no-ssl-verify
```

Run the tests:

```bash
python3 -m unittest discover -s tests
```

## Data Contract

`run.py --data-dir <dir>` expects matrix parquet files:

- `open.parquet`
- `high.parquet`
- `low.parquet`
- `close.parquet`
- `volume.parquet`
- optional `funding.parquet`
- optional `tradable.parquet`
- optional `metadata.json`

Each parquet file is a timestamp-indexed asset matrix.
