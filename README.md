# Q_Lab

Q_Lab is a hardened autonomous quant research harness. An external LLM agent iterates on one mutable file, [strategy.py](/Users/isaacentebi/Desktop/Q_Lab/strategy.py), while the fixed scaffold in [prepare.py](/Users/isaacentebi/Desktop/Q_Lab/prepare.py) handles point-in-time data loading, backtesting, evaluation, and auditing.

This repo is for research, not live trading. It searches for candidate long-only equity strategies, scores them on an inner validation loop, and then optionally audits stronger candidates on a hidden outer slice.

## Repo Shape

- `prepare.py`: fixed research scaffold, PIT data pipeline, backtest engine, evaluation, audit CLI
- `strategy.py`: the single file the autonomous agent edits during normal research
- `program.md`: operating instructions for the agent loop
- `results.tsv`: inner-loop experiment log
- `tests/`: regression coverage for PIT loading, execution timing, evaluation math, and harness constraints

## Current Design

- US-first equity research harness
- Long-only, explicit-cash only
- T+1 execution with `next_close`
- Split signal prices vs total-return prices
- Filing-aware fundamentals and ALFRED/FRED vintage-aware revised macro
- Inner visible search plus outer auditor-only validation
- Strict branch-per-run workflow

## What The Agent Does

The agent is external to the harness. The intended Karpathy-style loop is:

1. Read [program.md](/Users/isaacentebi/Desktop/Q_Lab/program.md)
2. Edit only [strategy.py](/Users/isaacentebi/Desktop/Q_Lab/strategy.py)
3. Run the inner evaluation
4. Read the printed metrics
5. Log the run to [results.tsv](/Users/isaacentebi/Desktop/Q_Lab/results.tsv)
6. Repeat

The harness itself does not spawn the search loop.

## Branch Workflow

Every autonomous run should start from clean `main` and use a fresh branch:

```bash
git checkout main
git pull
git checkout -b qlab/<tag>
```

Rules:

- one autonomous run = one branch
- the agent does not create or switch branches inside the loop
- the agent does not touch git state destructively
- `main` is updated only by a human after review

## Setup

Requirements:

- Python 3.10+
- dependencies from [pyproject.toml](/Users/isaacentebi/Desktop/Q_Lab/pyproject.toml)
- `FMP_API_KEY` in `.env`
- `FRED_API_KEY` in `.env` for vintage-aware revised macro

Install dependencies:

```bash
uv sync
```

Rebuild or refresh the PIT cache:

```bash
uv run prepare.py --download --rebuild
```

If the schema is already current and the cache exists, a rebuild is not required.

## Running The Inner Loop

Manual baseline:

```bash
python3 prepare.py --backtest --n-trials 0
```

The inner-loop objective is `score_inner`, which combines:

- median active Sharpe across inner slices
- turnover penalty
- concentration penalty
- slice-instability penalty

Higher is better.

Recommended `results.tsv` schema:

```tsv
commit	score_inner	active_sharpe_daily	turnover	status	description
```

## Running The Auditor

The auditor is Python, not an LLM. It evaluates the current [strategy.py](/Users/isaacentebi/Desktop/Q_Lab/strategy.py) on the hidden outer slice.

```bash
QLAB_AUDITOR_MODE=1 python3 prepare.py --audit --n-trials N --candidate-id <id>
```

The outer audit reports:

- `DSR_eff`
- `DSR_raw`
- active-return bootstrap CI
- candidate-level `spa_pvalue`
- `spa_family_pvalue`
- `audit_state`

Outer audit should be run only on strong inner candidates. Audit outputs are not part of the ordinary search loop.

## Data Availability Notes

The harness relies on FMP stable endpoints plus FRED/ALFRED.

Important limitations:

- legacy FMP `api/v3` is not assumed available
- historical S&P membership does not guarantee current vendor support for an old symbol
- some retired or recycled symbols have only partial coverage
- symbol-change is used as an identity hint only, not an automatic history splice

The practical backtestability gate is `2020+` price support. Symbols without usable price history are marked unsupported for this harness and do not flow into the later slow-data pipeline.

## Current Research Inventory

The current hardened cache supports:

- `1188` backtest-supported tickers
- `1553` trading days
- date range `2020-01-02` to `2026-03-09`
- `17` raw PIT fundamental fields
- `9` macro series total:
  - `5` market-observed daily macro series
  - `4` ALFRED vintage-aware revised macro series

Inside the usable ticker universe, current coverage is:

- prices: `1188 / 1188`
- market cap history: `1039 / 1188`
- statements: `1131 / 1188`
- earnings: `1188 / 1188`
- SEC filings: `1123 / 1188`

These counts can change after cache rebuilds or vendor-side updates.

## Safety

This repo does not place trades.

It is a research harness for:

- searching candidate strategies
- auditing candidate strategies
- producing a strategy definition in [strategy.py](/Users/isaacentebi/Desktop/Q_Lab/strategy.py)
- producing an experiment log in [results.tsv](/Users/isaacentebi/Desktop/Q_Lab/results.tsv)

Use paper trading and separate implementation review before treating any candidate as production-worthy.
