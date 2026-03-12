# Q_Lab_HL

Q_Lab_HL is now a minimal statistical Hyperliquid research harness.

The repo is intentionally narrow:

- data layer: parquet market panels and tradability rules
- strategy layer: feature engineering plus a trainable statistical model
- judge layer: deterministic next-bar backtest and period evaluation
- experiment layer: JSON result files and a flat leaderboard
- paper layer: current deploy candidate snapshot for paper trading

## Core Layout

- `strategy.py`: editable statistical strategy surface
- `strategy_model.py`: small helper for feature generation, dataset building, linear-model fitting, and fit diagnostics
- `run.py`: manual CLI for cache build and single-period evaluation
- `paper_trade.py`: emits the latest live-fit model snapshot, scores, and weights for paper trading
- `autoresearch.py`: deterministic bounded experiment runner
- `q_lab_hl/`: fixed data, execution, portfolio, and evaluation modules
- `autoresearch/`: experiment config, leaderboard, and result outputs
- `paper/`: generated paper-trading signal snapshots
- `data/market_cache_1h/`: bundled hourly market cache

## Current Default Strategy

The checked-in default is the overnight winner:

- OLS
- 3 rank features: `ret_1h`, `ma_gap_24h`, `funding_8h`
- 21-day train window
- 5 long / 5 short
- next open-to-close target
- 24h rebalance cadence

## What Stays Fixed

- next-bar execution only
- funding, fees, and slippage included in PnL
- tradability constraints are part of truth
- long/short gross and net exposure control
- split-based out-of-sample evaluation

The harness under `q_lab_hl/` is the judge. Normal research should not modify it.

## What You Edit

Default mutation scope:

- `strategy.py`
- `strategy_model.py`
- `autoresearch/config*.json`

Default fixed scope:

- `q_lab_hl/data.py`
- `q_lab_hl/backtest.py`
- `q_lab_hl/evaluate.py`
- `q_lab_hl/portfolio.py`
- `run.py`

## Manual Workflow

Install dependencies:

```bash
python3 -m pip install -e .
```

Evaluate the current strategy on the bundled hourly cache:

```bash
python3 run.py --evaluate --data-dir data/market_cache_1h --period inner --show-fit
```

Emit the same result as JSON:

```bash
python3 run.py --evaluate --data-dir data/market_cache_1h --period outer --json --show-fit
```

Refresh the Hyperliquid cache:

```bash
python3 run.py --build-cache --cache-dir data/market_cache_1h --start 2025-01-01 --timeframe 1h --top-n 20 --no-ssl-verify
```

Generate the current paper-trading snapshot:

```bash
python3 paper_trade.py --data-dir data/market_cache_1h --output paper/live_signal_latest.json
```

Refresh data + regenerate the paper snapshot in one step:

```bash
./scripts/run_paper_trade.sh
```

Install an hourly cron job for paper trading refresh:

```bash
./scripts/install_paper_trade_cron.sh
```

Preferred on macOS: install the `launchd` job:

```bash
./scripts/install_paper_trade_launchd.sh
```

## Fit Diagnostics That Matter More Than p-values

The latest fit summary now includes:

- current coefficients and intercept
- train R² and adjusted R²
- RMSE and MAE
- pooled prediction/target correlation
- pooled prediction/target rank correlation
- cross-sectional Pearson IC mean / median / positive-share
- cross-sectional rank IC mean / median / positive-share
- top-vs-bottom quintile realized target spread
- OLS parameter standard errors and t-stats when the model family is OLS

For this workflow, these are still secondary to true out-of-sample performance, but they are much more useful sanity checks than staring at p-values alone.

## Strategy Workflow

The current strategy is statistical, not threshold-rule based.

Its mutation surface is one explicit config object in `strategy.py`:

- feature list
- per-feature transform
- target definition
- model family and regularization
- train window
- position bucket

That same surface can now be supplied from JSON experiment files through:

- `strategy_spec`
- `execution_overrides`

At each rebalance timestamp it:

1. builds feature rows for tradable assets
2. constructs a historical training set using only prior bars
3. fits a small linear model
4. predicts next-bar scores for the current cross-section
5. converts those scores into long/short weights

## Autoresearch Workflow

The bounded experiment loop is:

1. read `RESEARCH_PROMPT.md`, `AUTORESEARCH_RULES.md`, and `autoresearch/config.agent.json`
2. run `python3 autoresearch.py --config autoresearch/config.agent.json`
3. parse stdout JSON
4. inspect `autoresearch/leaderboard.jsonl`
5. mutate `autoresearch/config.agent.json`, `strategy.py`, or `strategy_model.py`
6. rerun and compare against the prior result

Example:

```bash
python3 autoresearch.py --config autoresearch/config.agent.json
```

This is now enough to search the model from JSON alone. Agents do not need to rewrite `strategy.py` just to change:

- features
- transforms
- target kind
- model family
- regularization
- position bucket
- rebalance cadence

Real data is the primary workflow. The command above evaluates the current candidate on the bundled hourly cache in `data/market_cache_1h`, writes a full JSON result under `autoresearch/results/`, and appends a compact summary to `autoresearch/leaderboard.jsonl`.

Optional smoke test with no side effects:

```bash
python3 autoresearch.py \
  --config autoresearch/config.agent.json \
  --synthetic \
  --no-write-result \
  --no-append-leaderboard \
  --experiment-id smoke_stat_model \
  --candidate-id smoke_stat_model
```

## Data Contract

`run.py --data-dir <dir>` and `autoresearch.py` expect matrix parquet files:

- `open.parquet`
- `high.parquet`
- `low.parquet`
- `close.parquet`
- `volume.parquet`
- optional `funding.parquet`
- optional `tradable.parquet`
- optional `metadata.json`

Each parquet file is a timestamp-indexed asset matrix.

## Prompts

Prompt files are still part of the repo, but they are guidance around the workflow, not runtime components:

- `RESEARCH_PROMPT.md`: human-owned research objective and anti-goals
- `AUTORESEARCH_RULES.md`: operational edit/run rules for workers

## Tests

Run the suite with:

```bash
python3 -m unittest discover -s tests
```
