# Q_Lab: Autonomous Quantitative Research

This is an experiment to have the LLM do its own quantitative research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar8`). The branch `qlab/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b qlab/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — fixed constants, data pipeline, DataStore, backtest engine, evaluation. Do not modify.
   - `strategy.py` — the file you modify. Signals, portfolio construction, risk management.
4. **Download data**: Check if `~/.cache/qlab/` contains price and fundamental data (`ls ~/.cache/qlab/*.parquet | wc -l` should be >100). If not, run `uv run prepare.py --download` yourself. This fetches ~1500 tickers of prices, fundamentals, and macro data from the FMP API. It takes 10-20 minutes but is fully autonomous — just run it and wait.
5. **Initialize results.tsv**: Create `results.tsv` with header row and baseline entry. Run the baseline first to get initial metrics.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs a full backtest. The evaluation runs on the **validation period** (2024-03-01 to 2025-03-01). You launch it simply as: `uv run prepare.py --backtest --n-trials N > run.log 2>&1` (where N is the experiment number).

**What you CAN do:**
- Modify `strategy.py` — this is the only file you edit. Everything is fair game: signal functions, alpha factors, lookback windows, universe filtering, portfolio construction (number of holdings, weighting scheme), risk management (position caps, sector limits, vol targeting).

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, backtest engine, and constants.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest Deflated Sharpe Ratio (DSR).** The DSR automatically penalizes for the number of experiments you've run, so each trial becomes harder to pass. You want strategies with genuinely high risk-adjusted returns, not overfitted noise.

**Keep threshold**: DSR must improve AND bootstrap Sharpe CI lower bound must be > 0. If the lower bound of the 95% CI includes zero, the improvement is not statistically significant — discard it.

**Simplicity criterion**: All else being equal, simpler is better. A small DSR improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

## Data available to your strategy

The DataStore gives you:

- **prices(tickers, start, end)** — daily adjusted close prices
- **returns(tickers, period)** — daily or N-day returns
- **volume(tickers)** — daily volume
- **fundamental(field)** — POINT-IN-TIME quarterly data (date x ticker DataFrame). Each value is what was actually knowable on that date (fiscal quarter end + 45-day reporting lag, forward-filled). Fields: pe, pb, ps, ev_ebitda, fcf_yield, earnings_yield, roe, roa, roic, debt_to_equity, current_ratio, gross_margin, net_margin, revenue_growth, piotroski, altman_z
- **macro(field)** — date-indexed series. Fields: fed_funds, t10y, t2y, t3m, t10y_2y_spread, cpi, unemployment, consumer_sentiment, vix
- **universe(date)** — tickers with sufficient data as of date
- **sector(ticker)**, **country(ticker)**, **metadata_for(ticker)**
- **correlation(tickers, window)**

Fundamentals are historical quarterly, not static snapshots. The backtest sees different PE ratios in 2022 vs 2024.

## Output format

Once the script finishes it prints a summary like this:

```
---
dsr:            0.650000
sharpe:         1.234567
annual_return:  0.150000
max_drawdown:   -0.120000
sortino:        1.800000
calmar:         1.250000
turnover:       0.350000
n_positions:    50
sharpe_ci_95:   (0.45, 2.01)
complexity_loc: 42
```

You can extract the key metric from the log file:

```
grep "^dsr:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	dsr	sharpe	max_dd	turnover	status	description
```

1. git commit hash (short, 7 chars)
2. DSR achieved — use 0.000000 for crashes
3. Sharpe ratio — use 0.000000 for crashes
4. max drawdown — use 0.000000 for crashes
5. avg turnover per rebalance — use 0.000000 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `strategy.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run prepare.py --backtest --n-trials N > run.log 2>&1` (N = experiment number)
5. Read out the results: `grep "^dsr:\|^sharpe:\|^max_drawdown:\|^turnover:" run.log`
6. Also check: `grep "^sharpe_ci_95:" run.log` — if lower bound ≤ 0, discard even if DSR improved.
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace.
8. Record the results in the tsv
9. If DSR improved AND Sharpe CI lower > 0, keep the commit
10. If DSR is equal or worse (or CI includes zero), git reset back to where you started

**Timeout**: Each experiment should take <2 minutes. If a run exceeds 5 minutes, kill it.

**Crashes**: Fix typos/imports and re-run, or log "crash" and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. You are autonomous.

## Experimentation ideas

You control EVERYTHING in strategy.py — signals, universe, construction, risk. The download gives you ~1500 tickers across 40+ countries with prices, quarterly fundamentals, and macro data.

**Universe filtering:**
- S&P 500 only vs full international universe
- Min market cap / volume filters
- Country focus (US-only, developed markets, all)
- Sector exclusions
- Use `data.country(ticker)`, `data.sector(ticker)`, `data.metadata_for(ticker)`

**Signal ideas:**
- Different momentum lookbacks (3m, 6m, 12m, combined)
- Quality: Piotroski F-score, Altman Z-score, ROE, ROA stability
- Value: earnings yield, FCF yield, P/B, EV/EBITDA
- Low volatility: favor stocks with lower realized vol
- Mean reversion (short-term reversal)

**Portfolio construction:**
- Signal-weighted instead of equal-weight
- Risk-parity (inverse volatility)
- Sector-neutral (rank within sectors)
- Vary NUM_HOLDINGS: 30, 50, 75, 100
- Long/short

**Risk management:**
- Volatility targeting
- Max sector exposure
- Drawdown stops
- Position sizing by conviction

**Macro conditioning:**
- Yield curve slope → rotate factors
- VIX regime → defensive/quality tilt
- Rate environment → value vs growth

**Multi-factor composites:**
- Value + Momentum + Quality + Low-Vol
- Adaptive factor weights
- International diversification via companies.json nodes
