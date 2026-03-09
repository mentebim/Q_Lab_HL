# Q_Lab: Hardened Autonomous Quant Research

This repo is an autonomous research harness, not an open-ended codebase. The fixed scaffold lives in `prepare.py`; normal experimentation edits `strategy.py` only.

## Hard Rule

- Normal research may edit `strategy.py` only.
- `prepare.py` is fixed unless the human explicitly authorizes an infrastructure hardening pass.
- This repo has already completed a one-time hardening pass. Do not re-open `prepare.py` during ordinary research loops.

## Setup

1. Start every autonomous run from clean `main`, then create a fresh run branch like `qlab/<tag>`.

```bash
git checkout main
git pull
git checkout -b qlab/<tag>
```

2. Read `prepare.py`, `strategy.py`, and this file before changing anything.
3. Check the cache under `~/.cache/qlab/`. The cache is schema-versioned. If the schema is missing or mismatched, rebuild it with `uv run prepare.py --download --rebuild`.
4. Run the baseline inner evaluation before making changes:

```bash
uv run prepare.py --backtest --n-trials N
```

5. Record the result in `results.tsv`.

## What The Loop Optimizes

Do not optimize visible DSR directly.

- `--backtest` runs the **inner visible search**.
- `--audit` is auditor-only. It is not callable by the search agent and is intentionally hidden from the normal CLI help.
- `--test` runs the final untouched holdout.

The agent-visible score is the inner score printed by `--backtest`. It is based on:

- median active Sharpe across contiguous inner slices
- turnover penalty
- concentration penalty
- slice-instability penalty

This is the search objective. DSR is an audit statistic, not the optimization target.

## Search States

Daily search loop:

1. Edit `strategy.py`.
2. Run `uv run prepare.py --backtest --n-trials N`.
3. Read `score_inner`, `sharpe_daily`, turnover, concentration, and instability metrics.
4. Mark the result as:
   - `inner_keep` if it improves the local search tree
   - `inner_discard` if it does not
   - `promote_outer` if it is strong enough to queue for auditor review

The search agent updates its baseline using inner-search rules only. Outer audit never updates the agent-visible baseline.

Promotion to outer audit must be strict. Use `promote_outer` only if all are true:

- `score_inner > 0`
- `slice_median_active_sharpe > 0.05`
- at least 3 of 4 inner slices have positive active Sharpe
- `turnover_penalty = 0`
- `instability_penalty <= 0.10`
- `active_sharpe_daily > 0`
- the candidate improves materially over the current best, not by noise-level drift

Default to `inner_discard` if any are true:

- `score_inner` is below the current best
- `slice_median_active_sharpe <= 0`
- `turnover_penalty > 0`
- `instability_penalty` jumps materially
- raw portfolio metrics improve but active benchmark-relative metrics worsen

## Auditor-Only Audit

Promotion to outer audit is a separate workflow run by a human or separate auditor process:

```bash
QLAB_AUDITOR_MODE=1 uv run prepare.py --audit --n-trials N --candidate-id <id>
```

Outer audit writes to the hidden audited registry, not to the search agent’s working baseline.

Audit states:

- `audit_pass` if conservative `DSR_raw` is above threshold, the active-return bootstrap CI excludes zero, and candidate-level `spa_pvalue < 0.05`
- `audit_fail` otherwise
- `human_accept` / `human_reject` happen outside the search loop and determine whether a promoted candidate enters the research canon

Also monitor:

- `DSR_eff`
- `spa_family_pvalue`
- `N_eff`
- `N_raw`
- `outer_promotions_total`
- `outer_family_size_current`
- `pbo` when available

Audit outputs are stored outside the agent-visible run log and must not be added to the search agent’s prompt context.

## Branch Policy

- Do not create new branches inside the research loop.
- Do not switch branches inside the research loop.
- Do not run destructive git commands.
- One autonomous run = one branch.
- `main` is the clean blessed baseline.
- Only a human updates `main`, and only after reviewing audited results.
- If a run is bad, discard the branch. If a run is promising, review it and merge the winner back to `main`.

## Execution Reality

Assume the engine is conservative:

- signals are formed using information through close on day `D`
- trades happen on `D+1`
- current execution mode is `next_close`
- same-day signal/trade assumptions are invalid

Do not claim alpha that disappears when timing is lagged.

## DataStore API

Prefer these safe helpers over raw panels:

- `prices_signal(tickers, start, end)`
- `prices_total_return(tickers, start, end)`
- `open_prices(tickers, start, end)`
- `latest_fundamental(field, date)`
- `latest_macro(field, date)`
- `market_cap(date)`
- `dollar_volume(window, date)`
- `tradable_universe(date, min_history_days, min_price, min_dollar_volume, countries, exchanges, sp500_only)`
- `factor_rank(series)`
- `winsorize_cross_section(series, lower_pct, upper_pct)`
- `neutralize_cross_section(series, by=[...])`
- `can_trade(ticker, date)`

Raw accessors like `fundamental(field)` and `macro(field)` are for advanced use only. They are easier to misuse.

## Vendor Limits

- The current harness relies on FMP stable endpoints plus FRED/ALFRED.
- Legacy FMP `api/v3` endpoints are not assumed to be available. If the subscription returns legacy `403` errors, do not build research logic around them.
- Historical S&P membership does not imply vendor support for the old ticker on current stable endpoints.
- Some retired or recycled symbols have partial vendor coverage only:
  - statements or SEC may exist while price history does not
  - symbol-based SEC lookup can be ambiguous for recycled tickers
- The harness treats `2020+` price support as the minimum backtestability gate. Symbols without stable price support are marked as vendor-unsupported for this research harness and do not flow into later slow-data downloads.
- Coverage state is persisted in the cache metadata / coverage audit so unsupported names are explicit rather than silent.
- `symbol-change` is used only as a conservative identity hint. A successor symbol may be recorded as a `resolved_symbol_candidate`, but the harness does not automatically splice successor-history into the old symbol’s backtest path unless that mapping is explicitly authorized and verified.

## Baseline Research Constraints

- Start US-first unless the human explicitly expands scope.
- Treat uncertain slow-data timing as unusable, not “probably fine”.
- Prefer raw-accounting-aware factors, price-based factors, and clean cross-sectional construction over exotic data.
- Keep the strategy simple enough that the audit metrics are interpretable.

## Practical Research Ideas

- momentum variants using `prices_signal`
- value/profitability using `latest_fundamental`
- volatility, drawdown, and liquidity filters
- sector and size neutralization
- explicit cash buffering in risk-off regimes
- benchmark-relative, lower-turnover constructions

## Current Learnings

Carry these forward into future runs unless a later audited result disproves them:

- Buffered membership and moderate weight smoothing are useful construction primitives for this harness.
- Once `turnover_penalty` is already zero, prefer searching for stronger active edge rather than polishing turnover further.
- Smooth portfolios with negative `active_sharpe_daily` are not good candidates; they are benchmark-hugging, not alpha.
- Sector diversification and broader long-only books have been useful for reducing instability without obvious overfitting.
- Risk-adjusted momentum, medium-term trend confirmation, and longer low-vol windows are promising search directions.
- Do not move an unaudited `strategy.py` variant into `main` just because it improved `score_inner`; only process learnings belong on `main` by default.

Do not spend time on:

- same-day execution tricks
- visible-test overfitting
- pretending uncertain PIT data are trustworthy
- negative weights or synthetic shorting; V1 is explicit-cash only

## Results Logging

Use two logs, not one overloaded state machine:

- `results.tsv` for inner-search states like `inner_keep`, `inner_discard`, and `promote_outer`
- `~/.cache/qlab/audit_registry.tsv` for auditor-only states like `audit_pass`, `audit_fail`, and pending human review

Recommended inner-search schema:

```tsv
commit	score_inner	active_sharpe_daily	turnover	status	description
```
