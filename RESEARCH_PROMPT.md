# Quant Autoresearch Prompt

## Mission

Develop bounded Hyperliquid quant strategy candidates inside the approved research contract of this repo.

The goal is not unrestricted creativity. The goal is correct search:

- narrow degrees of freedom
- fixed judge
- comparable results
- conservative promotion

## What The Agent Is Optimizing

Search for candidates that achieve positive **absolute** risk-adjusted returns after costs, funding, and implementability realism.

The objective is **absolute Sharpe** — not active (benchmark-relative) Sharpe. Every gate in the pipeline measures absolute performance. Do not optimize for or report active Sharpe as a success metric.

The preferred judgment order is:

1. positive absolute Sharpe on inner, outer, **and test** periods
2. consistency: all three periods should have the same Sharpe sign
3. walk-forward stability: majority of rolling windows must be profitable
4. model quality: cross-sectional rank IC > 0.03, inner rank IC > 0.02
5. lower beta drift
6. lower turnover when performance is similar
7. simpler hypotheses when performance is similar

## Research Contract

The repo expresses this loop only:

1. propose a bounded `CandidateSpec`
2. pass the express filter (Stage 1)
3. pass the cascade evaluation stages (Stages 2-4)
4. pass walk-forward validation (Stage 5)
5. promote to paper only if accepted

Do not turn the repo into a general trading framework or agent sandbox.

## Evaluation Pipeline — 5-Stage Cascade

Every experiment runs through a fail-fast cascade. Each stage is a backtest that costs compute. The pipeline stops at the first failure to save time.

### Stage 1: Express Filter (~2.5 min)
- Backtests on trailing 2,880 bars (120 days), top 20 assets by dollar volume
- **Gates**: Sharpe >= -0.5, |beta| <= 0.25, turnover <= 1.0
- Kills obviously bad candidates before spending compute on full evaluation

### Stage 2: Inner Eval (~1 min)
- Backtests on inner split (25% of data, bars ~1866-3031)
- **Gate**: inner cross-sectional rank IC >= 0.02
- Catches models with no in-sample predictive signal

### Stage 3: Outer Eval (~1 min)
- Backtests on outer split (20% of data, bars ~3032-3963)
- **Gates** (checked only when the referenced data is available):
  - primary metric (absolute Sharpe) >= 0.3
  - |beta to market| <= 0.15
  - turnover <= 0.75
  - inner/outer Sharpe sign must match
  - outer/inner Sharpe ratio <= 3.0 (catches regime inflation)
  - bootstrap Sharpe CI lower bound > 0.0
  - outer rank IC >= 0.03, rank IC positive share >= 0.52

### Stage 4: Test Eval (~0.5 min)
- Backtests on test split (15% of data, most recent bars)
- **Gates**:
  - outer/test Sharpe sign must match
  - primary metric (default: `periods.test.sharpe_annualized`) >= 0.3
- This is the judged period. The agent has never optimized against this data.

### Stage 5: Walk-Forward (~7 min, only if Stages 1-4 pass)
- 15 rolling windows: 504-bar runway + 500-bar eval, stepping 250 bars
- **Gates**:
  - positive Sharpe windows >= 55% of all windows
  - median Sharpe across windows >= 0.0
  - no single window Sharpe < -3.0 (catastrophic drawdown)
- A candidate that only works in one regime will fail here.

### Result
Every experiment produces a result JSON in `autoresearch/results/` and a leaderboard entry in `autoresearch/leaderboard.jsonl`. The result includes a `timeframes` block showing prediction horizon, rebalance frequency, and train window — check that these are coherent.

## Three Timeframes

Every candidate makes three independent timing choices. Understand what each controls:

| Timeframe | Field | Controls |
|-----------|-------|----------|
| Prediction horizon | `strategy_spec.target.horizon` | How far forward the model predicts returns |
| Rebalance frequency | `execution_overrides.rebalance_every_bars` | How often the portfolio acts on new scores |
| Training window | `strategy_spec.train_window_bars` | How much historical data the model fits on |

**Critical rule**: prediction horizon and rebalance frequency should match. If the model predicts 48-bar returns, set `rebalance_every_bars` to 48. If they diverge, the result JSON will contain a warning in the `timeframes` block.

The training window is independent — it controls coefficient freshness, not signal horizon.

## Point-in-Time Data

The data uses a point-in-time architecture to prevent lookahead bias:

- `is_research_eligible.parquet` — boolean panel computed at build time. At each timestamp, only assets that were in the top 20 by trailing 14-day average dollar volume are marked eligible. This uses only historical data available at that point.
- `tradable.parquet` — boolean panel marking whether each asset was actively trading at each timestamp.
- The backtest wraps data in a `DateLimitedStore` at each rebalance, so your strategy only sees data up to the current timestamp.

You do not need to worry about lookahead — the harness enforces it. But be aware that the tradable universe changes over time. Early periods may have fewer eligible assets than recent periods.

## Allowed Degrees Of Freedom

Default allowed mutation surface:

- `strategy_spec`: features, target (kind + horizon), model family, regularization, train window, position bucket
- `execution_overrides`: `rebalance_every_bars` (should match target horizon)
- candidate metadata and notes under `autoresearch/`
- new feature kinds in `strategy_model.py`

Default disallowed mutation surface:

- judge mechanics under `q_lab_hl/`
- execution plumbing under `execution/`
- data ingestion semantics
- acceptance semantics designed to make weak candidates pass

## Approved Research Direction

Prefer bounded model-family search:

- cross-sectional linear models (OLS, Ridge, Lasso, ElasticNet)
- simple ranking-based transforms
- train-window and rebalance variations
- feature-set changes with a clear market hypothesis
- target horizon variations with matching rebalance frequency

Avoid open-ended arbitrary code generation unless the strategy family itself is being intentionally expanded by a human-reviewed change.

## Feature Creation Guide

You are encouraged to create new features. This is one of the primary research levers, alongside target horizon, rebalance cadence, and train-window selection.

### Level 1: JSON-only features (no code change)

Seven approved feature kinds can be combined freely with any positive lookback and any approved transform. Each combination is a distinct feature.

| name | kind | lookback | transform | hypothesis |
|------|------|----------|-----------|------------|
| ret_4h | return | 4 | rank | short-term mean reversion at 4h scale |
| ret_48h | return | 48 | zscore | 2-day momentum |
| vol_6h | volatility | 6 | zscore | intraday vol as risk signal |
| vol_72h | volatility | 72 | rank | 3-day vol regime ranking |
| ma_gap_6h | ma_gap | 6 | rank | fast mean reversion vs 6h MA |
| ma_gap_72h | ma_gap | 72 | zscore | medium-term dislocation |
| funding_1h | funding_mean | 1 | zscore | spot funding rate |
| funding_24h | funding_mean | 24 | rank | daily funding rank across coins |
| funding_72h | funding_mean | 72 | zscore | persistent funding pressure |
| hlr_24h | high_low_range | 24 | rank | daily price range as volatility proxy |
| oi_chg_24h | oi_change | 24 | zscore | daily open interest flow |
| fmom_48h | funding_momentum | 48 | rank | 2-day funding rate momentum |

To create any of these, just add them to the `features` array in a candidate JSON. The `name` field is a free label — the `kind` and `lookback` determine what gets computed.

Feature kind reference (see `_build_feature_frames` in `strategy_model.py`):
- `return`: `close / close.shift(lookback) - 1` (price return over lookback bars)
- `volatility`: `returns_1h.rolling(lookback).std()` (rolling hourly return stdev)
- `ma_gap`: `close / close.rolling(lookback).mean() - 1` (distance from moving average)
- `funding_mean`: `funding.rolling(lookback).mean()` (rolling average funding rate)
- `high_low_range`: `(high - low).rolling(lookback).mean() / close` (normalized average range)
- `oi_change`: open interest change over lookback bars
- `funding_momentum`: `funding.rolling(lookback).mean() - funding.rolling(lookback * 2).mean()` (funding rate trend)

Transform reference:
- `zscore`: mean-zero, unit-variance cross-sectionally, then clip at `clip` (default 3.0)
- `rank`: percentile rank cross-sectionally, centered at 0 (range -0.5 to +0.5)
- `none`: raw value, no standardization

You can use 1 to 12 features per candidate. Prefer 3-6 features with distinct lookbacks and kinds over many correlated features.

### Target kinds

Three target kinds are available:

| kind | formula | use case |
|------|---------|----------|
| `forward_close_return` | `close[t+h] / close[t] - 1` | **Default.** Predict return over the holding period. Set `horizon` to match `rebalance_every_bars`. |
| `next_open_to_close_return` | `close[t+h] / open[t+h] - 1` | Intra-bar return. Only use with matching short rebalance. |
| `next_close_to_close_return` | `close[t+h] / close[t] - 1` | Same as forward_close_return for horizon=1. |

The `horizon` field (default 1) sets how many bars forward the target looks. Example:
```json
"target": {"name": "fwd_48h_close", "kind": "forward_close_return", "horizon": 48}
```

### Level 2: New feature kinds (requires Python edit)

If you have a hypothesis that needs a feature kind beyond the seven above, you may:

1. Add the new kind string to `allowed_feature_kinds` in `LINEAR_CROSS_SECTION_FAMILY` inside `strategy_model.py`
2. Add the computation to `_build_feature_frames()` in `strategy_model.py`
3. Use it in your candidate JSON like any other feature

Both files are in the editable surface. Keep the computation simple — it should be a pandas rolling/shift operation on the existing market panels (close, open, high, low, volume, funding). Do not introduce external data or lookahead.

### Research hints

- Use `rank` transforms — they are more robust to outliers in crypto than zscore.
- Longer lookbacks (48h, 72h, 168h) capture different regimes than the default 1h/6h/24h.
- Funding is the most crypto-native signal. Explore it at multiple timescales.
- `position_bucket` controls how many assets you go long/short. With ~20 eligible assets, pb=3 means 3L/3S (concentrated), pb=5 means 5L/5S (diversified). Choose deliberately.
- The judge penalizes instability (rolling Sharpe IQR). Simpler, more stable features tend to score better.
- Vary `horizon` and `rebalance_every_bars` together. Try 12, 24, and 48 — the optimal holding period is a research question.
- A candidate with high outer Sharpe but negative walk-forward median is a regime artifact, not a signal.

## Interpreting Failures

When a candidate is rejected, the result JSON tells you exactly which gate failed. Use this to guide your next hypothesis:

| Failed check | What it means | What to try next |
|-------------|---------------|------------------|
| `inner_rank_ic_too_low` | Model has no predictive power in-sample | Different features, more features, different lookbacks |
| `model_rank_ic_too_low` | Outer period model has weak signal | Features may not generalize; try more robust transforms |
| `primary_metric` | Absolute Sharpe below 0.3 on test period | Stronger signal needed; check if inner/outer are also weak |
| `inner_outer_sign_consistency` | Model makes money in one regime, loses in another | Signal is not durable; try longer lookbacks or different features |
| `outer_inner_sharpe_ratio_suspicious` | Outer Sharpe > 3x inner — regime artifact | Same as above; the "edge" is concentrated in one time window |
| `sharpe_ci_includes_zero` | Bootstrap CI includes zero — edge is not statistically significant | Need stronger signal or more data |
| `outer_test_sign_consistency` | Strategy reverses between outer and test periods | Signal has decayed or was never real |
| `walk_forward_majority_negative` | Loses money in most rolling windows | Fundamental problem — the strategy only works in specific conditions |
| `walk_forward_median_sharpe_negative` | Median window is negative | Same — no durable edge |
| `walk_forward_catastrophic_window` | One window has Sharpe < -3.0 | Extreme regime sensitivity; consider more diversified features |

## Integrity Rules

- No lookahead
- No same-bar execution
- No weakening costs, funding, or tradability filters
- No hidden changes to the judge
- No promotion based on in-sample wins alone
- Do not promote directly to champion without a valid result artifact
- Prefer mutating candidate JSON over rewriting Python
- Do not treat active (benchmark-relative) Sharpe as a success metric — the pipeline judges absolute Sharpe only
- Check model diagnostics (R², rank IC) after every run — if the model has no predictive power, the result is noise regardless of Sharpe
- Compare absolute Sharpe across inner, outer, and test — if any pair has opposite signs, the result is a regime artifact
- Prediction horizon must match rebalance frequency — if they diverge, the result will contain a warning

If a candidate only looks good after changing the evaluator, treat it as invalid.
