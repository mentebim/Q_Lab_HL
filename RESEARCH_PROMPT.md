# Quant Autoresearch Prompt

## Mission

Develop bounded Hyperliquid quant strategy candidates inside the approved research contract of this repo.

The goal is not unrestricted creativity. The goal is correct search:

- narrow degrees of freedom
- fixed judge
- comparable results
- conservative promotion

## What The Agent Is Optimizing

The agent should search for candidates that achieve positive absolute risk-adjusted returns after costs, funding, and implementability realism.

The preferred judgment order is:

1. positive absolute Sharpe on both inner and outer periods
2. consistency: inner and outer absolute Sharpe should have the same sign
3. model quality: cross-sectional rank IC > 0.03, R² > 0 (however small)
4. lower beta drift
5. lower turnover when performance is similar
6. simpler hypotheses when performance is similar

## Research Contract

The repo expresses this loop only:

1. propose a bounded `CandidateSpec`
2. optionally pass an express filter
3. pass the full fixed judge
4. promote to paper only if accepted
5. promote to live only after paper validation

Do not turn the repo into a general trading framework or agent sandbox.

## Allowed Degrees Of Freedom

Default allowed mutation surface:

- `strategy_spec`
- selected `execution_overrides`
- bounded feature, target, transform, model, and train-window choices
- candidate metadata and notes under `autoresearch/`

Default disallowed mutation surface:

- judge mechanics under `q_lab_hl/`
- execution plumbing under `execution/`
- data ingestion semantics
- acceptance semantics designed to make weak candidates pass

## Approved Research Direction

Prefer bounded model-family search:

- cross-sectional linear models
- regularized linear models
- simple ranking-based transforms
- train-window and rebalance variations
- feature-set changes with a clear market hypothesis

Avoid open-ended arbitrary code generation unless the strategy family itself is being intentionally expanded by a human-reviewed change.

## Feature Creation Guide

You are encouraged to create new features. This is the primary research lever.

### Level 1: JSON-only features (no code change)

The four approved feature kinds can be combined freely with any positive lookback and any approved transform. Each combination is a distinct feature. Examples:

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

To create any of these, just add them to the `features` array in a candidate JSON. The `name` field is a free label — the `kind` and `lookback` determine what gets computed.

Feature kind reference (see `_build_feature_frames` in `strategy_model.py`):
- `return`: `close / close.shift(lookback) - 1` (price return over lookback bars)
- `volatility`: `returns_1h.rolling(lookback).std()` (rolling hourly return stdev)
- `ma_gap`: `close / close.rolling(lookback).mean() - 1` (distance from moving average)
- `funding_mean`: `funding.rolling(lookback).mean()` (rolling average funding rate)

Transform reference:
- `zscore`: mean-zero, unit-variance cross-sectionally, then clip at `clip` (default 3.0)
- `rank`: percentile rank cross-sectionally, centered at 0 (range -0.5 to +0.5)
- `none`: raw value, no standardization

You can use 1 to 12 features per candidate. Prefer 3–6 features with distinct lookbacks and kinds over many correlated features.

### Level 2: New feature kinds (requires Python edit)

If you have a hypothesis that needs a feature kind beyond the four above, you may:

1. Add the new kind string to `allowed_feature_kinds` in `LINEAR_CROSS_SECTION_FAMILY` inside `strategy_model.py`
2. Add the computation to `_build_feature_frames()` in `strategy_model.py`
3. Use it in your candidate JSON like any other feature

Both files are in the editable surface. Keep the computation simple — it should be a pandas rolling/shift operation on the existing market panels (close, open, high, low, volume, funding). Do not introduce external data or lookahead.

### Research hints

- The existing baseline uses zscore transforms on most features. Try rank transforms — they are more robust to outliers in crypto.
- Longer lookbacks (48h, 72h, 168h) capture different regimes than the default 1h/6h/24h.
- Funding is the most crypto-native signal. Explore it at multiple timescales.
- `position_bucket` controls how many assets you go long/short. With 20 tradable assets, pb=3 means 3L/3S (concentrated), pb=5 means 5L/5S (diversified). This directly affects gross exposure and portfolio diversification. Choose it deliberately.
- The judge penalizes instability (rolling Sharpe IQR). Simpler, more stable features tend to score better than complex ones.

## Integrity Rules

- No lookahead
- No same-bar execution
- No weakening costs, funding, or tradability filters
- No hidden changes to the judge
- No promotion based on in-sample wins alone
- Do not promote directly to champion without a valid result artifact
- Prefer mutating candidate JSON over rewriting Python
- Do not treat active (benchmark-relative) Sharpe as a success metric for market-neutral strategies
- Check model diagnostics (R², rank IC) after every run — if the model has no predictive power, the result is noise regardless of Sharpe
- Compare inner and outer absolute Sharpe — if they have opposite signs, the result is a regime artifact

If a candidate only looks good after changing the evaluator, treat it as invalid.
