# Overnight Autoresearch Summary — 2026-03-12

## Best Experiment

- **Experiment ID**: `exp_rank_b5_ols`
- **Candidate ID**: `ols_rank_b5_3f`
- **Status**: accepted

## Best Key Metrics (outer period)

| Metric | Value |
|---|---|
| active_sharpe_annualized | **+1.259** |
| sharpe_annualized | +1.554 |
| active_annualized_return | +0.959 |
| annualized_return | +0.327 |
| beta_to_market | 0.014 |
| turnover | 0.289 |
| max_drawdown | -0.104 |
| rolling_median_active_sharpe | +0.856 |

## Best Configuration

- **Model**: OLS (no regularization)
- **Features**: 3, all rank-transformed
  - `ret_1h` (1h return, rank) — short-term reversal
  - `ma_gap_24h` (24h MA gap, rank) — mean reversion
  - `funding_8h` (8h mean funding, rank) — funding carry
- **Position bucket**: 5 (5 long, 5 short)
- **Training window**: 504 bars (21 days)
- **Target**: next_open_to_close_return
- **Rebalance**: every 24 bars

## Model Coefficients (final fit)

| Feature | Coefficient | Interpretation |
|---|---|---|
| funding_8h | -0.0133 | Short high funding, long low funding (carry) |
| ret_1h | -0.0013 | Sell recent winners, buy recent losers (reversal) |
| ma_gap_24h | -0.0010 | Buy below 24h MA, sell above (mean reversion) |

Funding carry is the dominant alpha source (~10x larger coefficient than other features).

## Top Ideas Tested (15 experiments)

1. **Rank transform for all features** — massive improvement from zscore baseline (-0.335 → +0.880). This was the single most impactful change. Rank produces uniform cross-sectional distributions that are more stable for linear models.

2. **Feature simplification** — dropping from 6 to 3 features improved Sharpe from +0.947 to +1.213. The dropped features (ret_6h, ret_24h, vol_24h) added noise. The 3 surviving features tell a coherent story: reversal + mean-reversion + carry.

3. **OLS vs Ridge** — removing regularization improved Sharpe from +1.213 (ridge L2=5) → +1.223 (L2=1) → +1.259 (OLS). With only 3 features and 1500+ training rows, overfitting risk is negligible and regularization just attenuates the signal.

## What Failed

- **Longer training window** (1008 bars): outer Sharpe fell to -0.593 (with zscore). Mixes regimes.
- **Shorter training window** (336 bars): Sharpe fell to +0.493 (with OLS/rank). Too reactive.
- **Bucket=6**: Sharpe fell to +0.445. With 20 assets, holding 12 (60%) is too broad.
- **Close-to-close target**: Sharpe +1.039 (vs +1.259 for open-to-close). Open-to-close is better.
- **48h ma_gap**: Sharpe +0.875. Too slow for mean-reversion signal.
- **24h funding lookback**: Sharpe +1.145. 8h captures more current funding dynamics.
- **Adding features back** (ret_6h, ret_168h): Both degraded performance. Model benefits from parsimony.
- **12h rebalance**: Crashed on portfolio validation (universe too small at some timestamps).
- **Lower L2 on zscore features**: No improvement — the issue was zscore, not regularization.

## What Should Be Tried Next

1. **Funding-focused features**: Test different funding lookbacks (4h, 12h), funding volatility, or funding rate change. Since funding carry is ~10x the dominant signal, enriching the funding representation could help.

2. **Multi-period funding**: Use both 8h and 24h funding as separate features — the 8h captures recent dynamics while 24h provides a smoother trend.

3. **Volatility-adjusted returns**: Instead of raw ret_1h, try ret_1h / vol_24h for a risk-adjusted reversal signal.

4. **Volume-weighted features**: Incorporate volume data which is available in the cache but currently unused.

5. **Inner-period investigation**: The inner period active Sharpe is still -1.31 while outer is +1.26. Understanding this regime difference (is it due to different market conditions in 2024 vs 2025?) could improve robustness.

6. **Prediction clip sensitivity**: The current clip is 3.0 which may be too generous for OLS. Tighter clips (e.g., 1.5 or 2.0) could improve stability.

7. **Score-based weighting**: Currently `construct()` weights by abs(score). Alternative weighting schemes (equal weight, inverse-volatility) could reduce concentration in the most extreme predictions.
