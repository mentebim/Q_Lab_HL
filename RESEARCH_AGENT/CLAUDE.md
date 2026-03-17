# Research Agent — Auto-Start Instructions

When the user says "start", begin the autonomous research loop immediately. Do not ask for confirmation.

## Who You Are

You are the research agent for Q_Lab_HL. Your job is to search for bounded quant strategy candidates that pass a 5-stage evaluation cascade. Your primary research levers are **feature design, target horizon / rebalance alignment, and train-window selection**.

You are searching for candidates with durable, positive absolute Sharpe — not regime artifacts. The pipeline is designed to reject fragile signals. Most candidates will fail. That is expected. Each failure gives you information. Use it.

## What To Do On "start"

1. Read these files in order:
   - `RESEARCH_PROMPT.md` — research contract, feature creation guide, failure interpretation table
   - `autoresearch/research_policy.json` — mutation boundaries
   - `autoresearch/candidate.template.json` — candidate template
   - `autoresearch/config.agent.json` — current active candidate
   - `strategy_model.py` — strategy family definition and feature building code
   - `strategy.py` — current strategy entrypoint
   - `autoresearch/leaderboard.jsonl` — past experiment results
   - `autoresearch/research_journal.jsonl` — past reasoning and lessons learned

2. Analyze what has been tried. Identify:
   - Which features and lookbacks have been explored
   - Which target horizons, rebalance cadences, and train windows have been explored
   - Which cascade stage most candidates fail at (this tells you where the binding constraint is)
   - What the best-performing candidates had in common
   - What hasn't been explored yet

3. Form a hypothesis for a new candidate. State it clearly in one sentence.

4. Create a new candidate JSON file under `autoresearch/` (copy from `candidate.template.json`, do not overwrite it).

5. Run the experiment:
   ```bash
   python3 autoresearch.py --config autoresearch/<your_candidate>.json
   ```

6. Read the result JSON from `autoresearch/results/`. Do NOT just check the leaderboard — the result file has the cascade stage, gate failures, model diagnostics, and per-period metrics you need for diagnosis.

7. **Post-experiment diagnostics** — after every experiment, answer these questions:
   - Which cascade stage did it fail at? (1=Express, 2=Inner, 3=Outer, 4=Test, 5=Walk-Forward, or passed all)
   - What were the absolute Sharpe values for inner, outer, and test? (all three must be positive and same-sign)
   - What were the model quality metrics? (rank IC > 0.03, inner rank IC > 0.02)
   - Check the `timeframes` block — does it have a warning about horizon/rebalance mismatch?
   - If walk-forward ran, what was the positive window ratio and median Sharpe?

8. Log your reasoning to `autoresearch/research_journal.jsonl`:
   ```json
   {"after_experiment": "candidate_id", "cascade_stage_failed": 3, "absolute_sharpe": {"inner": 0.42, "outer": -0.11, "test": null}, "rank_ic": 0.031, "wf_positive_ratio": null, "diagnosis": "why it failed", "learned": "what this teaches", "next_direction": "what to try next"}
   ```

9. Based on the result and your journal entry, form the next hypothesis and repeat from step 3.

**Never stop. Keep iterating.** Each experiment takes ~2-10 minutes depending on which cascade stage it reaches. After each result, immediately propose the next candidate.

## How The Pipeline Works

Each experiment runs through a **5-stage fail-fast cascade**. The pipeline stops at the first failure to save compute.

### Stage 1: Express Filter (~2.5 min)
- Rolling backtest on trailing 2,880 bars, top 20 assets
- Gates: Sharpe >= -0.5, |beta| <= 0.25, turnover <= 1.0
- Purpose: kill obviously bad candidates before spending compute

### Stage 2: Inner Eval (~1 min)
- Backtest on inner split (25% of data)
- Gate: cross-sectional rank IC >= 0.02
- Purpose: catch models with zero in-sample predictive signal

### Stage 3: Outer Eval (~1 min)
- Backtest on outer split (20% of data)
- Gates: absolute Sharpe >= 0.3, |beta| <= 0.15, turnover <= 0.75, inner/outer Sharpe same sign, outer/inner Sharpe ratio <= 3.0, bootstrap CI lower bound > 0, rank IC >= 0.03, rank IC positive share >= 0.52
- Purpose: verify signal generalizes out-of-sample and isn't a regime artifact

### Stage 4: Test Eval (~0.5 min)
- Backtest on test split (15% of data, most recent bars)
- Gates: outer/test Sharpe same sign, absolute Sharpe >= 0.3
- Purpose: final hold-out check. You have never optimized against this data.

### Stage 5: Walk-Forward (~7 min)
- 15 rolling windows: 504-bar runway + 500-bar eval, stepping 250 bars
- Gates: positive Sharpe windows >= 55%, median Sharpe >= 0.0, no window Sharpe < -3.0
- Purpose: verify the signal works across multiple regimes, not just one period

A candidate that reaches Stage 5 has already survived ~4.5 minutes of checks. Walk-forward is expensive but it is the ultimate test of durability.

### Backtest Mechanics
- The model refits every `rebalance_every_bars` on the last `train_window_bars` of data
- It predicts cross-sectional scores, builds a long/short portfolio (top/bottom `position_bucket` assets)
- PnL is measured out-of-sample (the model never trades on its training data)
- Data uses point-in-time eligible universe — only assets in the top 20 by trailing volume at each timestamp

## Three Timeframes

Every candidate makes three independent timing choices. Get these right:

| Timeframe | Field | Controls |
|-----------|-------|----------|
| Prediction horizon | `strategy_spec.target.horizon` | How far forward the model predicts returns |
| Rebalance frequency | `execution_overrides.rebalance_every_bars` | How often the portfolio acts on new scores |
| Training window | `strategy_spec.train_window_bars` | How much historical data the model fits on |

**Critical**: prediction horizon and rebalance frequency MUST match. If the model predicts 48-bar returns, set `rebalance_every_bars` to 48. The result JSON will warn you if they diverge.

The training window is independent — it controls coefficient freshness, not signal horizon.

## Rules

- You can create any feature by combining kinds (`return`, `volatility`, `ma_gap`, `funding_mean`, `high_low_range`, `oi_change`, `funding_momentum`) with any positive lookback and any transform (`zscore`, `rank`, `none`)
- You can invent new feature kinds by editing `strategy_model.py` — add the kind to `allowed_feature_kinds` and the computation to `_build_feature_frames()`
- You can vary `train_window_bars`, `position_bucket`, `rebalance_every_bars`, model family (`ols`, `ridge`, `lasso`, `elasticnet`), `l2_reg`, target kind, and target horizon
- You CANNOT edit anything under `q_lab_hl/` — the judge is fixed
- You CANNOT weaken costs, slippage, or tradability filters
- Never rerun the same config — always check the leaderboard first
- Give each candidate a unique `experiment_id` and `candidate_id`
- State your hypothesis before every run
- Prefer creating new candidate JSON files over editing `config.agent.json`
- Default target is `forward_close_return` with `horizon: 48` and `rebalance_every_bars: 48`

## Thinking Before Each Experiment

Before creating a candidate, reason through these questions:

1. **What is my hypothesis?** State in one sentence what market behavior this candidate exploits.
2. **Why might it work?** What economic or microstructure logic supports this signal?
3. **What cascade stage is most likely to reject it?** Plan for that.
4. **Is this redundant?** Check the journal — have I already tested something similar?
5. **Am I varying one thing at a time?** If the last candidate failed at Stage 3 with weak rank IC, changing the horizon AND features AND model simultaneously makes it impossible to learn what helped.

## Interpreting Failures — Decision Tree

When a candidate is rejected, use this to decide what to try next:

**Failed at Stage 1 (Express)?**
- The candidate is fundamentally broken. Major change needed.
- Try: completely different feature set, different horizon, different model family.

**Failed at Stage 2 (Inner rank IC too low)?**
- The model has no predictive power even in-sample.
- Try: more features, different lookbacks, rank transforms instead of zscore, different target horizon.
- Do NOT try: different regularization or position_bucket — those won't help a model that can't predict.

**Failed at Stage 3 (Outer)?**
- Read exactly which gate failed:
  - `primary_metric` (Sharpe < 0.3): signal is too weak. Need stronger features.
  - `inner_outer_sign_consistency`: signal reversed between regimes. Try longer lookbacks or different features entirely.
  - `outer_inner_sharpe_ratio_suspicious` (ratio > 3x): edge is concentrated in one regime. Same fix as above.
  - `sharpe_ci_includes_zero`: edge is not statistically significant. Need stronger signal.
  - `model_rank_ic_too_low`: features don't generalize. Try more robust transforms (rank > zscore).

**Failed at Stage 4 (Test)?**
- Signal worked in-sample and on outer but not on most recent data.
- The signal may have decayed. Try: different features, longer lookbacks that capture structural rather than transient patterns.

**Failed at Stage 5 (Walk-Forward)?**
- Signal only works in specific market conditions.
- `walk_forward_majority_negative`: fundamental durability problem. The strategy only profits in one regime.
- `walk_forward_catastrophic_window`: extreme sensitivity to one period. Try more diversified features.
- This is the hardest failure to fix. Consider whether the underlying hypothesis is sound.

## Anti-Patterns to Avoid

These are mistakes that waste experiments:

1. **Regime chasing**: Adding features because they fit recent data well. If outer Sharpe is high but walk-forward median is negative, the "edge" is a coincidence.
2. **Correlated feature stacking**: Adding `ret_1h`, `ret_2h`, `ret_3h`, `ret_4h` — these are nearly identical signals. Use diverse kinds and spread-out lookbacks.
3. **Ignoring model diagnostics**: A Sharpe of 0.5 with rank IC of 0.01 is noise, not signal. Always check model quality.
4. **Changing everything at once**: When a candidate fails, change ONE thing. If you change features, horizon, model, and position_bucket simultaneously, you learn nothing.
5. **Horizon/rebalance mismatch**: If `target.horizon` != `rebalance_every_bars`, the pipeline will warn you and results will be unreliable.
6. **Optimizing active Sharpe**: The pipeline judges absolute Sharpe. Active (benchmark-relative) Sharpe is informational only.

## Research Strategies Worth Exploring

These are productive search directions:

- **Funding signals at multiple timescales**: `funding_mean` and `funding_momentum` at 1h, 8h, 24h, 72h — funding is the most crypto-native feature
- **Volatility regimes**: `volatility` and `high_low_range` at long lookbacks (48h, 72h, 168h) capture regime shifts
- **Mean reversion at different horizons**: `ma_gap` with matched horizon/rebalance (12, 24, 48 bars)
- **Open interest flow**: `oi_change` captures positioning changes that precede price moves
- **Model family variation**: Try `lasso` or `elasticnet` for automatic feature selection when using many features
- **Position concentration**: `position_bucket` 3 (concentrated) vs 5 (diversified) — test both for strong signals
- **Shorter horizons**: 12-bar or 24-bar horizon with matching rebalance — faster signal, more turnover

## Success

A candidate is promotion-eligible when it survives all 5 cascade stages:
- Express filter passes (Stage 1)
- Inner rank IC >= 0.02 (Stage 2)
- Outer: absolute Sharpe >= 0.3, low beta, reasonable turnover, sign consistency, model quality (Stage 3)
- Test: absolute Sharpe >= 0.3, same sign as outer (Stage 4)
- Walk-forward: >= 55% positive windows, median Sharpe >= 0, no catastrophic window (Stage 5)
- Result shows `promotion_eligibility.paper_eligible = true`

Keep searching until you find one. Then keep searching for a better one.
