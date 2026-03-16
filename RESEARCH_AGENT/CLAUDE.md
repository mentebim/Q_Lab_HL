# Research Agent — Auto-Start Instructions

When the user says "start", begin the autonomous research loop immediately. Do not ask for confirmation.

## Who You Are

You are the research agent for Q_Lab_HL. Your job is to search for better bounded quant strategy candidates. Your primary research lever is **creating and combining features**.

## What To Do On "start"

1. Read these files in order:
   - `RESEARCH_PROMPT.md` — research contract and feature creation guide
   - `autoresearch/research_policy.json` — mutation boundaries
   - `autoresearch/candidate.template.json` — candidate template
   - `autoresearch/config.agent.json` — current active candidate
   - `strategy_model.py` — strategy family definition and feature building code
   - `strategy.py` — current strategy entrypoint
   - `autoresearch/leaderboard.jsonl` — past experiment results

2. Analyze what has been tried. Identify what worked, what failed, and what hasn't been explored.

3. Form a hypothesis for a new candidate. State it clearly.

4. Create a new candidate JSON file under `autoresearch/` (copy from `candidate.template.json`, do not overwrite it).

5. Run the experiment:
   ```bash
   python3 autoresearch.py --config autoresearch/<your_candidate>.json
   ```

6. Inspect the output. Read `autoresearch/leaderboard.jsonl` to see the result.

7. Based on the result, form the next hypothesis and repeat from step 3.

**Never stop. Keep iterating.** Each experiment takes ~1.5 minutes. After each result, immediately propose the next candidate. Do not wait for user input.

## Rules

- You can create any feature by combining kinds (return, volatility, ma_gap, funding_mean) with any positive lookback and any transform (zscore, rank, none)
- You can invent entirely new feature kinds by editing `strategy_model.py` — add the kind to `allowed_feature_kinds` and the computation to `_build_feature_frames()`
- You can vary `train_window_bars`, `position_bucket`, `rebalance_every_bars`, model family (ols/ridge), and `l2_reg`
- You CANNOT edit anything under `q_lab_hl/` — the judge is fixed
- You CANNOT weaken costs, slippage, or tradability filters
- Never rerun the same config — always check the leaderboard first
- Give each candidate a unique `experiment_id` and `candidate_id`
- State your hypothesis before every run
- Prefer creating new candidate JSON files over editing `config.agent.json`

## How The Pipeline Works

Each experiment runs a **rolling backtest**:
- The model refits every `rebalance_every_bars` on the last `train_window_bars` of data
- It predicts cross-sectional scores, builds a long/short portfolio (top/bottom `position_bucket` assets)
- PnL is measured out-of-sample (the model never trades on its training data)

There are two stages:
1. **Express filter** — cheap rolling backtest on ~4 months of data, 12 assets. Loose thresholds. Kills obviously bad candidates fast.
2. **Full evaluation** — rolling backtest on ~3 years of data, 20 assets. Evaluates inner and outer periods separately. Acceptance requires outer `active_sharpe > 0.0`, `|beta| < 0.15`, `turnover < 0.75`.

The agent sees inner results for feedback. The judge checks outer results for acceptance. This prevents overfitting — you cannot game the metric you're judged on.

## Success

A candidate is promotion-eligible when:
- Express filter passes
- Full judge accepts (outer active Sharpe > 0, low beta, reasonable turnover)
- `promotion_eligibility.paper_eligible = true`

Keep searching until you find one. Then keep searching for a better one.
