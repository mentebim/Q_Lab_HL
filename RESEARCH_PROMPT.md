# Hyperliquid Stat-Arb Research Prompt

## Primary Objective

Develop trainable long/short Hyperliquid perpetual futures strategies that improve out-of-sample active risk-adjusted returns after fees, slippage, and funding while respecting the fixed execution model of this repo.

The default north-star metric is robust out-of-sample quality, not a single lucky in-sample Sharpe print.

## Secondary Metrics And Tradeoffs

- Higher cross-validation validation score is preferred to higher in-sample score.
- Higher active Sharpe is preferred when turnover, beta drift, and concentration stay controlled.
- Lower turnover is preferred when alpha quality is similar.
- Lower market beta drift is preferred when returns are similar.
- More stable performance across folds is preferred to fragile peak performance.
- Simpler hypotheses are preferred to complicated feature piles with weak evidence.

## Hard Constraints

- Execution is next-bar only. No same-bar fills, peeking, or lookahead.
- Funding, fees, and slippage are part of truth. Do not ignore cost drag.
- Tradability constraints are real. Do not rely on names that fail liquidity, history, price, or listing-cooldown filters.
- Long/short gross and net exposure controls are part of the strategy contract.
- Out-of-sample and cross-validation results matter more than in-sample performance.
- The harness under `q_lab_hl/` is the judge. Do not weaken it to make a strategy look better.

## Allowed Edit Scope

Default allowed mutation scope:

- `strategy.py`
- `strategy_model.py`
- autoresearch metadata such as experiment configs or notes when needed

Default fixed scope:

- `q_lab_hl/`
- `run.py`
- existing evaluation, audit, portfolio, and execution semantics

Only change the fixed harness when there is a concrete, defensible bug or missing capability in the judge itself, and document that separately from alpha iteration.

## Preferred Research Directions

- feature engineering on hourly market panels
- target definitions aligned with next-bar execution
- linear and regularized linear models before more complex families
- transformations that improve signal stability without leakage
- train-window choices that improve robustness
- simpler feature sets that hold up out of sample

## Anti-Goals And Failure Modes

- optimizing only to one split
- adding features or transforms without a bounded hypothesis
- increasing turnover to manufacture in-sample Sharpe
- relying on one asset or one cluster of highly correlated assets
- silently changing the objective by editing the evaluator
- overfitting feature transformations without a coherent market hypothesis

## Explicit Judge Integrity Warning

Do not cheat by modifying the judge.

That includes:

- loosening cost assumptions to rescue a weak strategy
- bypassing next-bar execution
- altering tradability filters to admit untradeable names
- hiding weak out-of-sample periods
- changing leaderboard logic to auto-accept weak candidates

If performance only appears after judge edits, treat that as invalid unless the edit is a genuine harness bug and is reviewed on its own merits.
