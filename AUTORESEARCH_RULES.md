# Autoresearch Rules

## Default Edit Scope

Agents may edit by default:

- `strategy.py`
- `strategy_model.py`
- `RESEARCH_PROMPT.md`
- files under `autoresearch/`

Agents must treat as fixed unless explicitly authorized:

- `q_lab_hl/`
- `run.py`
- tests unrelated to the change being made
- evaluation, execution, cost, and portfolio rules

## Standard Experiment Command

Run bounded experiments through:

```bash
python3 autoresearch.py --config autoresearch/config.agent.json
```

Useful overrides:

```bash
python3 autoresearch.py --config autoresearch/config.agent.json --candidate-id cand_x --experiment-id exp_x
python3 autoresearch.py --config autoresearch/config.agent.json --no-append-leaderboard
python3 autoresearch.py --config autoresearch/config.agent.json --strategy-path strategy.py
```

## What Metrics Matter

- `periods.outer.active_sharpe_annualized` or `periods.test.active_sharpe_annualized`
- `periods.outer.beta_to_market`
- `periods.outer.turnover`

If no outer/test run is present, use inner metrics only for provisional filtering, not final judgment.

## Acceptance Standard

An experiment is acceptable only when:

- hard thresholds in the config are satisfied
- the primary metric meets its minimum
- any requested reference comparison is met
- there is no evidence that the strategy only improved by weakening the judge

Status meanings:

- `accepted`: passes configured thresholds and any requested comparison
- `rejected`: fails a hard threshold or comparison
- `needs_review`: thresholds pass but a required reference comparison is missing

## Required Artifacts

Each leaderboard entry should carry:

- timestamp
- experiment id
- candidate id
- strategy hash
- git commit if available
- hypothesis
- experiment spec
- key metrics
- status

Each result file should be machine-readable JSON.

## Anti-Cheating Rule

Do not modify the backtest or evaluation core as part of normal alpha iteration.
