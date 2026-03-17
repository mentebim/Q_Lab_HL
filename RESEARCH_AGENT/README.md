# Research Agent Bundle

This folder is the minimal entry bundle for a research agent.

It is also the control plane for building a narrower level-2 research workspace.

It is not a second repo contract. The canonical files remain in the main repo.

## Agent Job

The agent searches for candidates that survive a 5-stage evaluation cascade (Express → Inner → Outer → Test → Walk-Forward). It judges on **absolute Sharpe**, not active Sharpe.

The agent should:

- propose bounded quant research candidates
- run the 5-stage cascade through the existing pipeline
- read result JSONs (not just leaderboard) to diagnose which stage and gate failed
- log structured reasoning to `autoresearch/research_journal.jsonl`
- change one variable at a time to learn from each experiment
- improve candidate specs without changing the fixed judge

The agent should not:

- modify execution behavior
- modify cache ingestion semantics
- weaken the evaluator
- promote directly to champions
- optimize for active (benchmark-relative) Sharpe

## Load Order

1. `README.md`
2. `RESEARCH_PROMPT.md`
3. `autoresearch/research_policy.json`
4. `autoresearch/candidate.template.json`
5. `autoresearch/config.agent.json`
6. `strategy_model.py`
7. `strategy.py`
8. `autoresearch/leaderboard.jsonl`
9. `autoresearch/research_journal.jsonl`

## Editable Surface

- `autoresearch/config.agent.json`
- new candidate JSON files under `autoresearch/`
- approved parts of `strategy.py`
- approved parts of `strategy_model.py`

## Fixed Surface

- `q_lab_hl/`
- `data/active_1h/`
- `data/archive_1h/`

## Excluded Surface

- `execution/`
- runtime state and log files
- champion files

## Workspace Commands

Build the isolated research workspace:

```bash
python3 RESEARCH_AGENT/bootstrap_workspace.py
```

Sync approved research changes back into the main repo:

```bash
python3 RESEARCH_AGENT/sync_back.py --apply
```

## Standard Command

```bash
python3 autoresearch.py --config autoresearch/config.agent.json
```

## Outputs To Inspect

- `autoresearch/results/`
- `autoresearch/leaderboard.jsonl`

Only accepted and promotion-eligible results should flow into paper/live promotion.
