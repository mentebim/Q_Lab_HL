# Q_Lab_HL AgentHL

This branch is the research layer only.

Its job is narrow:

1. define bounded candidates inside the approved strategy family
2. run the express filter
3. run the full fixed judge
4. write research artifacts for downstream promotion

It does not own champion promotion.
It does not own live execution.

## Branch Surface

- `RESEARCH_AGENT/`: agent entry bundle and workspace tooling
- `autoresearch/`: candidate configs, leaderboard, and result artifacts
- `strategy.py`: active strategy entrypoint
- `strategy_model.py`: approved strategy-family surface
- `q_lab_hl/`: fixed judge modules needed for research evaluation
- `data/market_cache_1h/`: local read-only market cache for evaluation

## Research Contract

This branch exists for:

- bounded strategy search
- fixed-judge evaluation
- artifact generation

It should output:

- `autoresearch/results/*.json`
- `autoresearch/leaderboard.jsonl`
- `promotion_eligibility` inside result artifacts

Those outputs are then handed to the `Promotion` branch.

## Core Commands

Run bounded autoresearch:

```bash
python3 autoresearch.py --config autoresearch/config.agent.json
```

Build the isolated research workspace:

```bash
python3 RESEARCH_AGENT/bootstrap_workspace.py --data-mode link --force
```

Dry-run sync back from the isolated workspace:

```bash
python3 RESEARCH_AGENT/sync_back.py
```

## Tests

Run:

```bash
python3 -m unittest discover -s tests
```
