# Q_Lab_HL

Q_Lab_HL is a constrained autoresearch repo for Hyperliquid quant strategies.

The repo is not a general trading sandbox. Its job is to support one loop only:

1. define a bounded candidate inside an approved strategy family
2. run a fast express filter
3. evaluate survivors with the fixed judge
4. promote only accepted candidates
5. execute only promoted champions in paper or live mode

## Architecture

- `autoresearch/`: human-owned research policy, candidate specs, leaderboard, and result artifacts
- `strategy.py`: current approved strategy entrypoint
- `strategy_model.py`: approved model-family implementation surface
- `q_lab_hl/`: fixed judge modules for data, backtest, and evaluation
- `execution/`: thin promotion and execution layer for pinned champions
- `data/market_cache_1h/`: local Hyperliquid market cache

## Repo Contract

The repo has three hard separations:

- fixed judge
  `q_lab_hl/data.py`, `q_lab_hl/backtest.py`, `q_lab_hl/evaluate.py`, and most of `q_lab_hl/config.py`
- bounded research surface
  candidate specs, approved strategy-family parameters, and limited strategy/model code
- gated execution
  only pinned champions are eligible for paper or live execution

Agents should not mutate the judge as part of normal research.

## Research Objects

The object model is moving toward explicit research contracts:

- `ResearchPolicy`: human-owned mission and mutation boundary
- `StrategyFamily`: approved model family and mutable parameter surface
- `CandidateSpec`: one bounded candidate for evaluation
- `ExpressFilterConfig`: fast first-stage gate before the full judge
- `AcceptancePolicy`: judge thresholds and comparison rules
- `RecordingConfig`: result and leaderboard output settings

Current defaults live in:

- [autoresearch/research_policy.json](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/autoresearch/research_policy.json)
- [autoresearch/candidate.template.json](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/autoresearch/candidate.template.json)
- [autoresearch/config.agent.json](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/autoresearch/config.agent.json)

## Allowed Degrees Of Freedom

Normal research should mostly mutate:

- `strategy_spec`
- selected `execution_overrides`
- approved model-family parameters
- candidate metadata in `autoresearch/`

Normal research should not mutate:

- data ingestion semantics
- backtest rules
- evaluation logic
- live execution plumbing

## Core Commands

Install dependencies:

```bash
python3 -m pip install -e .
```

Evaluate the current strategy on real data:

```bash
python3 run.py --evaluate --data-dir data/market_cache_1h --period outer --json --show-fit
```

Run bounded autoresearch from a candidate spec:

```bash
python3 autoresearch.py --config autoresearch/config.agent.json
```

Refresh the local cache:

```bash
python3 -m execution.update_cache --data-dir data/market_cache_1h
```

Run one paper execution cycle for the pinned champion:

```bash
python3 -m execution.run_live --champion execution/champion.paper.json
```

## Promotion Model

Promotion is intentionally conservative:

- candidate research result
- accepted result artifact
- pinned paper champion
- monitored paper behavior
- pinned live champion

Automatic execution is for promoted champions only.

## Current Roadmap

- Done: repo identity cleanup and explicit research object model
- Done: strategy family registry and bounded mutation contract
- Done: staged promotion pipeline cleanup
- Done: Phase 6 express filter for faster research throughput
- Next: final integration cleanup around research-to-promotion handoff

## Tests

Run the suite with:

```bash
python3 -m unittest discover -s tests
```
