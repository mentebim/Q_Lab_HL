# Execution Layer

This directory is the AWS/runtime trading surface.

## Files

- `champion.paper.json`: pinned paper promotion target
- `champion.live.json`: pinned live promotion target
- `run_live.py`: run one hourly execution cycle
- `select_champion.py`: champion validation and optional champion pinning helper
- `state.py`: idempotency and paper-position state
- `portfolio_live.py`: convert target weights into executable deltas
- `exchange_hl.py`: Hyperliquid paper/live venue wrapper
- `risk.py`: kill switch and pre-trade checks

## Runtime Flow

1. Read the pinned champion.
2. Validate the referenced result artifact.
3. Load the trusted local cache.
4. Refit the pinned strategy spec on fresh data.
5. Read actual account value from HL in live mode.
6. Derive gross exposure from margin-budget controls.
7. Reconcile venue state, build legal orders, and execute.

## Commands

Run one paper execution cycle:

```bash
python -m execution.run_live --champion execution/champion.paper.json
```

Run one live execution cycle:

```bash
python -m execution.run_live --champion execution/champion.live.json
```

Fill these fields before real trading:

- `live.account_address`
- `live.secret_key_env`

## AWS deployment notes

For AWS execution deployment, keep credentials out of git and inject them at runtime:

- set `HL_SECRET_KEY` as an environment secret
- set `live.account_address` in the champion files
- keep paper and live separated by config, state file, and log directory
- recommended first live phase: small capital, same champion, compare realized live vs paper drift
- if the whole wallet is dedicated to the strategy, still size from margin-budget controls instead of consuming 100% of available margin

## Cron / launchd

For hourly trading, run only the execution cycle against the already-refreshed trusted cache:

```cron
5 * * * * cd /Users/marcosentebi/Q_Lab_HL && /users/marcosentebi/anaconda3/envs/vscode/bin/python -m execution.run_live --champion execution/champion.live.json >> execution/live.log 2>&1
```

`launchd` is preferred on macOS once the paper loop is stable.

## Safety Defaults

- default mode is `paper`
- one paper champion and one live champion
- refuse to trade stale data
- read live account value from HL for sizing
- derive position size from margin usage instead of fixed wallet multiples
- refuse to trade the same signal bar twice unless `--force`
- kill switch file: `execution/STOP`
