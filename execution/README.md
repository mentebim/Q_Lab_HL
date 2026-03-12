# Execution Layer

This directory is the thin live-execution layer on top of the fixed research harness.

## Files

- `champion.json`: pinned live strategy spec
- `run_live.py`: run one hourly execution cycle
- `select_champion.py`: write `champion.json` from leaderboard/result files
- `update_cache.py`: refresh the trailing market-cache window
- `state.py`: idempotency and paper-position state
- `portfolio_live.py`: convert target weights into executable deltas
- `exchange_hl.py`: Hyperliquid paper/live venue wrapper
- `risk.py`: kill switch and pre-trade checks

## Recommended Hourly Flow

1. Refresh the trailing local cache:

```bash
python -m execution.update_cache --data-dir data/market_cache_1h
```

2. Pin the live champion explicitly:

```bash
python -m execution.select_champion --policy best-accepted --out execution/champion.json
```

3. Run one paper execution cycle:

```bash
python -m execution.run_live --champion execution/champion.json
```

4. Once paper logs look correct, switch `execution/champion.json` to `"mode": "live"` and set:

- `live.account_address`
- `live.secret_key_env`

Then rerun the same command.

## AWS deployment notes

For AWS research deployment, keep credentials out of git and inject them at runtime:

- set `HL_SECRET_KEY` as an environment secret
- set `live.account_address` in `execution/champion.json` or inject a generated champion file at deploy time
- keep `mode: "paper"` until you are ready to compare paper vs real
- recommended first live phase: small capital, same champion, same logs, compare realized live vs paper drift

## Cron / launchd

For hourly cron use two steps, not one:

```cron
2 * * * * cd /Users/marcosentebi/Q_Lab_HL && /users/marcosentebi/anaconda3/envs/vscode/bin/python -m execution.update_cache >> execution/update.log 2>&1
5 * * * * cd /Users/marcosentebi/Q_Lab_HL && /users/marcosentebi/anaconda3/envs/vscode/bin/python -m execution.run_live --champion execution/champion.json >> execution/live.log 2>&1
```

`launchd` is preferred on macOS once the paper loop is stable.

## Safety Defaults

- default mode is `paper`
- one champion only
- refuse to trade stale data
- refuse to trade the same signal bar twice unless `--force`
- kill switch file: `execution/STOP`
