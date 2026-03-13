# Promotion Outputs

This directory holds the champion files and selector used by the promotion layer.

## Files

- `champion.paper.json`: pinned paper promotion target
- `champion.live.json`: pinned live promotion target
- `select_champion.py`: promote a paper or live champion from leaderboard results

## Promotion Commands

Promote the paper champion:

```bash
python -m execution.select_champion --stage paper --out execution/champion.paper.json
```

Promote the live champion:

```bash
python -m execution.select_champion --stage live --out execution/champion.live.json
```

## What This Branch Does Not Do

- no research generation
- no market data refresh
- no live execution
- no order routing
