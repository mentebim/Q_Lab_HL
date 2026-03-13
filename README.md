# Q_Lab_HL Promotion

This branch is the promotion layer only.

Its job is narrow:

1. read synced research artifacts from `AgentHL`
2. apply promotion policy
3. pin `champion.paper.json` or `champion.live.json`

It does not run research.
It does not own market data.
It does not execute trades.

## Inputs

- `autoresearch/leaderboard.jsonl`
- `autoresearch/results/*.json`
- `autoresearch/promotion_policy.json`
- existing champion files under `execution/`

## Outputs

- `execution/champion.paper.json`
- `execution/champion.live.json`

## Core Command

Promote the paper champion:

```bash
python3 -m execution.select_champion --stage paper --out execution/champion.paper.json
```

Promote the live champion:

```bash
python3 -m execution.select_champion --stage live --out execution/champion.live.json
```

## Contract

Promotion only accepts artifacts that are:

- accepted by the fixed judge
- express-filter passed
- marked promotion-eligible
- present on disk as valid result artifacts

Live promotion may additionally require matching the current paper champion candidate.

## Tests

Run:

```bash
python3 -m unittest discover -s tests
```
