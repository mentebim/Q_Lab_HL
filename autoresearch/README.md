# Research Inputs

This branch does not generate research.

This directory exists only to receive synced research outputs from `AgentHL`.

## Files Used Here

- `leaderboard.jsonl`: summary of candidate results
- `results/`: detailed result artifacts, usually ignored by git
- `promotion_policy.json`: policy used by the promotion selector

## What Promotion Reads

Promotion expects each winning result artifact to contain:

- `acceptance`
- `express_filter`
- `promotion_eligibility`
- `result_path`
- strategy/spec metadata

## Handoff

`AgentHL` produces these files.

This branch reads them and converts eligible winners into champion files under `execution/`.
