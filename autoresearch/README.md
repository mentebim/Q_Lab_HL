# Autoresearch Layer

This directory holds the human-owned research contract and the machine-operable candidate artifacts.

## Files

- `research_policy.json`: repo mission and mutation boundary
- `candidate.template.json`: template for new bounded candidates
- `config.agent.json`: default active candidate spec
- `leaderboard.jsonl`: append-only summary log
- `results/`: detailed experiment artifacts, ignored by git

## Intended Loop

1. Read `RESEARCH_PROMPT.md`.
2. Start from `candidate.template.json` or `config.agent.json`.
3. Run `python3 autoresearch.py --config <candidate>.json`.
4. Inspect the emitted result artifact and leaderboard row.
5. Check whether the candidate passed the express filter.
6. Mutate bounded candidate parameters, not the judge.
7. Hand off valid accepted and promotion-eligible artifacts to the `Promotion` branch.

## Design Intent

- The judge lives outside this directory.
- This directory should express research policy and candidate objects, not hidden evaluator changes.
- Most experiments should be JSON mutations inside an approved strategy family.
- The express filter is a cheap first-stage gate, not a replacement for the full judge.
- Results should remain branch-local and machine-readable.
