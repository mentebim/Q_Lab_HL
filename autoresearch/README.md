# Autoresearch Layout

This directory is the thin machine-operable layer on top of the fixed Hyperliquid statistical research harness.

## Files

- `config.example.json`: example bounded experiment spec
- `config.agent.json`: default real-data candidate spec to copy and mutate
- `leaderboard.jsonl`: append-only experiment summary log
- `results/`: detailed per-run JSON outputs, ignored by git

## Intended Loop

1. Read `RESEARCH_PROMPT.md` and `AUTORESEARCH_RULES.md`.
2. Start from `config.agent.json`.
3. Run `python3 autoresearch.py --config autoresearch/config.agent.json`.
3. Parse the emitted JSON.
4. Inspect `leaderboard.jsonl` and the latest file under `results/`.
5. Mutate the candidate config or strategy.
6. Decide whether to keep iterating, commit, or abandon the branch.

The bundled `config.agent.json` is the default real-data experiment: evaluate the current model on the hourly parquet cache, record the result, and keep the experiment metadata in flat files.

## Design Notes

- The harness remains the judge.
- `strategy.py` and `strategy_model.py` remain the main code mutation surface.
- `config.agent.json` is the normal place to change model spec, candidate id, and hypothesis without rewriting code.
- Results are flat files so they stay branch-local and easy to inspect.
- The leaderboard stores compact summaries.
- Detailed result JSON goes into `results/`.
