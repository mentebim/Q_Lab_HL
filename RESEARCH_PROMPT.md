# Quant Autoresearch Prompt

## Mission

Develop bounded Hyperliquid quant strategy candidates inside the approved research contract of this repo.

The goal is not unrestricted creativity. The goal is correct search:

- narrow degrees of freedom
- fixed judge
- comparable results
- conservative promotion

## What The Agent Is Optimizing

The agent should search for candidates that improve out-of-sample active risk-adjusted returns after costs, funding, and implementability realism.

The preferred judgment order is:

1. outer/test quality
2. stability across periods
3. lower beta drift
4. lower turnover when alpha quality is similar
5. simpler hypotheses when performance is similar

## Research Contract

The repo expresses this loop only:

1. propose a bounded `CandidateSpec`
2. optionally pass an express filter
3. pass the full fixed judge
4. promote to paper only if accepted
5. promote to live only after paper validation

Do not turn the repo into a general trading framework or agent sandbox.

## Allowed Degrees Of Freedom

Default allowed mutation surface:

- `strategy_spec`
- selected `execution_overrides`
- bounded feature, target, transform, model, and train-window choices
- candidate metadata and notes under `autoresearch/`

Default disallowed mutation surface:

- judge mechanics under `q_lab_hl/`
- execution plumbing under `execution/`
- data ingestion semantics
- acceptance semantics designed to make weak candidates pass

## Approved Research Direction

Prefer bounded model-family search:

- cross-sectional linear models
- regularized linear models
- simple ranking-based transforms
- train-window and rebalance variations
- feature-set changes with a clear market hypothesis

Avoid open-ended arbitrary code generation unless the strategy family itself is being intentionally expanded by a human-reviewed change.

## Integrity Rules

- No lookahead
- No same-bar execution
- No weakening costs, funding, or tradability filters
- No hidden changes to the judge
- No promotion based on in-sample wins alone
- Do not promote directly to champion without a valid result artifact
- Prefer mutating candidate JSON over rewriting Python

If a candidate only looks good after changing the evaluator, treat it as invalid.
