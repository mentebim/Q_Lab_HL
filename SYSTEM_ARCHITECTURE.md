# System Architecture

This branch is the docs-only architecture map of the production system.

Each role is organized as `inputs/`, `process/`, and `outputs/` so the system can be understood by branch responsibility instead of by mixed code.

## 1. data

Responsibility:
- own ingestion
- own cache freshness and validation
- publish the trusted market dataset used by research and execution

Primary output:
- trusted market cache

Consumes:
- Hyperliquid market data

Feeds:
- `AgentHL`
- `Execution`

## 2. AgentHL

Responsibility:
- search inside the approved strategy family
- run express filter and full judge
- publish machine-readable research results

Primary outputs:
- result artifacts
- leaderboard rows
- promotion eligibility metadata

Consumes:
- trusted cache from `data`
- research policy
- strategy family contract
- fixed judge

Feeds:
- `Promotion`

## 3. Promotion

Responsibility:
- read accepted research artifacts
- apply staged promotion policy
- pin paper and live champions

Primary outputs:
- paper champion
- live champion

Consumes:
- research artifacts from `AgentHL`
- promotion policy
- current champion state

Feeds:
- `Execution`

## 4. Execution

Responsibility:
- read pinned champions
- read trusted data
- refit the pinned strategy on fresh data
- trade in paper or live mode

Primary outputs:
- orders
- logs
- state
- reconciliation records

Consumes:
- trusted cache from `data`
- champion files from `Promotion`
- live account state from Hyperliquid

## Important Runtime Principle

All wallet capital can belong to the winning strategy.

That does not mean execution should consume all margin immediately.

Execution should size from:
- live account value
- leverage map
- target margin usage ratio
- hard margin ceiling
- minimum headroom reserve

So the execution service uses all strategy capital as risk capital while still leaving liquidation and execution headroom.

## System In One Line

`data -> AgentHL -> Promotion -> Execution`
