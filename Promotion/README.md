# Promotion

This folder represents the `Promotion` branch.

That branch is the conservative gate between research and trading.

## Structure

- `inputs/`: what promotion reads
- `process/`: how staged promotion works
- `outputs/`: what it publishes to execution

## Responsibility

- read accepted research artifacts
- apply staged promotion policy
- choose paper and live champions
- write pinned champion files

## Inputs

- leaderboard rows
- result artifacts
- promotion policy
- current paper and live champion state

## Outputs

- pinned paper champion
- pinned live champion

## Non-Goals

- do not search for new strategies
- do not change judge semantics
- do not execute trades
- do not own data ingestion

## Handoff

This branch consumes artifacts from `AgentHL` and feeds pinned champions to `execution`.

Start with:

- [Promotion/inputs/research_artifacts.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/Promotion/inputs/research_artifacts.md)
- [Promotion/process/promotion_flow.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/Promotion/process/promotion_flow.md)
- [Promotion/outputs/champions.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/Promotion/outputs/champions.md)
