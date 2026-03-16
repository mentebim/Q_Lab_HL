# AgentHL

This folder represents the `AgentHL` branch.

That branch is the bounded research service.

## Structure

- `inputs/`: what the research branch consumes
- `process/`: how bounded autoresearch works
- `outputs/`: what the branch publishes downstream

## Responsibility

- read trusted market data from `data`
- search inside the approved strategy family
- run the express filter
- run the full judge on survivors
- publish machine-readable research artifacts

## Inputs

- trusted market cache from `data`
- research policy
- candidate template
- active candidate config
- strategy family boundary
- fixed judge semantics
- prior leaderboard and result history

## Outputs

- result artifacts
- leaderboard rows
- promotion-eligibility metadata

## Non-Goals

- do not promote champions
- do not execute trades
- do not weaken the judge
- do not own data freshness

## Handoff

This branch feeds `Promotion` through research artifacts only.

Start with:

- [AgentHL/inputs/research_policy.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/AgentHL/inputs/research_policy.md)
- [AgentHL/process/autoresearch_loop.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/AgentHL/process/autoresearch_loop.md)
- [AgentHL/outputs/result_artifacts.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/AgentHL/outputs/result_artifacts.md)
