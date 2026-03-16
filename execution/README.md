# execution

This folder represents the `Execution` production branch.

That branch is the runtime trading service that should be deployed.

## Structure

- `inputs/`: what execution consumes
- `process/`: runtime trading behavior
- `outputs/`: what execution emits

## Responsibility

- read pinned paper or live champions
- read trusted market data from `data`
- mechanically refit the pinned strategy on fresh data
- respect rebalance cadence
- reconcile venue state
- size positions from live account value and margin budget
- trade in paper or live mode

## Inputs

- trusted market cache
- paper champion
- live champion
- exchange credentials
- account address
- runtime state

## Outputs

- paper or live orders
- runtime logs
- reconciliation snapshots
- updated state

## Important Sizing Rule

All wallet capital can belong to the strategy.

Execution still sizes from a margin budget, not from a blind gross multiple. It should read live account value from Hyperliquid and preserve headroom through margin controls.

## Non-Goals

- do not search for new models
- do not promote champions
- do not own data ingestion

Start with:

- [execution/inputs/champions.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/execution/inputs/champions.md)
- [execution/process/runtime_loop.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/execution/process/runtime_loop.md)
- [execution/outputs/runtime_state.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/execution/outputs/runtime_state.md)
