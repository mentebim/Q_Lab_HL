# data

This folder represents the `data` branch.

That branch is the central data engine of the system.

## Structure

- `inputs/`: upstream market information
- `process/`: ingestion, validation, and publication
- `outputs/`: the trusted cache contract

## Responsibility

- ingest Hyperliquid market data
- build and validate the trusted cache
- enforce freshness and schema checks
- publish the cache used by research and execution

## Inputs

- Hyperliquid candles
- Hyperliquid funding
- market metadata

## Outputs

- trusted market cache
- cache metadata
- freshness and validation status

## Non-Goals

- do not search for strategies
- do not promote champions
- do not trade

## Handoff

This branch feeds the same trusted cache to both `AgentHL` and `Execution`.

Start with:

- [data/inputs/hyperliquid_market.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/data/inputs/hyperliquid_market.md)
- [data/process/ingestion_and_validation.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/data/process/ingestion_and_validation.md)
- [data/outputs/trusted_cache.md](/Users/marcosentebi/Q_Lab_HL_deploy-winner1-on-main/data/outputs/trusted_cache.md)
